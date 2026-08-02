use crate::{Problem, ScheduledBlock};

use super::{
    collision::{BlockOrient, OrientPairCollision},
    precompute::Precompute,
};

pub(super) type Interval = (i64, i64);

#[derive(Clone, Copy)]
struct XEvent {
    x_offset: u32,
    old_idx: u32,
    update: i8,
}

#[derive(Clone, Copy, Default)]
struct HitState {
    new_old: u16,
    old_new: u16,
}

#[derive(Clone, Copy)]
struct OldTimeInfo {
    old_entry_time: i64,
    old_exit_time: i64,
    new_process_time: i64,
    overlap_entry_time_min: i64,
    overlap_entry_time_max: i64,
}

#[derive(Clone, Copy, Default)]
struct IntervalSet {
    intervals: [Interval; 2],
    len: u8,
}

impl IntervalSet {
    fn push(&mut self, interval: Interval) {
        self.intervals[self.len as usize] = interval;
        self.len += 1;
    }

    fn as_slice(&self) -> &[Interval] {
        &self.intervals[..self.len as usize]
    }
}

#[derive(Clone, Copy, Default)]
struct SlotSet {
    slots: [u32; 2],
    len: u8,
}

impl SlotSet {
    fn as_slice(&self) -> &[u32] {
        &self.slots[..self.len as usize]
    }
}

struct ForbiddenSlotPrecompute {
    intervals: Vec<Interval>,
    owners: Vec<u32>,
    states: Vec<[SlotSet; 4]>,
}

struct ActiveIntervalSlots {
    words: Vec<u64>,
    summary: Vec<u64>,
}

impl ActiveIntervalSlots {
    fn new(slot_count: usize) -> Self {
        let word_count = slot_count.div_ceil(64);
        Self {
            words: vec![0; word_count],
            summary: vec![0; word_count.div_ceil(64)],
        }
    }

    fn clear(&mut self) {
        self.words.fill(0);
        self.summary.fill(0);
    }

    fn set(&mut self, slot: usize, active: bool) {
        let word_idx = slot / 64;
        let bit = 1u64 << (slot % 64);
        let was_empty = self.words[word_idx] == 0;
        if active {
            self.words[word_idx] |= bit;
        } else {
            self.words[word_idx] &= !bit;
        }
        let is_empty = self.words[word_idx] == 0;
        if was_empty != is_empty {
            let summary_word = &mut self.summary[word_idx / 64];
            let summary_bit = 1u64 << (word_idx % 64);
            if is_empty {
                *summary_word &= !summary_bit;
            } else {
                *summary_word |= summary_bit;
            }
        }
    }

    fn set_all(&mut self, slots: SlotSet, active: bool) {
        for &slot in slots.as_slice() {
            self.set(slot as usize, active);
        }
    }

    fn iter<'a>(&'a self, intervals: &'a [Interval]) -> ActiveIntervalIter<'a> {
        ActiveIntervalIter {
            active: self,
            intervals,
            summary_idx: 0,
            summary_bits: 0,
            word_idx: 0,
            word_bits: 0,
        }
    }
}

struct ActiveIntervalIter<'a> {
    active: &'a ActiveIntervalSlots,
    intervals: &'a [Interval],
    summary_idx: usize,
    summary_bits: u64,
    word_idx: usize,
    word_bits: u64,
}

impl Iterator for ActiveIntervalIter<'_> {
    type Item = Interval;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.word_bits != 0 {
                let bit = self.word_bits.trailing_zeros() as usize;
                self.word_bits &= self.word_bits - 1;
                return self.intervals.get(self.word_idx * 64 + bit).copied();
            }
            if self.summary_bits != 0 {
                let bit = self.summary_bits.trailing_zeros() as usize;
                self.summary_bits &= self.summary_bits - 1;
                self.word_idx = (self.summary_idx - 1) * 64 + bit;
                self.word_bits = self.active.words.get(self.word_idx).copied().unwrap_or(0);
                continue;
            }
            self.summary_bits = *self.active.summary.get(self.summary_idx)?;
            self.summary_idx += 1;
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ForbiddenIntervals<'a> {
    active: &'a ActiveIntervalSlots,
    intervals: &'a [Interval],
    owners: &'a [u32],
}

impl ForbiddenIntervals<'_> {
    pub(super) fn first_feasible_time(
        self,
        base: &[Interval],
        min_t: i64,
        max_t: i64,
    ) -> Option<i64> {
        let mut t = min_t;
        let mut base_pos = base.partition_point(|&(_, right)| right < t);
        let mut added = self.active.iter(self.intervals).peekable();

        loop {
            let base_next = base.get(base_pos).copied();
            let added_next = added.peek().copied();
            let next = match (base_next, added_next) {
                (Some(base_interval), Some(added_interval)) => {
                    if base_interval <= added_interval {
                        base_pos += 1;
                        base_interval
                    } else {
                        added.next();
                        added_interval
                    }
                }
                (Some(base_interval), None) => {
                    base_pos += 1;
                    base_interval
                }
                (None, Some(added_interval)) => {
                    added.next();
                    added_interval
                }
                (None, None) => return (t <= max_t).then_some(t),
            };

            let (left, right) = next;
            if right < t {
                continue;
            }
            if left > t {
                return Some(t);
            }
            t = right.checked_add(1)?;
            if t > max_t {
                return None;
            }
        }
    }
}

pub(super) struct PlacementBlocker {
    pub block_id: usize,
    pub overlap_days: i64,
}

pub(super) struct PlacementTrace {
    pub earliest_entry: i64,
    pub blockers: Vec<PlacementBlocker>,
}

pub(super) struct PlacementXScanner<'a> {
    pre: &'a Precompute,
    block_id: usize,
    bay_id: usize,
    process_t: i64,
    min_t: i64,
    max_t: i64,
    bay_old_blocks: Vec<ScheduledBlock>,
    old_time_infos: Vec<Option<OldTimeInfo>>,
    forbidden_slots: ForbiddenSlotPrecompute,
    active_slots: ActiveIntervalSlots,
    events: Vec<XEvent>,
    event_sort_scratch: Vec<XEvent>,
    event_counts: Vec<usize>,
    states: Vec<HitState>,
    event_group_seen: Vec<u64>,
    event_group_generation: u64,
    touched_old_ids: Vec<usize>,
    crane_pair_cache: Vec<Option<(&'a OrientPairCollision, &'a OrientPairCollision)>>,
    prepared_orient_idx: Option<usize>,
}

impl<'a> PlacementXScanner<'a> {
    pub(super) fn new(
        problem: &Problem,
        pre: &'a Precompute,
        schedule: &[ScheduledBlock],
        block_id: usize,
        bay_id: usize,
        min_entry_time: i64,
        max_entry_time: i64,
    ) -> Option<Self> {
        let block = &problem.blocks[block_id];
        let process_t = block.processing_time;
        let min_t = block.release_time.max(min_entry_time);
        let max_t = max_entry_time;
        if min_t > max_t {
            return None;
        }
        let bay_old_blocks: Vec<_> = schedule
            .iter()
            .copied()
            .filter(|old| old.bay_id == bay_id)
            .collect();
        let old_time_infos: Vec<_> = bay_old_blocks
            .iter()
            .map(|&old| old_time_info(old, process_t, min_t, max_t))
            .collect();
        let forbidden_slots = build_forbidden_slot_precompute(&old_time_infos);
        let old_count = bay_old_blocks.len();
        Some(Self {
            pre,
            block_id,
            bay_id,
            process_t,
            min_t,
            max_t,
            bay_old_blocks,
            old_time_infos,
            active_slots: ActiveIntervalSlots::new(forbidden_slots.intervals.len()),
            forbidden_slots,
            events: Vec::with_capacity(old_count * 4),
            event_sort_scratch: Vec::with_capacity(old_count * 4),
            event_counts: Vec::new(),
            states: vec![HitState::default(); old_count],
            event_group_seen: vec![0; old_count],
            event_group_generation: 0,
            touched_old_ids: Vec::with_capacity(old_count),
            crane_pair_cache: Vec::with_capacity(old_count),
            prepared_orient_idx: None,
        })
    }

    pub(super) fn scan_y(
        &mut self,
        orient_idx: usize,
        y: i64,
        mut on_candidate: impl FnMut(ScheduledBlock) -> bool,
    ) -> bool {
        let block_id = self.block_id;
        let bay_id = self.bay_id;
        let process_t = self.process_t;
        let min_t = self.min_t;
        let max_t = self.max_t;
        self.scan_y_ranges(orient_idx, y, i64::MIN, i64::MAX, |x, forbidden| {
            let Some(entry_time) = forbidden.first_feasible_time(&[], min_t, max_t) else {
                return false;
            };
            on_candidate(ScheduledBlock {
                block_id,
                bay_id,
                orient_idx,
                x,
                y,
                entry_time,
                exit_time: entry_time + process_t,
            })
        })
    }

    pub(super) fn trace_fixed_placement(
        &mut self,
        orient_idx: usize,
        x: i64,
        y: i64,
        target_entry: i64,
    ) -> Option<PlacementTrace> {
        let min_t = self.min_t.max(target_entry);
        let max_t = self.max_t;
        if min_t > max_t {
            return None;
        }

        let mut earliest_entry = None;
        let mut overlap_by_owner = vec![0; self.bay_old_blocks.len()];
        self.scan_y_ranges(orient_idx, y, x, x, |_, forbidden| {
            let Some(entry) = forbidden.first_feasible_time(&[], min_t, max_t) else {
                return false;
            };
            earliest_entry = Some(entry);
            if entry == target_entry {
                return true;
            }

            let overlap_end = entry - 1;
            for slot in 0..forbidden.intervals.len() {
                if forbidden.active.words[slot / 64] & (1u64 << (slot % 64)) == 0 {
                    continue;
                }
                let (left, right) = forbidden.intervals[slot];
                let overlap_left = left.max(target_entry);
                let overlap_right = right.min(overlap_end);
                if overlap_left <= overlap_right {
                    overlap_by_owner[forbidden.owners[slot] as usize] +=
                        overlap_right - overlap_left + 1;
                }
            }
            true
        });

        let earliest_entry = earliest_entry?;
        let mut blockers: Vec<_> = overlap_by_owner
            .into_iter()
            .enumerate()
            .filter(|&(_, overlap_days)| overlap_days > 0)
            .map(|(old_idx, overlap_days)| PlacementBlocker {
                block_id: self.bay_old_blocks[old_idx].block_id,
                overlap_days,
            })
            .collect();
        blockers.sort_unstable_by(|a, b| {
            b.overlap_days
                .cmp(&a.overlap_days)
                .then_with(|| a.block_id.cmp(&b.block_id))
        });
        Some(PlacementTrace {
            earliest_entry,
            blockers,
        })
    }

    fn scan_y_ranges(
        &mut self,
        orient_idx: usize,
        y: i64,
        requested_min_x: i64,
        requested_max_x: i64,
        mut on_x: impl FnMut(i64, ForbiddenIntervals<'_>) -> bool,
    ) -> bool {
        let Some(fit_range) = self
            .pre
            .collision
            .fit_range(self.bay_id, self.block_id, orient_idx)
        else {
            return false;
        };
        if y < fit_range.min_y || y > fit_range.max_y {
            return false;
        }
        let min_x = fit_range.min_x.max(requested_min_x);
        let max_x = fit_range.max_x.min(requested_max_x);
        if min_x > max_x {
            return false;
        }

        if self.prepared_orient_idx != Some(orient_idx) {
            let new_orient = BlockOrient {
                block_id: self.block_id,
                orient_idx,
            };
            self.crane_pair_cache.clear();
            self.crane_pair_cache
                .extend(
                    self.bay_old_blocks
                        .iter()
                        .enumerate()
                        .map(|(old_idx, old)| {
                            if self.old_time_infos[old_idx].is_none() {
                                return None;
                            }
                            let old_orient = BlockOrient {
                                block_id: old.block_id,
                                orient_idx: old.orient_idx,
                            };
                            self.pre
                                .collision
                                .crane_pairs_both_directions(new_orient, old_orient)
                        }),
                );
            self.prepared_orient_idx = Some(orient_idx);
        }

        self.events.clear();
        self.states.fill(HitState::default());
        self.active_slots.clear();
        for (old_idx, &old) in self.bay_old_blocks.iter().enumerate() {
            if self.old_time_infos[old_idx].is_none() {
                continue;
            }
            let Some((new_old_pair, old_new_pair)) = self.crane_pair_cache[old_idx] else {
                continue;
            };
            for &(lo, hi) in new_old_pair.crane.dx_intervals(old.y - y) {
                push_x_event(
                    old.x - hi,
                    old.x - lo,
                    old_idx,
                    1,
                    min_x,
                    max_x,
                    &mut self.events,
                );
            }
            for &(lo, hi) in old_new_pair.crane.dx_intervals(y - old.y) {
                push_x_event(
                    old.x + lo,
                    old.x + hi,
                    old_idx,
                    2,
                    min_x,
                    max_x,
                    &mut self.events,
                );
            }
        }
        counting_sort_x_events(
            &mut self.events,
            &mut self.event_sort_scratch,
            &mut self.event_counts,
            min_x,
            max_x,
        );

        let mut accepted = false;
        let mut event_pos = 0;
        let mut x_offset = 0u32;
        let mut x = min_x;
        loop {
            self.event_group_generation += 1;
            self.touched_old_ids.clear();
            while event_pos < self.events.len() && self.events[event_pos].x_offset == x_offset {
                let event = self.events[event_pos];
                let old_idx = event.old_idx as usize;
                if self.event_group_seen[old_idx] != self.event_group_generation {
                    self.event_group_seen[old_idx] = self.event_group_generation;
                    self.touched_old_ids.push(old_idx);
                    self.active_slots.set_all(
                        self.forbidden_slots.states[old_idx][hit_state_index(self.states[old_idx])],
                        false,
                    );
                }
                apply_x_event(event, &mut self.states);
                event_pos += 1;
            }
            for &old_idx in &self.touched_old_ids {
                self.active_slots.set_all(
                    self.forbidden_slots.states[old_idx][hit_state_index(self.states[old_idx])],
                    true,
                );
            }

            accepted |= on_x(
                x,
                ForbiddenIntervals {
                    active: &self.active_slots,
                    intervals: &self.forbidden_slots.intervals,
                    owners: &self.forbidden_slots.owners,
                },
            );

            if event_pos >= self.events.len() {
                break;
            }
            x_offset = self.events[event_pos].x_offset;
            x = min_x + x_offset as i64;
            if x > max_x {
                break;
            }
        }

        accepted
    }
}

fn old_time_info(
    old: ScheduledBlock,
    process_t: i64,
    min_t: i64,
    max_t: i64,
) -> Option<OldTimeInfo> {
    let a = old.entry_time;
    let b = old.exit_time;
    let ol = (a - process_t + 1).max(min_t);
    let or = (b - 1).min(max_t);
    if ol <= or {
        Some(OldTimeInfo {
            old_entry_time: a,
            old_exit_time: b,
            new_process_time: process_t,
            overlap_entry_time_min: ol,
            overlap_entry_time_max: or,
        })
    } else {
        None
    }
}

fn push_x_event(
    l: i64,
    r: i64,
    old_idx: usize,
    update: i8,
    min_x: i64,
    max_x: i64,
    events: &mut Vec<XEvent>,
) {
    let l = l.max(min_x);
    let r = r.min(max_x);
    if l > r {
        return;
    }

    events.push(XEvent {
        x_offset: (l - min_x) as u32,
        old_idx: old_idx as u32,
        update,
    });
    if let Some(x) = r.checked_add(1) {
        events.push(XEvent {
            x_offset: (x - min_x) as u32,
            old_idx: old_idx as u32,
            update: -update,
        });
    }
}

fn counting_sort_x_events(
    events: &mut Vec<XEvent>,
    scratch: &mut Vec<XEvent>,
    counts: &mut Vec<usize>,
    min_x: i64,
    max_x: i64,
) {
    let bucket_count = (max_x - min_x + 2) as usize;
    counts.resize(bucket_count, 0);
    counts.fill(0);
    for event in events.iter() {
        counts[event.x_offset as usize] += 1;
    }

    let mut offset = 0;
    for count in counts.iter_mut() {
        let size = *count;
        *count = offset;
        offset += size;
    }

    scratch.resize(
        events.len(),
        XEvent {
            x_offset: 0,
            old_idx: 0,
            update: 0,
        },
    );
    for &event in events.iter() {
        let position = &mut counts[event.x_offset as usize];
        scratch[*position] = event;
        *position += 1;
    }
    std::mem::swap(events, scratch);
}

fn apply_x_event(event: XEvent, states: &mut [HitState]) {
    let state = &mut states[event.old_idx as usize];
    match event.update {
        1 => state.new_old += 1,
        -1 => {
            debug_assert!(state.new_old > 0);
            state.new_old -= 1;
        }
        2 => state.old_new += 1,
        -2 => {
            debug_assert!(state.old_new > 0);
            state.old_new -= 1;
        }
        _ => unreachable!(),
    }
}

fn hit_state_index(state: HitState) -> usize {
    usize::from(state.new_old > 0) | usize::from(state.old_new > 0) << 1
}

fn forbidden_interval_set(info: OldTimeInfo, new_old_hit: bool, old_new_hit: bool) -> IntervalSet {
    let mut result = IntervalSet::default();
    let new_old_clear = !new_old_hit;
    let old_new_clear = !old_new_hit;
    if new_old_clear && old_new_clear {
        return result;
    }

    let ol = info.overlap_entry_time_min;
    let or = info.overlap_entry_time_max;
    if !new_old_clear && !old_new_clear {
        result.push((ol, or));
        return result;
    }

    let (allow_l, allow_r) = if new_old_clear {
        (
            (info.old_entry_time + 1).max(ol),
            (info.old_exit_time - info.new_process_time - 1).min(or),
        )
    } else {
        (
            (info.old_exit_time - info.new_process_time + 1).max(ol),
            (info.old_entry_time - 1).min(or),
        )
    };

    if allow_l > allow_r {
        result.push((ol, or));
        return result;
    }
    if ol < allow_l {
        result.push((ol, allow_l - 1));
    }
    if allow_r < or {
        result.push((allow_r + 1, or));
    }
    result
}

fn build_forbidden_slot_precompute(
    old_time_infos: &[Option<OldTimeInfo>],
) -> ForbiddenSlotPrecompute {
    let mut states = vec![[SlotSet::default(); 4]; old_time_infos.len()];
    let mut pending = Vec::new();
    for (old_idx, &info) in old_time_infos.iter().enumerate() {
        let Some(info) = info else {
            continue;
        };
        for state_idx in 0..4 {
            let set = forbidden_interval_set(info, state_idx & 1 != 0, state_idx & 2 != 0);
            states[old_idx][state_idx].len = set.len;
            for (item_idx, &interval) in set.as_slice().iter().enumerate() {
                pending.push((interval, old_idx, state_idx, item_idx));
            }
        }
    }
    pending.sort_unstable_by_key(|&(interval, _, _, _)| interval);

    let mut intervals = Vec::with_capacity(pending.len());
    let mut owners = Vec::with_capacity(pending.len());
    for (slot, (interval, old_idx, state_idx, item_idx)) in pending.into_iter().enumerate() {
        intervals.push(interval);
        owners.push(old_idx as u32);
        states[old_idx][state_idx].slots[item_idx] = slot as u32;
    }
    ForbiddenSlotPrecompute {
        intervals,
        owners,
        states,
    }
}
