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

    fn reset(&mut self, slot_count: usize) {
        let word_count = slot_count.div_ceil(64);
        self.words.resize(word_count, 0);
        self.summary.resize(word_count.div_ceil(64), 0);
        self.clear();
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

pub(super) struct Scratch<'a> {
    bay_old_blocks: Vec<ScheduledBlock>,
    old_time_infos: Vec<Option<OldTimeInfo>>,
    forbidden_slots: ForbiddenSlotPrecompute,
    forbidden_slot_pending: Vec<(Interval, usize, usize, usize)>,
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

impl<'a> Scratch<'a> {
    pub(super) fn new() -> Self {
        Self {
            bay_old_blocks: Vec::new(),
            old_time_infos: Vec::new(),
            forbidden_slots: ForbiddenSlotPrecompute {
                intervals: Vec::new(),
                states: Vec::new(),
            },
            forbidden_slot_pending: Vec::new(),
            active_slots: ActiveIntervalSlots::new(0),
            events: Vec::new(),
            event_sort_scratch: Vec::new(),
            event_counts: Vec::new(),
            states: Vec::new(),
            event_group_seen: Vec::new(),
            event_group_generation: 0,
            touched_old_ids: Vec::new(),
            crane_pair_cache: Vec::new(),
            prepared_orient_idx: None,
        }
    }
}

pub(super) struct PlacementXScanner<'pre, 'scratch> {
    pre: &'pre Precompute,
    block_id: usize,
    bay_id: usize,
    process_t: i64,
    min_t: i64,
    max_t: i64,
    scratch: &'scratch mut Scratch<'pre>,
}

impl<'pre, 'scratch> PlacementXScanner<'pre, 'scratch> {
    pub(super) fn new(
        problem: &Problem,
        pre: &'pre Precompute,
        scratch: &'scratch mut Scratch<'pre>,
        bay_old_blocks: &[ScheduledBlock],
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

        scratch.bay_old_blocks.clear();
        scratch.bay_old_blocks.extend_from_slice(bay_old_blocks);
        scratch.old_time_infos.clear();
        scratch.old_time_infos.extend(
            scratch
                .bay_old_blocks
                .iter()
                .map(|&old| old_time_info(old, process_t, min_t, max_t)),
        );
        build_forbidden_slot_precompute(
            &scratch.old_time_infos,
            &mut scratch.forbidden_slots,
            &mut scratch.forbidden_slot_pending,
        );

        let old_count = scratch.bay_old_blocks.len();
        scratch
            .active_slots
            .reset(scratch.forbidden_slots.intervals.len());
        scratch.events.clear();
        scratch.event_sort_scratch.clear();
        scratch.event_counts.clear();
        scratch.states.resize(old_count, HitState::default());
        scratch.states.fill(HitState::default());
        scratch.event_group_seen.resize(old_count, 0);
        scratch.event_group_seen.fill(0);
        scratch.event_group_generation = 0;
        scratch.touched_old_ids.clear();
        scratch.crane_pair_cache.clear();
        scratch.prepared_orient_idx = None;

        Some(Self {
            pre,
            block_id,
            bay_id,
            process_t,
            min_t,
            max_t,
            scratch,
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

        let pre = self.pre;
        let block_id = self.block_id;
        let scratch = &mut *self.scratch;
        if scratch.prepared_orient_idx != Some(orient_idx) {
            let new_orient = BlockOrient {
                block_id,
                orient_idx,
            };
            scratch.crane_pair_cache.clear();
            for (old_idx, old) in scratch.bay_old_blocks.iter().enumerate() {
                let pair = if scratch.old_time_infos[old_idx].is_none() {
                    None
                } else {
                    let old_orient = BlockOrient {
                        block_id: old.block_id,
                        orient_idx: old.orient_idx,
                    };
                    pre.collision
                        .crane_pairs_both_directions(new_orient, old_orient)
                };
                scratch.crane_pair_cache.push(pair);
            }
            scratch.prepared_orient_idx = Some(orient_idx);
        }

        scratch.events.clear();
        scratch.states.fill(HitState::default());
        scratch.active_slots.clear();
        for (old_idx, &old) in scratch.bay_old_blocks.iter().enumerate() {
            if scratch.old_time_infos[old_idx].is_none() {
                continue;
            }
            let Some((new_old_pair, old_new_pair)) = scratch.crane_pair_cache[old_idx] else {
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
                    &mut scratch.events,
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
                    &mut scratch.events,
                );
            }
        }
        counting_sort_x_events(
            &mut scratch.events,
            &mut scratch.event_sort_scratch,
            &mut scratch.event_counts,
            min_x,
            max_x,
        );

        let mut accepted = false;
        let mut event_pos = 0;
        let mut x_offset = 0u32;
        let mut x = min_x;
        loop {
            scratch.event_group_generation += 1;
            scratch.touched_old_ids.clear();
            while event_pos < scratch.events.len() && scratch.events[event_pos].x_offset == x_offset
            {
                let event = scratch.events[event_pos];
                let old_idx = event.old_idx as usize;
                if scratch.event_group_seen[old_idx] != scratch.event_group_generation {
                    scratch.event_group_seen[old_idx] = scratch.event_group_generation;
                    scratch.touched_old_ids.push(old_idx);
                    scratch.active_slots.set_all(
                        scratch.forbidden_slots.states[old_idx]
                            [hit_state_index(scratch.states[old_idx])],
                        false,
                    );
                }
                apply_x_event(event, &mut scratch.states);
                event_pos += 1;
            }
            for &old_idx in &scratch.touched_old_ids {
                scratch.active_slots.set_all(
                    scratch.forbidden_slots.states[old_idx]
                        [hit_state_index(scratch.states[old_idx])],
                    true,
                );
            }

            accepted |= on_x(
                x,
                ForbiddenIntervals {
                    active: &scratch.active_slots,
                    intervals: &scratch.forbidden_slots.intervals,
                },
            );

            if event_pos >= scratch.events.len() {
                break;
            }
            x_offset = scratch.events[event_pos].x_offset;
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
    output: &mut ForbiddenSlotPrecompute,
    pending: &mut Vec<(Interval, usize, usize, usize)>,
) {
    output
        .states
        .resize(old_time_infos.len(), [SlotSet::default(); 4]);
    output.states.fill([SlotSet::default(); 4]);
    pending.clear();
    for (old_idx, &info) in old_time_infos.iter().enumerate() {
        let Some(info) = info else {
            continue;
        };
        for state_idx in 0..4 {
            let set = forbidden_interval_set(info, state_idx & 1 != 0, state_idx & 2 != 0);
            output.states[old_idx][state_idx].len = set.len;
            for (item_idx, &interval) in set.as_slice().iter().enumerate() {
                pending.push((interval, old_idx, state_idx, item_idx));
            }
        }
    }
    pending.sort_unstable_by_key(|&(interval, _, _, _)| interval);

    output.intervals.clear();
    for (slot, &(interval, old_idx, state_idx, item_idx)) in pending.iter().enumerate() {
        output.intervals.push(interval);
        output.states[old_idx][state_idx].slots[item_idx] = slot as u32;
    }
}
