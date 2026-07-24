use crate::{
    Problem, ScheduledBlock,
    collision::{BlockOrient, OrientPairCollision},
    params::InsertParams,
    precompute::Precompute,
    solver_util::normalized_imbalance,
    util::rand::Random,
};

#[cfg(feature = "profile-insert")]
use std::{
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

type Interval = (i64, i64);

#[cfg(feature = "profile-insert")]
#[derive(Default)]
struct InsertProfile {
    calls: u64,
    total: Duration,
    bay_setup: Duration,
    pair_cache: Duration,
    y_prepare: Duration,
    event_build: Duration,
    event_sort: Duration,
    event_apply: Duration,
    forbidden_build: Duration,
    feasible_time: Duration,
    candidate_eval: Duration,
    bays: u64,
    orientations: u64,
    ys: u64,
    x_positions: u64,
    events: u64,
    event_x_old_groups: u64,
    active_old_visits: u64,
    forbidden_intervals: u64,
    feasible_iterations: u64,
    feasible_interval_visits: u64,
    candidates: u64,
    min_entry_candidates: u64,
    ys_reaching_min_entry: u64,
    x_positions_after_min_entry: u64,
    x_range_width_sum: u64,
}

#[cfg(feature = "profile-insert")]
impl InsertProfile {
    fn merge(&mut self, other: Self) {
        self.calls += other.calls;
        self.total += other.total;
        self.bay_setup += other.bay_setup;
        self.pair_cache += other.pair_cache;
        self.y_prepare += other.y_prepare;
        self.event_build += other.event_build;
        self.event_sort += other.event_sort;
        self.event_apply += other.event_apply;
        self.forbidden_build += other.forbidden_build;
        self.feasible_time += other.feasible_time;
        self.candidate_eval += other.candidate_eval;
        self.bays += other.bays;
        self.orientations += other.orientations;
        self.ys += other.ys;
        self.x_positions += other.x_positions;
        self.events += other.events;
        self.event_x_old_groups += other.event_x_old_groups;
        self.active_old_visits += other.active_old_visits;
        self.forbidden_intervals += other.forbidden_intervals;
        self.feasible_iterations += other.feasible_iterations;
        self.feasible_interval_visits += other.feasible_interval_visits;
        self.candidates += other.candidates;
        self.min_entry_candidates += other.min_entry_candidates;
        self.ys_reaching_min_entry += other.ys_reaching_min_entry;
        self.x_positions_after_min_entry += other.x_positions_after_min_entry;
        self.x_range_width_sum += other.x_range_width_sum;
    }
}

#[cfg(feature = "profile-insert")]
static INSERT_PROFILE: OnceLock<Mutex<InsertProfile>> = OnceLock::new();

#[cfg(feature = "profile-insert")]
fn finish_insert_profile(mut profile: InsertProfile, start: Instant) {
    profile.total = start.elapsed();
    INSERT_PROFILE
        .get_or_init(|| Mutex::new(InsertProfile::default()))
        .lock()
        .unwrap()
        .merge(profile);
}

#[cfg(feature = "profile-insert")]
pub(crate) fn print_insert_profile() {
    let Some(profile) = INSERT_PROFILE.get() else {
        return;
    };
    let profile = profile.lock().unwrap();
    if profile.calls == 0 {
        return;
    }

    let total_seconds = profile.total.as_secs_f64();
    eprintln!(
        "[insert-profile] calls={} total={:.6}s avg={:.3}ms",
        profile.calls,
        total_seconds,
        total_seconds * 1000.0 / profile.calls as f64,
    );
    let phases = [
        ("bay_setup", profile.bay_setup),
        ("pair_cache", profile.pair_cache),
        ("y_prepare", profile.y_prepare),
        ("event_build", profile.event_build),
        ("event_sort", profile.event_sort),
        ("event_apply", profile.event_apply),
        ("forbidden", profile.forbidden_build),
        ("feasible_time", profile.feasible_time),
        ("candidate", profile.candidate_eval),
    ];
    let measured: Duration = phases.iter().map(|(_, duration)| *duration).sum();
    for (name, duration) in phases {
        let seconds = duration.as_secs_f64();
        eprintln!(
            "[insert-profile] {name:<14} {seconds:.6}s {:5.1}%",
            seconds / total_seconds * 100.0,
        );
    }
    let other = profile.total.saturating_sub(measured).as_secs_f64();
    eprintln!(
        "[insert-profile] {:<14} {:.6}s {:5.1}%",
        "other",
        other,
        other / total_seconds * 100.0,
    );
    eprintln!(
        "[insert-profile] counts bays={} orientations={} y={} x={} events={} event_x_old_groups={} active_old_visits={} forbidden_intervals={} feasible_iterations={} feasible_interval_visits={} candidates={} x_event_size={}",
        profile.bays,
        profile.orientations,
        profile.ys,
        profile.x_positions,
        profile.events,
        profile.event_x_old_groups,
        profile.active_old_visits,
        profile.forbidden_intervals,
        profile.feasible_iterations,
        profile.feasible_interval_visits,
        profile.candidates,
        std::mem::size_of::<XEvent>(),
    );
    eprintln!(
        "[insert-profile] opportunities min_entry_candidates={} ys_reaching_min_entry={} x_positions_after_min_entry={} x_range_width_sum={}",
        profile.min_entry_candidates,
        profile.ys_reaching_min_entry,
        profile.x_positions_after_min_entry,
        profile.x_range_width_sum,
    );
}

struct InsertCandidate {
    scheduled: ScheduledBlock,
    score_delta: f64,
    bbox_left: f64,
    bbox_right: f64,
    bbox_bottom: f64,
    bbox_top: f64,
}

#[derive(Clone, Copy)]
enum BBoxAnchor {
    RightTop,
    RightBottom,
    LeftTop,
    LeftBottom,
}

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

    #[cfg(not(feature = "profile-insert"))]
    fn first_feasible_time(&self, intervals: &[Interval], min_t: i64, max_t: i64) -> Option<i64> {
        self.first_feasible_time_impl(intervals, min_t, max_t, || {})
    }

    #[cfg(feature = "profile-insert")]
    fn first_feasible_time_profiled(
        &self,
        intervals: &[Interval],
        min_t: i64,
        max_t: i64,
        iterations: &mut u64,
        interval_visits: &mut u64,
        forbidden_intervals: &mut u64,
    ) -> Option<i64> {
        *iterations += 1;
        self.first_feasible_time_impl(intervals, min_t, max_t, || {
            *interval_visits += 1;
            *forbidden_intervals += 1;
        })
    }

    fn first_feasible_time_impl(
        &self,
        intervals: &[Interval],
        min_t: i64,
        max_t: i64,
        mut on_interval: impl FnMut(),
    ) -> Option<i64> {
        let mut t = min_t;
        for (summary_idx, &summary) in self.summary.iter().enumerate() {
            let mut summary = summary;
            while summary != 0 {
                let word_offset = summary.trailing_zeros() as usize;
                let word_idx = summary_idx * 64 + word_offset;
                let mut word = self.words[word_idx];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    let (l, r) = intervals[word_idx * 64 + bit];
                    on_interval();
                    if l > t {
                        return Some(t);
                    }
                    if t <= r {
                        if r == i64::MAX {
                            return None;
                        }
                        t = r + 1;
                        if t > max_t {
                            return None;
                        }
                    }
                    word &= word - 1;
                }
                summary &= summary - 1;
            }
        }
        Some(t)
    }
}

pub(crate) struct PlacementXScanner<'a> {
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
    #[cfg(feature = "profile-insert")]
    profile: InsertProfile,
}

impl<'a> PlacementXScanner<'a> {
    pub(crate) fn new(
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
            #[cfg(feature = "profile-insert")]
            profile: InsertProfile::default(),
        })
    }

    pub(crate) fn scan_y(
        &mut self,
        orient_idx: usize,
        y: i64,
        mut on_candidate: impl FnMut(ScheduledBlock) -> bool,
    ) -> bool {
        let Some(range) = self
            .pre
            .collision
            .fit_range(self.bay_id, self.block_id, orient_idx)
        else {
            return false;
        };
        if y < range.min_y || y > range.max_y {
            return false;
        }

        if self.prepared_orient_idx != Some(orient_idx) {
            #[cfg(feature = "profile-insert")]
            let pair_cache_start = Instant::now();
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
            #[cfg(feature = "profile-insert")]
            {
                self.profile.pair_cache += pair_cache_start.elapsed();
            }
        }

        #[cfg(feature = "profile-insert")]
        {
            self.profile.ys += 1;
            self.profile.x_range_width_sum += (range.max_x - range.min_x + 1) as u64;
        }
        #[cfg(feature = "profile-insert")]
        let mut y_x_positions = 0u64;
        #[cfg(feature = "profile-insert")]
        let mut first_min_entry_position = None;
        #[cfg(feature = "profile-insert")]
        let event_build_start = Instant::now();

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
                    range.min_x,
                    range.max_x,
                    &mut self.events,
                );
            }
            for &(lo, hi) in old_new_pair.crane.dx_intervals(y - old.y) {
                push_x_event(
                    old.x + lo,
                    old.x + hi,
                    old_idx,
                    2,
                    range.min_x,
                    range.max_x,
                    &mut self.events,
                );
            }
        }
        #[cfg(feature = "profile-insert")]
        {
            self.profile.event_build += event_build_start.elapsed();
            self.profile.events += self.events.len() as u64;
        }
        #[cfg(feature = "profile-insert")]
        let event_sort_start = Instant::now();
        counting_sort_x_events(
            &mut self.events,
            &mut self.event_sort_scratch,
            &mut self.event_counts,
            range.min_x,
            range.max_x,
        );
        #[cfg(feature = "profile-insert")]
        {
            self.profile.event_sort += event_sort_start.elapsed();
        }

        let mut valid_y = false;
        let mut event_pos = 0;
        let mut x_offset = 0u32;
        let mut x = range.min_x;
        loop {
            #[cfg(feature = "profile-insert")]
            {
                self.profile.x_positions += 1;
                y_x_positions += 1;
            }
            #[cfg(feature = "profile-insert")]
            let event_apply_start = Instant::now();
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
                    #[cfg(feature = "profile-insert")]
                    {
                        self.profile.event_x_old_groups += 1;
                    }
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
            #[cfg(feature = "profile-insert")]
            {
                self.profile.event_apply += event_apply_start.elapsed();
            }

            #[cfg(feature = "profile-insert")]
            let feasible_start = Instant::now();
            #[cfg(feature = "profile-insert")]
            let entry_time = self.active_slots.first_feasible_time_profiled(
                &self.forbidden_slots.intervals,
                self.min_t,
                self.max_t,
                &mut self.profile.feasible_iterations,
                &mut self.profile.feasible_interval_visits,
                &mut self.profile.forbidden_intervals,
            );
            #[cfg(not(feature = "profile-insert"))]
            let entry_time = self.active_slots.first_feasible_time(
                &self.forbidden_slots.intervals,
                self.min_t,
                self.max_t,
            );
            #[cfg(feature = "profile-insert")]
            {
                self.profile.feasible_time += feasible_start.elapsed();
            }

            if let Some(entry_time) = entry_time {
                #[cfg(feature = "profile-insert")]
                {
                    self.profile.candidates += 1;
                    if entry_time == self.min_t {
                        self.profile.min_entry_candidates += 1;
                        if first_min_entry_position.is_none() {
                            first_min_entry_position = Some(y_x_positions);
                            self.profile.ys_reaching_min_entry += 1;
                        }
                    }
                }
                let scheduled = ScheduledBlock {
                    block_id: self.block_id,
                    bay_id: self.bay_id,
                    orient_idx,
                    x,
                    y,
                    entry_time,
                    exit_time: entry_time + self.process_t,
                };
                #[cfg(feature = "profile-insert")]
                let candidate_start = Instant::now();
                valid_y |= on_candidate(scheduled);
                #[cfg(feature = "profile-insert")]
                {
                    self.profile.candidate_eval += candidate_start.elapsed();
                }
            }

            if event_pos >= self.events.len() {
                break;
            }
            x_offset = self.events[event_pos].x_offset;
            x = range.min_x + x_offset as i64;
            if x > range.max_x {
                break;
            }
        }

        #[cfg(feature = "profile-insert")]
        if let Some(position) = first_min_entry_position {
            self.profile.x_positions_after_min_entry += y_x_positions - position;
        }
        valid_y
    }
}

pub(crate) fn insert_greedy<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    min_entry_time: i64,
    max_entry_time: i64,
    schedule: &[ScheduledBlock],
    loads: &[f64],
    params: &InsertParams,
    reconstruct_random_strength: Option<f64>,
    bay_order: &[usize],
    rng: &mut R,
) -> Option<ScheduledBlock> {
    fn insert_candidate_better(
        a: &InsertCandidate,
        b: &InsertCandidate,
        anchor: BBoxAnchor,
    ) -> bool {
        let bbox_order = match anchor {
            BBoxAnchor::RightTop => a
                .bbox_right
                .total_cmp(&b.bbox_right)
                .then(a.bbox_top.total_cmp(&b.bbox_top)),
            BBoxAnchor::RightBottom => a
                .bbox_right
                .total_cmp(&b.bbox_right)
                .then(b.bbox_bottom.total_cmp(&a.bbox_bottom)),
            BBoxAnchor::LeftTop => b
                .bbox_left
                .total_cmp(&a.bbox_left)
                .then(a.bbox_top.total_cmp(&b.bbox_top)),
            BBoxAnchor::LeftBottom => b
                .bbox_left
                .total_cmp(&a.bbox_left)
                .then(b.bbox_bottom.total_cmp(&a.bbox_bottom)),
        };
        a.score_delta
            .total_cmp(&b.score_delta)
            .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
            .then(bbox_order)
            .then(a.scheduled.block_id.cmp(&b.scheduled.block_id))
            .is_lt()
    }

    #[cfg(feature = "profile-insert")]
    let profile_start = Instant::now();
    #[cfg(feature = "profile-insert")]
    let mut profile = InsertProfile {
        calls: 1,
        ..InsertProfile::default()
    };

    let block_id = original.block_id;
    let block = &problem.blocks[block_id];
    let process_t = block.processing_time;
    let min_t = block.release_time.max(min_entry_time);
    let max_t = max_entry_time;
    if min_t > max_t {
        #[cfg(feature = "profile-insert")]
        finish_insert_profile(profile, profile_start);
        return None;
    }

    let current_obj2 = normalized_imbalance(pre, loads);
    let original_tardiness = (original.exit_time - block.due_date).max(0);
    let anchor = match reconstruct_random_strength {
        Some(strength)
            if strength > 0.0
                && rng.nextf() < params.reconstruct_bbox_anchor_probability * strength =>
        {
            match rng.gen_index(4) {
                0 => BBoxAnchor::RightTop,
                1 => BBoxAnchor::RightBottom,
                2 => BBoxAnchor::LeftTop,
                _ => BBoxAnchor::LeftBottom,
            }
        }
        _ => BBoxAnchor::RightTop,
    };
    let mut best: Option<InsertCandidate> = None;

    for &bay_id in bay_order {
        if reconstruct_random_strength.is_some_and(|strength| {
            strength > 0.0
                && bay_id != original.bay_id
                && rng.nextf() < params.reconstruct_bay_skip_probability * strength
        }) {
            continue;
        }
        #[cfg(feature = "profile-insert")]
        {
            profile.bays += 1;
        }
        #[cfg(feature = "profile-insert")]
        let bay_setup_start = Instant::now();

        let mut next_loads = loads.to_vec();
        next_loads[bay_id] += block.workload as f64;
        // TODO: obj2は最後の方でだけ気にする
        let delta_obj23 = problem.weights.w2
            * (normalized_imbalance(pre, &next_loads) - current_obj2)
            + problem.weights.w3 * pre.pref_penalty[block_id][bay_id] as f64;
        let min_tardiness = min_t
            .saturating_add(process_t)
            .saturating_sub(block.due_date)
            .max(0);
        let lower_score_delta = problem.weights.w1 * min_tardiness as f64 + delta_obj23;
        if best
            .as_ref()
            .is_some_and(|best| lower_score_delta > best.score_delta)
        {
            #[cfg(feature = "profile-insert")]
            {
                profile.bay_setup += bay_setup_start.elapsed();
            }
            continue;
        }

        let mut scanner =
            PlacementXScanner::new(problem, pre, schedule, block_id, bay_id, min_t, max_t)?;
        #[cfg(feature = "profile-insert")]
        {
            profile.bay_setup += bay_setup_start.elapsed();
        }

        for &orient_idx in &pre.orientation_order_by_bbox[block_id] {
            if reconstruct_random_strength.is_some_and(|strength| {
                strength > 0.0
                    && (bay_id != original.bay_id || orient_idx != original.orient_idx)
                    && rng.nextf() < params.reconstruct_orientation_skip_probability * strength
            }) {
                continue;
            }
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            #[cfg(feature = "profile-insert")]
            {
                profile.orientations += 1;
            }
            let bounds = pre.orientation_bbox_bounds[block_id][orient_idx];

            #[cfg(feature = "profile-insert")]
            let y_prepare_start = Instant::now();
            let mut ys: Vec<i64> = (range.min_y..=range.max_y).collect();
            rng.shuffle(&mut ys);
            if let Some(strength) = reconstruct_random_strength {
                let original_y = (bay_id == original.bay_id
                    && orient_idx == original.orient_idx
                    && range.min_y <= original.y
                    && original.y <= range.max_y)
                    .then_some(original.y);
                let skip_probability = params.reconstruct_y_skip_probability * strength;
                if skip_probability > 0.0 {
                    ys.retain(|&y| Some(y) == original_y || rng.nextf() >= skip_probability);
                }
            }
            #[cfg(feature = "profile-insert")]
            {
                profile.y_prepare += y_prepare_start.elapsed();
            }
            let mut remaining_y_buffer = None;

            for y in ys {
                if remaining_y_buffer == Some(0) {
                    break;
                }

                let valid_y = scanner.scan_y(orient_idx, y, |scheduled| {
                    let tardiness = (scheduled.exit_time - block.due_date).max(0);
                    let score_delta = problem.weights.w1 * tardiness as f64 + delta_obj23;
                    let candidate = InsertCandidate {
                        scheduled,
                        score_delta,
                        bbox_left: scheduled.x as f64 + bounds.min_x,
                        bbox_right: scheduled.x as f64 + bounds.max_x,
                        bbox_bottom: scheduled.y as f64 + bounds.min_y,
                        bbox_top: scheduled.y as f64 + bounds.max_y,
                    };
                    if best.as_ref().map_or(true, |best| {
                        insert_candidate_better(&candidate, best, anchor)
                    }) {
                        best = Some(candidate);
                    }
                    tardiness <= original_tardiness
                });

                match &mut remaining_y_buffer {
                    None if valid_y => remaining_y_buffer = Some(params.y_buffer),
                    Some(remaining) => *remaining -= 1,
                    None => {}
                }
            }
        }
        #[cfg(feature = "profile-insert")]
        profile.merge(scanner.profile);
    }

    let result = best.map(|candidate| candidate.scheduled);
    #[cfg(feature = "profile-insert")]
    finish_insert_profile(profile, profile_start);
    result
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
    for (slot, (interval, old_idx, state_idx, item_idx)) in pending.into_iter().enumerate() {
        intervals.push(interval);
        states[old_idx][state_idx].slots[item_idx] = slot as u32;
    }
    ForbiddenSlotPrecompute { intervals, states }
}
