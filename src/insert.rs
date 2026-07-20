use crate::{
    Problem, ScheduledBlock,
    collision::{BlockOrient, BlockPlacement, CollisionResult},
    params::InsertParams,
    precompute::Precompute,
    solver_util::normalized_imbalance,
    util::rand::Random,
};

type Interval = (i64, i64);

struct InsertCandidate {
    scheduled: ScheduledBlock,
    score_delta: f64,
    bbox_right: f64,
    bbox_top: f64,
}

#[derive(Clone, Copy)]
enum HitDir {
    NewOld,
    OldNew,
}

#[derive(Clone, Copy)]
struct XEvent {
    x: i64,
    old_idx: usize,
    dir: HitDir,
    delta: i8,
}

#[derive(Clone, Copy, Default)]
struct HitState {
    new_old: u16,
    old_new: u16,
}

impl HitState {
    fn is_active(self) -> bool {
        self.new_old > 0 || self.old_new > 0
    }
}

#[derive(Clone, Copy)]
struct OldTimeInfo {
    old_entry_time: i64,
    old_exit_time: i64,
    new_process_time: i64,
    overlap_entry_time_min: i64,
    overlap_entry_time_max: i64,
}

pub fn insert_greedy<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    min_entry_time: i64,
    max_entry_time: i64,
    schedule: &[ScheduledBlock],
    loads: &[f64],
    params: &InsertParams,
    bay_order: &[usize],
    rng: &mut R,
) -> Option<ScheduledBlock> {
    fn insert_candidate_better(a: &InsertCandidate, b: &InsertCandidate) -> bool {
        a.score_delta
            .total_cmp(&b.score_delta)
            .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
            .then(a.bbox_right.total_cmp(&b.bbox_right))
            .then(a.bbox_top.total_cmp(&b.bbox_top))
            .then(a.scheduled.block_id.cmp(&b.scheduled.block_id))
            .is_lt()
    }

    let block_id = original.block_id;
    let block = &problem.blocks[block_id];
    let process_t = block.processing_time;
    let min_t = block.release_time.max(min_entry_time);
    let max_t = max_entry_time;
    if min_t > max_t {
        return None;
    }

    let current_obj2 = normalized_imbalance(pre, loads);
    let original_tardiness = (original.exit_time - block.due_date).max(0);
    let mut best: Option<InsertCandidate> = None;

    for &bay_id in bay_order {
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
            continue;
        }

        let bay_old_blocks: Vec<ScheduledBlock> = schedule
            .iter()
            .copied()
            .filter(|old| old.bay_id == bay_id)
            .collect();
        let old_time_infos: Vec<Option<OldTimeInfo>> = bay_old_blocks
            .iter()
            .map(|&old| old_time_info(old, process_t, min_t, max_t))
            .collect();
        let mut events = Vec::with_capacity(bay_old_blocks.len() * 4);
        let mut states = vec![HitState::default(); bay_old_blocks.len()];
        let mut active_old_ids = Vec::with_capacity(bay_old_blocks.len());
        let mut active_pos = vec![None; bay_old_blocks.len()];
        let mut forbidden = Vec::with_capacity(16);

        for &orient_idx in &pre.orientation_order_by_bbox[block_id] {
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            let new_orient = BlockOrient {
                block_id,
                orient_idx,
            };
            let bounds = pre.orientation_bbox_bounds[block_id][orient_idx];
            let crane_pair_cache: Vec<_> = bay_old_blocks
                .iter()
                .enumerate()
                .map(|(old_idx, old)| {
                    if old_time_infos[old_idx].is_none() {
                        return None;
                    }
                    let old_orient = BlockOrient {
                        block_id: old.block_id,
                        orient_idx: old.orient_idx,
                    };
                    pre.collision
                        .crane_pairs_both_directions(new_orient, old_orient)
                })
                .collect();
            let mut ys: Vec<i64> = (range.min_y..=range.max_y).collect();
            rng.shuffle(&mut ys);
            let sample_count = (ys.len() as f64 * params.y_sample_ratio).ceil() as usize;
            ys.truncate(sample_count);
            let mut remaining_y_buffer = None;

            for y in ys {
                if remaining_y_buffer == Some(0) {
                    break;
                }

                let mut valid_y = false;
                events.clear();
                states.fill(HitState::default());
                active_old_ids.clear();
                active_pos.fill(None);

                for (old_idx, &old) in bay_old_blocks.iter().enumerate() {
                    if old_time_infos[old_idx].is_none() {
                        continue;
                    }

                    let Some((new_old_pair, old_new_pair)) = crane_pair_cache[old_idx] else {
                        continue;
                    };

                    for &(lo, hi) in new_old_pair.crane.dx_intervals(old.y - y) {
                        push_x_event(
                            old.x - hi,
                            old.x - lo,
                            old_idx,
                            HitDir::NewOld,
                            range.min_x,
                            range.max_x,
                            &mut events,
                        );
                    }
                    for &(lo, hi) in old_new_pair.crane.dx_intervals(y - old.y) {
                        push_x_event(
                            old.x + lo,
                            old.x + hi,
                            old_idx,
                            HitDir::OldNew,
                            range.min_x,
                            range.max_x,
                            &mut events,
                        );
                    }
                }

                events.sort_unstable_by_key(|event| event.x);
                let mut event_pos = 0;
                let mut x = range.min_x;
                loop {
                    while event_pos < events.len() && events[event_pos].x == x {
                        apply_x_event(
                            events[event_pos],
                            &mut states,
                            &mut active_old_ids,
                            &mut active_pos,
                        );
                        event_pos += 1;
                    }

                    forbidden.clear();
                    for &old_idx in &active_old_ids {
                        let Some(info) = old_time_infos[old_idx] else {
                            continue;
                        };
                        let state = states[old_idx];
                        add_forbidden_from_hit_state(
                            info,
                            state.new_old > 0,
                            state.old_new > 0,
                            &mut forbidden,
                        );
                    }

                    if let Some(entry_time) = first_feasible_time(&forbidden, min_t, max_t) {
                        let scheduled = ScheduledBlock {
                            block_id,
                            bay_id,
                            orient_idx,
                            x,
                            y,
                            entry_time,
                            exit_time: entry_time + process_t,
                        };
                        let tardiness = (scheduled.exit_time - block.due_date).max(0);
                        let score_delta = problem.weights.w1 * tardiness as f64 + delta_obj23;
                        let candidate = InsertCandidate {
                            scheduled,
                            score_delta,
                            bbox_right: scheduled.x as f64 + bounds.max_x,
                            bbox_top: scheduled.y as f64 + bounds.max_y,
                        };
                        if best
                            .as_ref()
                            .map_or(true, |best| insert_candidate_better(&candidate, best))
                        {
                            best = Some(candidate);
                        }

                        if tardiness <= original_tardiness {
                            valid_y = true;
                        }
                    }

                    if event_pos >= events.len() {
                        break;
                    }
                    x = events[event_pos].x;
                    if x > range.max_x {
                        break;
                    }
                }

                match &mut remaining_y_buffer {
                    None if valid_y => remaining_y_buffer = Some(params.y_buffer),
                    Some(remaining) => *remaining -= 1,
                    None => {}
                }
            }
        }
    }

    best.map(|candidate| candidate.scheduled)
}

pub fn try_place_block(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    block_id: usize,
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
    min_entry_time: i64,
    max_entry_time: i64,
) -> Option<ScheduledBlock> {
    let block = &problem.blocks[block_id];
    let tentative = ScheduledBlock {
        block_id,
        bay_id,
        orient_idx,
        x,
        y,
        entry_time: 0,
        exit_time: block.processing_time,
    };
    let min_entry_time = block.release_time.max(min_entry_time);
    if min_entry_time > max_entry_time {
        return None;
    }
    let entry_time = get_insert_t(pre, tentative, schedule, min_entry_time, max_entry_time)?;
    Some(ScheduledBlock {
        entry_time,
        exit_time: entry_time + block.processing_time,
        ..tentative
    })
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
    dir: HitDir,
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
        x: l,
        old_idx,
        dir,
        delta: 1,
    });
    if let Some(x) = r.checked_add(1) {
        events.push(XEvent {
            x,
            old_idx,
            dir,
            delta: -1,
        });
    }
}

fn apply_x_event(
    event: XEvent,
    states: &mut [HitState],
    active_old_ids: &mut Vec<usize>,
    active_pos: &mut [Option<usize>],
) {
    let old_idx = event.old_idx;
    let was_active = states[old_idx].is_active();

    match event.dir {
        HitDir::NewOld => {
            if event.delta > 0 {
                states[old_idx].new_old += 1;
            } else {
                debug_assert!(states[old_idx].new_old > 0);
                states[old_idx].new_old -= 1;
            }
        }
        HitDir::OldNew => {
            if event.delta > 0 {
                states[old_idx].old_new += 1;
            } else {
                debug_assert!(states[old_idx].old_new > 0);
                states[old_idx].old_new -= 1;
            }
        }
    }

    let is_active = states[old_idx].is_active();
    if !was_active && is_active {
        active_pos[old_idx] = Some(active_old_ids.len());
        active_old_ids.push(old_idx);
    } else if was_active && !is_active {
        let pos = active_pos[old_idx].take().unwrap();
        let last = active_old_ids.pop().unwrap();
        if pos < active_old_ids.len() {
            active_old_ids[pos] = last;
            active_pos[last] = Some(pos);
        }
    }
}

fn add_forbidden_from_hit_state(
    info: OldTimeInfo,
    new_old_hit: bool,
    old_new_hit: bool,
    forbidden: &mut Vec<Interval>,
) {
    let new_old_clear = !new_old_hit;
    let old_new_clear = !old_new_hit;
    if new_old_clear && old_new_clear {
        return;
    }

    let ol = info.overlap_entry_time_min;
    let or = info.overlap_entry_time_max;
    if !new_old_clear && !old_new_clear {
        forbidden.push((ol, or));
        return;
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
        forbidden.push((ol, or));
        return;
    }
    if ol < allow_l {
        forbidden.push((ol, allow_l - 1));
    }
    if allow_r < or {
        forbidden.push((allow_r + 1, or));
    }
}

fn first_feasible_time(forbidden: &[Interval], min_t: i64, max_t: i64) -> Option<i64> {
    let mut t = min_t;
    loop {
        let mut next_t = t;
        for &(l, r) in forbidden {
            if l <= t && t <= r {
                if r == i64::MAX {
                    return None;
                }
                next_t = next_t.max(r + 1);
            }
        }

        if next_t == t {
            return Some(t);
        }
        if next_t > max_t {
            return None;
        }
        t = next_t;
    }
}

fn get_insert_t(
    pre: &Precompute,
    new_block: ScheduledBlock,
    schedule: &[ScheduledBlock],
    min_t: i64,
    max_t: i64,
) -> Option<i64> {
    let mut forbidden = Vec::with_capacity(16);
    for &old in schedule.iter().filter(|old| old.bay_id == new_block.bay_id) {
        let process_t = new_block.exit_time - new_block.entry_time;
        let Some(info) = old_time_info(old, process_t, min_t, max_t) else {
            continue;
        };

        let new_place = BlockPlacement {
            block_id: new_block.block_id,
            orient_idx: new_block.orient_idx,
            x: new_block.x,
            y: new_block.y,
        };
        let old_place = BlockPlacement {
            block_id: old.block_id,
            orient_idx: old.orient_idx,
            x: old.x,
            y: old.y,
        };

        let new_old_hit = pre.collision.crane(new_place, old_place) == CollisionResult::Hit;
        let old_new_hit = pre.collision.crane(old_place, new_place) == CollisionResult::Hit;
        add_forbidden_from_hit_state(info, new_old_hit, old_new_hit, &mut forbidden);
    }
    first_feasible_time(&forbidden, min_t, max_t)
}
