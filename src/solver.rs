use crate::{
    collision::{BlockPlacement, CollisionResult},
    precompute::Precompute,
    util::{RandPcg64Mcg, Random, time},
    *,
};
use std::cmp::Reverse;
use std::collections::BTreeMap;

const LOCAL_SEARCH_TIME_RATIO: f64 = 0.95;
const MAX_BAY_ASSIGNMENTS: usize = 32;
const MAX_TIME_CANDIDATES: usize = 48;
const RANDOM_POSITION_TRIALS: usize = 48;

#[derive(Clone, Copy, Debug)]
struct ScheduledBlock {
    block_id: usize,
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
    entry_time: i64,
    exit_time: i64,
}

#[derive(Clone, Debug)]
struct BayAssignment {
    cost: f64,
    bays: Vec<usize>,
}

pub fn solve(problem: &Problem, timelimit: f64) -> Result<Solution, String> {
    eprintln!("building precompute...");
    let pre = Precompute::build(problem);
    eprintln!("elapsed: {:.4}", time::elapsed_seconds());

    let mut best = build_initial_schedule(problem, &pre)?;
    let mut best_score = score_schedule(problem, &pre, &best);
    eprintln!("initial score: {:.3}", best_score);

    let mut rng = RandPcg64Mcg::new(1);
    let deadline = timelimit * LOCAL_SEARCH_TIME_RATIO;
    let mut iter = 0usize;
    let mut accepted = 0usize;

    while time::elapsed_seconds() < deadline {
        iter += 1;
        let k = rng.gen_range(2, 10).min(problem.blocks.len());
        let removed = choose_removed_blocks(problem, &pre, &best, k, &mut rng);
        if removed.is_empty() {
            break;
        }

        if let Some(candidate) = try_remove_reinsert(problem, &pre, &best, &removed, &mut rng) {
            let score = score_schedule(problem, &pre, &candidate);
            if score + 1e-9 < best_score {
                eprintln!("new best score: {:.3}", score);
                best = candidate;
                best_score = score;
                accepted += 1;
            }
        }
    }

    eprintln!(
        "local search: iter={}, accepted={}, score={:.3}, elapsed={:.4}",
        iter,
        accepted,
        best_score,
        time::elapsed_seconds()
    );

    Ok(schedule_to_solution(&best))
}

fn build_initial_schedule(
    problem: &Problem,
    pre: &Precompute,
) -> Result<Vec<ScheduledBlock>, String> {
    let n = problem.blocks.len();
    let mut schedule = Vec::with_capacity(n);
    let mut entered = vec![false; n];
    let mut finished = 0;
    let mut active: Vec<ScheduledBlock> = Vec::new();
    let mut t = 0;

    while finished < n {
        let mut i = 0;
        while i < active.len() {
            if active[i].exit_time <= t {
                active.swap_remove(i);
                finished += 1;
            } else {
                i += 1;
            }
        }

        let mut candidates: Vec<usize> = (0..n)
            .filter(|&block_id| !entered[block_id] && problem.blocks[block_id].release_time <= t)
            .collect();
        candidates.sort_by_key(|&block_id| {
            let block = &problem.blocks[block_id];
            let lateness = (t + block.processing_time - block.due_date).max(0);
            (
                Reverse(lateness),
                block.due_date,
                block.release_time,
                block_id,
            )
        });

        let mut entered_any = false;
        for block_id in candidates {
            if let Some(place) = find_bottom_left(problem, pre, block_id, &active) {
                let block = &problem.blocks[block_id];
                let scheduled = ScheduledBlock {
                    block_id,
                    bay_id: place.bay_id,
                    orient_idx: place.orient_idx,
                    x: place.x,
                    y: place.y,
                    entry_time: t,
                    exit_time: t + block.processing_time,
                };
                schedule.push(scheduled);
                active.push(scheduled);
                entered[block_id] = true;
                entered_any = true;
            }
        }

        if finished == n {
            break;
        }
        if !entered_any
            && active.is_empty()
            && entered
                .iter()
                .enumerate()
                .any(|(block_id, &done)| !done && problem.blocks[block_id].release_time <= t)
        {
            return Err(format!("failed to place any released block at time {t}"));
        }

        t += 1;
    }

    Ok(schedule)
}

fn find_bottom_left(
    problem: &Problem,
    pre: &Precompute,
    block_id: usize,
    active: &[ScheduledBlock],
) -> Option<Placement> {
    for &bay_id in &pre.bay_order_by_pref[block_id] {
        for orient_idx in 0..problem.blocks[block_id].shape.len() {
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            for y in range.min_y..=range.max_y {
                for x in range.min_x..=range.max_x {
                    if can_place_with(block_id, bay_id, orient_idx, x, y, active, pre) {
                        return Some(Placement {
                            bay_id,
                            orient_idx,
                            x,
                            y,
                        });
                    }
                }
            }
        }
    }

    None
}

fn try_remove_reinsert<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    removed_ids: &[usize],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    let mut removed = Vec::with_capacity(removed_ids.len());
    let mut base = Vec::with_capacity(schedule.len() - removed_ids.len());
    for &s in schedule {
        if removed_ids.contains(&s.block_id) {
            removed.push(s);
        } else {
            base.push(s);
        }
    }

    if removed.len() != removed_ids.len() {
        return None;
    }

    let mut assignments = enumerate_bay_assignments(problem, pre, &base, &removed);
    if assignments.is_empty() {
        return None;
    }
    assignments.sort_by(|a, b| a.cost.total_cmp(&b.cost));
    assignments.truncate(MAX_BAY_ASSIGNMENTS);

    for assignment in assignments {
        let mut order = removed.clone();
        if rng.nextf() < 0.5 {
            rng.shuffle(&mut order);
        } else {
            order.sort_by_key(|s| problem.blocks[s.block_id].due_date);
        }

        let mut cur = base.clone();
        let mut ok = true;
        for old in order {
            let Some(pos) = removed.iter().position(|s| s.block_id == old.block_id) else {
                ok = false;
                break;
            };
            let bay_id = assignment.bays[pos];
            let inserted = find_insert_position(problem, pre, old, Some(bay_id), &cur, rng)
                .or_else(|| find_insert_position(problem, pre, old, None, &cur, rng));
            if let Some(scheduled) = inserted {
                cur.push(scheduled);
            } else {
                ok = false;
                break;
            }
        }

        if ok {
            return Some(cur);
        }
    }

    None
}

fn enumerate_bay_assignments(
    problem: &Problem,
    pre: &Precompute,
    base: &[ScheduledBlock],
    removed: &[ScheduledBlock],
) -> Vec<BayAssignment> {
    let mut base_loads = vec![0.0; problem.bays.len()];
    for s in base {
        base_loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }

    let mut candidates = Vec::new();
    let mut cur = Vec::with_capacity(removed.len());
    enumerate_bay_assignments_dfs(
        problem,
        pre,
        removed,
        &base_loads,
        0,
        &mut cur,
        &mut candidates,
    );
    candidates
}

fn enumerate_bay_assignments_dfs(
    problem: &Problem,
    pre: &Precompute,
    removed: &[ScheduledBlock],
    base_loads: &[f64],
    pos: usize,
    cur: &mut Vec<usize>,
    out: &mut Vec<BayAssignment>,
) {
    if out.len() > 10_000 {
        return;
    }
    if pos == removed.len() {
        let mut loads = base_loads.to_vec();
        let mut obj3 = 0.0;
        for (s, &bay_id) in removed.iter().zip(cur.iter()) {
            loads[bay_id] += problem.blocks[s.block_id].workload as f64;
            obj3 += pre.pref_penalty[s.block_id][bay_id] as f64;
        }
        let cost =
            problem.weights.w2 * normalized_imbalance(pre, &loads) + problem.weights.w3 * obj3;
        out.push(BayAssignment {
            cost,
            bays: cur.clone(),
        });
        return;
    }

    let block_id = removed[pos].block_id;
    for &bay_id in &pre.bay_order_by_pref[block_id] {
        if !has_fit_position(problem, pre, block_id, bay_id) {
            continue;
        }
        cur.push(bay_id);
        enumerate_bay_assignments_dfs(problem, pre, removed, base_loads, pos + 1, cur, out);
        cur.pop();
    }
}

fn find_insert_position<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    fixed_bay_id: Option<usize>,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<ScheduledBlock> {
    for on_time_only in [true, false] {
        let times = time_candidates(problem, original, schedule, on_time_only, rng);
        if times.is_empty() {
            continue;
        }

        let mut bay_order: Vec<usize> = match fixed_bay_id {
            Some(bay_id) => vec![bay_id],
            None => pre.bay_order_by_pref[original.block_id].clone(),
        };
        if fixed_bay_id.is_none() && rng.nextf() < 0.2 {
            rng.shuffle(&mut bay_order);
        }

        if rng.nextf() < 0.12 {
            if let Some(scheduled) = try_random_positions(
                problem,
                pre,
                original.block_id,
                &bay_order,
                &times,
                schedule,
                rng,
            ) {
                return Some(scheduled);
            }
        }

        let mut directions = [0usize, 1, 2, 3];
        if rng.nextf() < 0.2 {
            rng.shuffle(&mut directions);
        }

        for &entry_time in &times {
            for &bay_id in &bay_order {
                for &direction in &directions {
                    let mut orient_order: Vec<usize> =
                        (0..problem.blocks[original.block_id].shape.len()).collect();
                    if rng.nextf() < 0.2 {
                        rng.shuffle(&mut orient_order);
                    }
                    for orient_idx in orient_order {
                        let Some(range) =
                            pre.collision
                                .fit_range(bay_id, original.block_id, orient_idx)
                        else {
                            continue;
                        };
                        let x_len = (range.max_x - range.min_x + 1) as usize;
                        let y_len = (range.max_y - range.min_y + 1) as usize;
                        let x_asc = direction == 0 || direction == 2;
                        let y_asc = direction == 0 || direction == 1;
                        for yi in 0..y_len {
                            let y = if y_asc {
                                range.min_y + yi as i64
                            } else {
                                range.max_y - yi as i64
                            };
                            for xi in 0..x_len {
                                let x = if x_asc {
                                    range.min_x + xi as i64
                                } else {
                                    range.max_x - xi as i64
                                };
                                let scheduled = ScheduledBlock {
                                    block_id: original.block_id,
                                    bay_id,
                                    orient_idx,
                                    x,
                                    y,
                                    entry_time,
                                    exit_time: entry_time
                                        + problem.blocks[original.block_id].processing_time,
                                };
                                if can_insert(pre, scheduled, schedule) {
                                    return Some(scheduled);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

fn try_random_positions<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    block_id: usize,
    bay_order: &[usize],
    times: &[i64],
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<ScheduledBlock> {
    for _ in 0..RANDOM_POSITION_TRIALS {
        let entry_time = rng.choice(times);
        let bay_id = rng.choice(bay_order);
        let orient_idx = rng.gen_range(0, problem.blocks[block_id].shape.len());
        let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
            continue;
        };
        let x_len = (range.max_x - range.min_x + 1) as usize;
        let y_len = (range.max_y - range.min_y + 1) as usize;
        let x = range.min_x + rng.gen_range(0, x_len) as i64;
        let y = range.min_y + rng.gen_range(0, y_len) as i64;
        let scheduled = ScheduledBlock {
            block_id,
            bay_id,
            orient_idx,
            x,
            y,
            entry_time,
            exit_time: entry_time + problem.blocks[block_id].processing_time,
        };
        if can_insert(pre, scheduled, schedule) {
            return Some(scheduled);
        }
    }
    None
}

fn time_candidates<R: Random>(
    problem: &Problem,
    original: ScheduledBlock,
    schedule: &[ScheduledBlock],
    on_time_only: bool,
    rng: &mut R,
) -> Vec<i64> {
    let block = &problem.blocks[original.block_id];
    let p = block.processing_time;
    let earliest = block.release_time;
    let latest_on_time = block.due_date - p;

    let (lo, hi) = if on_time_only {
        if latest_on_time < earliest {
            return Vec::new();
        }
        (earliest, latest_on_time)
    } else {
        let max_exit = schedule
            .iter()
            .map(|s| s.exit_time)
            .max()
            .unwrap_or(original.exit_time)
            .max(original.exit_time);
        let lo = earliest.max(latest_on_time + 1);
        let hi = max_exit
            .max(block.due_date)
            .max(original.entry_time)
            .max(lo)
            + p
            + 30;
        (lo, hi)
    };

    let mut times = Vec::new();
    push_time_candidate(&mut times, original.entry_time, lo, hi);
    push_time_candidate(&mut times, earliest, lo, hi);
    push_time_candidate(&mut times, latest_on_time, lo, hi);
    push_time_candidate(&mut times, lo, lo, hi);
    push_time_candidate(&mut times, hi, lo, hi);

    for s in schedule {
        push_time_candidate(&mut times, s.entry_time, lo, hi);
        push_time_candidate(&mut times, s.exit_time, lo, hi);
        push_time_candidate(&mut times, s.entry_time - p + 1, lo, hi);
        push_time_candidate(&mut times, s.exit_time - p + 1, lo, hi);
    }

    let span = hi - lo + 1;
    if span <= MAX_TIME_CANDIDATES as i64 {
        for t in lo..=hi {
            times.push(t);
        }
    } else {
        while times.len() < MAX_TIME_CANDIDATES {
            times.push(lo + rng.gen_range(0, span as usize) as i64);
        }
    }

    times.sort_unstable();
    times.dedup();
    times.sort_by_key(|&t| ((t + p - block.due_date).max(0), t));
    if times.len() > MAX_TIME_CANDIDATES {
        times.truncate(MAX_TIME_CANDIDATES);
    }
    times
}

fn push_time_candidate(times: &mut Vec<i64>, t: i64, lo: i64, hi: i64) {
    if lo <= t && t <= hi {
        times.push(t);
    }
}

fn choose_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    k: usize,
    rng: &mut R,
) -> Vec<usize> {
    let n = schedule.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let k = k.min(n);
    let mut ids: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    if rng.nextf() < 0.25 {
        rng.shuffle(&mut ids);
        ids.truncate(k);
        return ids;
    }

    let mut loads = vec![0.0; problem.bays.len()];
    for s in schedule {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }
    let heavy_bay = most_loaded_bay(pre, &loads);

    let mut badness = vec![0i64; problem.blocks.len()];
    for s in schedule {
        let block = &problem.blocks[s.block_id];
        let tardiness = (s.exit_time - block.due_date).max(0);
        let pref_penalty = pre.pref_penalty[s.block_id][s.bay_id];
        badness[s.block_id] = tardiness * 10_000
            + pref_penalty * 100
            + if Some(s.bay_id) == heavy_bay {
                1_000
            } else {
                0
            };
    }

    rng.shuffle(&mut ids);
    ids.sort_by_key(|&block_id| Reverse(badness[block_id]));
    let pool_len = (k * 8).min(ids.len()).max(k);
    ids.truncate(pool_len);
    rng.shuffle(&mut ids);
    ids.truncate(k);
    ids
}

fn most_loaded_bay(pre: &Precompute, loads: &[f64]) -> Option<usize> {
    if loads.is_empty() {
        return None;
    }
    let mut best = 0;
    let mut best_load = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let normalized = load * pre.bay_load_scale[bay_id];
        if normalized > best_load {
            best_load = normalized;
            best = bay_id;
        }
    }
    Some(best)
}

fn score_schedule(problem: &Problem, pre: &Precompute, schedule: &[ScheduledBlock]) -> f64 {
    let mut obj1 = 0.0;
    let mut obj3 = 0.0;
    let mut loads = vec![0.0; problem.bays.len()];

    for s in schedule {
        let block = &problem.blocks[s.block_id];
        obj1 += (s.exit_time - block.due_date).max(0) as f64;
        loads[s.bay_id] += block.workload as f64;
        obj3 += pre.pref_penalty[s.block_id][s.bay_id] as f64;
    }

    let obj2 = normalized_imbalance(pre, &loads);
    problem.weights.w1 * obj1 + problem.weights.w2 * obj2 + problem.weights.w3 * obj3
}

fn normalized_imbalance(pre: &Precompute, loads: &[f64]) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }

    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let normalized = load * pre.bay_load_scale[bay_id];
        min_value = min_value.min(normalized);
        max_value = max_value.max(normalized);
    }
    (max_value - min_value).floor()
}

fn has_fit_position(problem: &Problem, pre: &Precompute, block_id: usize, bay_id: usize) -> bool {
    (0..problem.blocks[block_id].shape.len()).any(|orient_idx| {
        pre.collision
            .fit_range(bay_id, block_id, orient_idx)
            .is_some()
    })
}

fn can_insert(pre: &Precompute, new_block: ScheduledBlock, schedule: &[ScheduledBlock]) -> bool {
    let new_place = BlockPlacement {
        block_id: new_block.block_id,
        orient_idx: new_block.orient_idx,
        x: new_block.x,
        y: new_block.y,
    };

    for old in schedule.iter().filter(|old| {
        old.bay_id == new_block.bay_id
            && interval_overlaps(
                old.entry_time,
                old.exit_time,
                new_block.entry_time,
                new_block.exit_time,
            )
    }) {
        if !placements_clear(pre, new_place, *old) {
            return false;
        }
    }
    true
}

fn can_place_with(
    block_id: usize,
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
    active: &[ScheduledBlock],
    pre: &Precompute,
) -> bool {
    let new_place = BlockPlacement {
        block_id,
        orient_idx,
        x,
        y,
    };

    for old in active.iter().filter(|old| old.bay_id == bay_id) {
        if !placements_clear(pre, new_place, *old) {
            return false;
        }
    }
    true
}

fn placements_clear(pre: &Precompute, new_place: BlockPlacement, old: ScheduledBlock) -> bool {
    let old_place = BlockPlacement {
        block_id: old.block_id,
        orient_idx: old.orient_idx,
        x: old.x,
        y: old.y,
    };
    pre.collision.crane(new_place, old_place) == CollisionResult::Clear
        && pre.collision.crane(old_place, new_place) == CollisionResult::Clear
}

fn interval_overlaps(a0: i64, a1: i64, b0: i64, b1: i64) -> bool {
    a0 < b1 && b0 < a1
}

fn schedule_to_solution(schedule: &[ScheduledBlock]) -> Solution {
    let mut operations: BTreeMap<i64, Vec<Operation>> = BTreeMap::new();

    for s in schedule {
        operations.entry(s.exit_time).or_default().push(Operation {
            op_type: "EXIT",
            block_id: s.block_id,
            bay_id: s.bay_id,
            x: None,
            y: None,
            orient_idx: None,
        });
    }
    for s in schedule {
        operations.entry(s.entry_time).or_default().push(Operation {
            op_type: "ENTRY",
            block_id: s.block_id,
            bay_id: s.bay_id,
            x: Some(s.x),
            y: Some(s.y),
            orient_idx: Some(s.orient_idx),
        });
    }

    operations.retain(|_, ops| !ops.is_empty());
    Solution { operations }
}
