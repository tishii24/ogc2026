use crate::{
    collision::{BlockPlacement, CollisionResult},
    precompute::Precompute,
    util::rand::{RandPcg64Mcg, Random},
    util::time,
    *,
};
use std::cmp::Reverse;
use std::collections::BTreeMap;

const RNG_SEED: u64 = 1;

const LOCAL_SEARCH_TIME_RATIO: f64 = 0.95;
const START_TEMP: f64 = 1e4;
const END_TEMP: f64 = 1.0;

const MIN_REMOVED_BLOCKS: usize = 1;
const MAX_REMOVED_BLOCKS: usize = 8;
const REMOVE_POOL_FACTOR: usize = 8;
const REMOVE_SEED_COUNT: usize = 3;
const REMOVE_RANDOM_SEED_COUNT: usize = 1;
const REMOVE_NEIGHBOR_POOL_FACTOR: usize = 4;
const INSERT_X_BUFFER: i64 = 10;

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

type Interval = (i64, i64);

struct InsertCandidate {
    scheduled: ScheduledBlock,
    tardiness: i64,
    delta_obj23: f64,
    orient_rank: usize,
}

pub fn solve(problem: &Problem, timelimit: f64) -> Result<Solution, String> {
    eprintln!("building precompute...");
    let pre = Precompute::build(problem);
    eprintln!("elapsed: {:.4}", time::elapsed_seconds());

    let mut rng = RandPcg64Mcg::new(RNG_SEED);
    let mut current = build_initial_schedule(problem, &pre)?;
    let mut current_score = score_schedule(problem, &pre, &current);
    let mut best = current.clone();
    let mut best_score = current_score;
    eprintln!("initial score: {:.3}", best_score);
    let deadline = timelimit * LOCAL_SEARCH_TIME_RATIO;
    let mut iter = 0usize;
    let mut accepted = 0usize;
    let mut improved = 0usize;

    while time::elapsed_seconds() < deadline {
        iter += 1;
        let progress = (time::elapsed_seconds() / deadline).clamp(0.0, 1.0);
        let temp = START_TEMP * (END_TEMP / START_TEMP).powf(progress);
        let k = rng
            .gen_range(MIN_REMOVED_BLOCKS, MAX_REMOVED_BLOCKS + 1)
            .min(problem.blocks.len());
        let removed = choose_removed_blocks(problem, &pre, &current, k, &mut rng);
        if removed.is_empty() {
            break;
        }

        if let Some(candidate) = try_remove_reinsert(problem, &pre, &current, &removed) {
            let score = score_schedule(problem, &pre, &candidate);
            let delta = score - current_score;
            if delta <= 0.0 || rng.nextf() < (-delta / temp).exp() {
                current = candidate;
                current_score = score;
                accepted += 1;

                if current_score + 1e-9 < best_score {
                    eprintln!("new best score: {:.3}", current_score);
                    best = current.clone();
                    best_score = current_score;
                    improved += 1;
                }
            }
        }
    }

    eprintln!(
        "annealing: iter={}, accepted={}, improved={}, current={:.3}, best={:.3}, elapsed={:.4}",
        iter,
        accepted,
        improved,
        current_score,
        best_score,
        time::elapsed_seconds()
    );

    Ok(schedule_to_solution(&best))
}

fn build_initial_schedule(
    problem: &Problem,
    pre: &Precompute,
) -> Result<Vec<ScheduledBlock>, String> {
    let mut schedule = Vec::with_capacity(problem.blocks.len());
    let mut loads = vec![0.0; problem.bays.len()];
    let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
    order.sort_by_key(|&block_id| {
        let block = &problem.blocks[block_id];
        (block.due_date, block.release_time, block_id)
    });

    for block_id in order {
        let block = &problem.blocks[block_id];
        let original = ScheduledBlock {
            block_id,
            bay_id: 0,
            orient_idx: 0,
            x: 0,
            y: 0,
            entry_time: block.release_time,
            exit_time: block.release_time + block.processing_time,
        };
        let scheduled = find_best_insert_position(problem, pre, original, &schedule, &loads)
            .ok_or_else(|| format!("failed to place block {block_id} in initial schedule"))?;
        loads[scheduled.bay_id] += block.workload as f64;
        schedule.push(scheduled);
    }

    Ok(schedule)
}

fn try_remove_reinsert(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    removed_ids: &[usize],
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

    removed.sort_by(|a, b| {
        pre.block_area[b.block_id]
            .total_cmp(&pre.block_area[a.block_id])
            .then(a.block_id.cmp(&b.block_id))
    });

    let mut cur = base;
    let mut loads = vec![0.0; problem.bays.len()];
    for s in &cur {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }

    for old in removed {
        let scheduled = find_best_insert_position(problem, pre, old, &cur, &loads)?;
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        cur.push(scheduled);
    }

    Some(cur)
}

fn find_best_insert_position(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    schedule: &[ScheduledBlock],
    loads: &[f64],
) -> Option<ScheduledBlock> {
    let block_id = original.block_id;
    let block = &problem.blocks[block_id];
    let p = block.processing_time;
    let lo = block.release_time;
    let cur_max_exit = schedule.iter().map(|s| s.exit_time).max().unwrap_or(lo);
    let hi = cur_max_exit.max(lo);
    let current_obj2 = normalized_imbalance(pre, loads);
    let original_tardiness = (original.exit_time - block.due_date).max(0);
    let mut best: Option<InsertCandidate> = None;

    for bay_id in 0..problem.bays.len() {
        let mut next_loads = loads.to_vec();
        next_loads[bay_id] += block.workload as f64;
        let delta_obj2 =
            problem.weights.w2 * (normalized_imbalance(pre, &next_loads) - current_obj2);
        let delta_obj3 = problem.weights.w3 * pre.pref_penalty[block_id][bay_id] as f64;
        let delta_obj23 = delta_obj2 + delta_obj3;

        for (orient_rank, &orient_idx) in pre.orientation_order_by_bbox[block_id].iter().enumerate()
        {
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            let mut anchor_x: Option<i64> = None;
            let mut found_acceptable_in_orientation = false;

            for x in range.min_x..=range.max_x {
                if let Some(anchor_x) = anchor_x {
                    if x > anchor_x + INSERT_X_BUFFER {
                        break;
                    }
                }

                for y in range.min_y..=range.max_y {
                    let tentative = ScheduledBlock {
                        block_id,
                        bay_id,
                        orient_idx,
                        x,
                        y,
                        entry_time: 0,
                        exit_time: p,
                    };
                    let Some(entry_time) =
                        best_time_for_fixed_placement(pre, tentative, schedule, lo, hi)
                    else {
                        continue;
                    };
                    let scheduled = ScheduledBlock {
                        entry_time,
                        exit_time: entry_time + p,
                        ..tentative
                    };
                    debug_assert!(can_insert(pre, scheduled, schedule));
                    let tardiness = (scheduled.exit_time - block.due_date).max(0);
                    let candidate = InsertCandidate {
                        scheduled,
                        tardiness,
                        delta_obj23,
                        orient_rank,
                    };
                    if best
                        .as_ref()
                        .map_or(true, |best| insert_candidate_better(&candidate, best))
                    {
                        best = Some(candidate);
                    }

                    if tardiness <= original_tardiness {
                        if anchor_x.is_none() {
                            anchor_x = Some(x);
                        }
                        found_acceptable_in_orientation = true;
                    }
                }
            }

            if found_acceptable_in_orientation {
                break;
            }
        }
    }

    best.map(|candidate| candidate.scheduled)
}

fn insert_candidate_better(a: &InsertCandidate, b: &InsertCandidate) -> bool {
    match a.tardiness.cmp(&b.tardiness) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    match a.delta_obj23.total_cmp(&b.delta_obj23) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    match a.scheduled.entry_time.cmp(&b.scheduled.entry_time) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    match a.orient_rank.cmp(&b.orient_rank) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    match a.scheduled.x.cmp(&b.scheduled.x) {
        std::cmp::Ordering::Less => return true,
        std::cmp::Ordering::Greater => return false,
        std::cmp::Ordering::Equal => {}
    }
    a.scheduled.y < b.scheduled.y
}

fn clamp_interval(l: i64, r: i64, lo: i64, hi: i64) -> Option<Interval> {
    let l = l.max(lo);
    let r = r.min(hi);
    if l <= r { Some((l, r)) } else { None }
}

fn add_forbidden_intervals_for_old(
    pre: &Precompute,
    new_block: ScheduledBlock,
    old: ScheduledBlock,
    lo: i64,
    hi: i64,
    forbidden: &mut Vec<Interval>,
) {
    let p = new_block.exit_time - new_block.entry_time;
    let a = old.entry_time;
    let b = old.exit_time;

    let Some((ol, or)) = clamp_interval(a - p + 1, b - 1, lo, hi) else {
        return;
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

    let new_old_clear = pre.collision.crane(new_place, old_place) == CollisionResult::Clear;
    let old_new_clear = pre.collision.crane(old_place, new_place) == CollisionResult::Clear;
    if new_old_clear && old_new_clear {
        return;
    }

    let mut points = vec![ol, or + 1];

    if let Some((l, r)) = clamp_interval(a + 1, b - p - 1, ol, or) {
        points.push(l);
        points.push(r + 1);
    }
    if let Some((l, r)) = clamp_interval(b - p + 1, a - 1, ol, or) {
        points.push(l);
        points.push(r + 1);
    }

    points.sort_unstable();
    points.dedup();

    for window in points.windows(2) {
        let l = window[0];
        let r = window[1] - 1;
        if l > r {
            continue;
        }

        let t = l;
        let ok = if a < t && t + p < b {
            new_old_clear
        } else if t < a && b < t + p {
            old_new_clear
        } else {
            new_old_clear && old_new_clear
        };

        if !ok {
            forbidden.push((l, r));
        }
    }
}

fn merge_intervals(intervals: &mut Vec<Interval>) {
    intervals.sort_unstable_by_key(|&(l, r)| (l, r));
    let mut merged: Vec<Interval> = Vec::new();
    for &(l, r) in intervals.iter() {
        if let Some(last) = merged.last_mut() {
            if l <= last.1 + 1 {
                last.1 = last.1.max(r);
                continue;
            }
        }
        merged.push((l, r));
    }
    *intervals = merged;
}

fn first_feasible_time(mut forbidden: Vec<Interval>, lo: i64, hi: i64) -> Option<i64> {
    merge_intervals(&mut forbidden);
    let mut t = lo;
    for (l, r) in forbidden {
        if t < l {
            return Some(t);
        }
        if t <= r {
            t = r + 1;
        }
        if t > hi {
            return None;
        }
    }
    if t <= hi { Some(t) } else { None }
}

fn best_time_for_fixed_placement(
    pre: &Precompute,
    new_block: ScheduledBlock,
    schedule: &[ScheduledBlock],
    lo: i64,
    hi: i64,
) -> Option<i64> {
    let mut forbidden = Vec::new();
    for &old in schedule.iter().filter(|old| old.bay_id == new_block.bay_id) {
        add_forbidden_intervals_for_old(pre, new_block, old, lo, hi, &mut forbidden);
    }
    first_feasible_time(forbidden, lo, hi)
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
        let score = tardiness as f64 * problem.weights.w1
            + pref_penalty as f64 * problem.weights.w3
            + if Some(s.bay_id) == heavy_bay {
                problem.weights.w2
            } else {
                0.
            };
        badness[s.block_id] = score as i64;
    }

    let mut by_block = vec![None; problem.blocks.len()];
    for &s in schedule {
        by_block[s.block_id] = Some(s);
    }

    fn center(pre: &Precompute, s: ScheduledBlock) -> (f64, f64) {
        let (cx, cy) = pre.orientation_bbox_center[s.block_id][s.orient_idx];
        (s.x as f64 + cx, s.y as f64 + cy)
    }

    fn push_selected(selected: &mut Vec<usize>, used: &mut [bool], block_id: usize, k: usize) {
        if selected.len() < k && !used[block_id] {
            selected.push(block_id);
            used[block_id] = true;
        }
    }

    rng.shuffle(&mut ids);
    ids.sort_by_key(|&block_id| Reverse(badness[block_id]));
    let pool_len = (k * REMOVE_POOL_FACTOR).min(ids.len()).max(k);
    ids.truncate(pool_len);
    let bad_pool = ids;

    let mut selected = Vec::with_capacity(k);
    let mut used = vec![false; problem.blocks.len()];
    let seed_count = REMOVE_SEED_COUNT.min(k);
    let random_seed_count = REMOVE_RANDOM_SEED_COUNT.min(seed_count);
    let bad_seed_count = seed_count - random_seed_count;

    let mut seed_pool = bad_pool.clone();
    rng.shuffle(&mut seed_pool);
    for &block_id in seed_pool.iter().take(bad_seed_count) {
        push_selected(&mut selected, &mut used, block_id, k);
    }

    let mut random_seed_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut random_seed_pool);
    for block_id in random_seed_pool {
        if selected.len() >= seed_count {
            break;
        }
        push_selected(&mut selected, &mut used, block_id, k);
    }

    let seeds = selected.clone();
    for (seed_index, &seed_id) in seeds.iter().enumerate() {
        if selected.len() >= k {
            break;
        }
        let Some(seed) = by_block[seed_id] else {
            continue;
        };
        let seeds_left = seeds.len() - seed_index;
        let need = (k - selected.len() + seeds_left - 1) / seeds_left;
        let (sx, sy) = center(pre, seed);
        let mut neighbors: Vec<(f64, usize)> = schedule
            .iter()
            .filter(|s| s.bay_id == seed.bay_id && !used[s.block_id])
            .map(|&s| {
                let (x, y) = center(pre, s);
                let dx = x - sx;
                let dy = y - sy;
                (dx * dx + dy * dy, s.block_id)
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let pool_len = (need * REMOVE_NEIGHBOR_POOL_FACTOR).min(neighbors.len());
        let mut neighbor_ids: Vec<usize> = neighbors
            .into_iter()
            .take(pool_len)
            .map(|(_, block_id)| block_id)
            .collect();
        rng.shuffle(&mut neighbor_ids);
        for block_id in neighbor_ids.into_iter().take(need) {
            push_selected(&mut selected, &mut used, block_id, k);
        }
    }

    let mut fill_pool = bad_pool;
    rng.shuffle(&mut fill_pool);
    for block_id in fill_pool {
        push_selected(&mut selected, &mut used, block_id, k);
    }

    selected
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

fn can_insert(pre: &Precompute, new_block: ScheduledBlock, schedule: &[ScheduledBlock]) -> bool {
    fn interval_overlaps(a0: i64, a1: i64, b0: i64, b1: i64) -> bool {
        a0 < b1 && b0 < a1
    }

    fn place(s: ScheduledBlock) -> BlockPlacement {
        BlockPlacement {
            block_id: s.block_id,
            orient_idx: s.orient_idx,
            x: s.x,
            y: s.y,
        }
    }

    fn crane_clear(pre: &Precompute, moving: BlockPlacement, fixed: BlockPlacement) -> bool {
        pre.collision.crane(moving, fixed) == CollisionResult::Clear
    }

    fn placements_clear(pre: &Precompute, new_block: ScheduledBlock, old: ScheduledBlock) -> bool {
        let new_place = place(new_block);
        let old_place = place(old);

        if new_block.entry_time < old.entry_time && old.exit_time < new_block.exit_time {
            crane_clear(pre, old_place, new_place)
        } else if old.entry_time < new_block.entry_time && new_block.exit_time < old.exit_time {
            crane_clear(pre, new_place, old_place)
        } else {
            // ABAB 型は両方向が必要。同時刻の ENTRY/EXIT も操作順に依存するため保守的に両方向を見る。
            crane_clear(pre, new_place, old_place) && crane_clear(pre, old_place, new_place)
        }
    }

    for old in schedule.iter().filter(|old| {
        old.bay_id == new_block.bay_id
            && interval_overlaps(
                old.entry_time,
                old.exit_time,
                new_block.entry_time,
                new_block.exit_time,
            )
    }) {
        if !placements_clear(pre, new_block, *old) {
            return false;
        }
    }
    true
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
