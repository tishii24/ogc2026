use crate::{
    collision::{BlockPlacement, CollisionResult},
    core::*,
    precompute::Precompute,
    util::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
    *,
};
use rayon::prelude::*;
use std::cmp::Reverse;

macro_rules! log {
    ($timer:expr, $($arg:tt)*) => {
        eprintln!("[{:.4}] {}", $timer.elapsed_seconds(), format_args!($($arg)*))
    };
}

const RNG_SEED: u64 = 1;
const MAX_WORKER_COUNT: usize = 4;

const LOCAL_SEARCH_TIME_BUFFER_SECONDS: f64 = 3.;
const START_TEMP: f64 = 1e1;
const END_TEMP: f64 = 1e0;

const INITIAL_AREA_WEIGHT_MIN: f64 = 0.0;
const INITIAL_AREA_WEIGHT_MAX: f64 = 2.0;

const MIN_REMOVED_BLOCKS: usize = 1;
const MAX_REMOVED_BLOCKS: usize = 13;
const REMOVE_POOL_FACTOR: usize = 8;
const REMOVE_SEED_COUNT: usize = 3;
const REMOVE_RANDOM_SEED_COUNT: usize = 1;
const REMOVE_NEIGHBOR_POOL_FACTOR: usize = 4;
const INSERT_X_BUFFER: i64 = 10;
const INSERT_PARAMS: InsertSearchParams = InsertSearchParams {
    x_step: 1,
    y_step: 1,
};
const ORDER_SLACK_WEIGHT_MIN: f64 = 0.0;
const ORDER_SLACK_WEIGHT_MAX: f64 = 4.0;

const MOVE_MAX_SHIFT_X: i64 = 10;
const MOVE_MAX_SHIFT_Y: i64 = 10;

const NEIGHBOR_KIND_COUNT: usize = 2;
const NEIGHBOR_PROBS: &[(NeighborKind, f64)] = &[
    (NeighborKind::LargeReconstruct, 0.2),
    (NeighborKind::Move, 0.8),
];

type Interval = (i64, i64);

#[derive(Clone, Copy, Debug)]
enum NeighborKind {
    LargeReconstruct,
    Move,
}

#[derive(Clone, Copy, Default)]
struct NeighborStats {
    selected: usize,
    accepted: usize,
}

#[derive(Clone, Copy)]
struct InsertSearchParams {
    x_step: i64,
    y_step: i64,
}

struct InsertCandidate {
    scheduled: ScheduledBlock,
    score_delta: f64,
    bbox_right: f64,
    bbox_top: f64,
}

struct AnnealingResult {
    worker_id: usize,
    schedule: Vec<ScheduledBlock>,
    score: f64,
    current_score: f64,
    iter: usize,
    accepted: usize,
    improved: usize,
    neighbor_stats: [NeighborStats; NEIGHBOR_KIND_COUNT],
}

pub fn solve(problem: &Problem, timelimit: f64, timer: Timer) -> Result<Solution, String> {
    log!(timer, "building precompute...");
    let pre = Precompute::build(problem);
    log!(timer, "precompute built");

    let deadline = timelimit - LOCAL_SEARCH_TIME_BUFFER_SECONDS;
    let worker_count = rayon::current_num_threads().clamp(1, MAX_WORKER_COUNT);
    log!(timer, "annealing workers: {}", worker_count);
    let worker_timers = vec![timer; worker_count];
    let results: Vec<_> = worker_timers
        .into_par_iter()
        .enumerate()
        .map(|(worker_id, worker_timer)| {
            let mut initial_rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(worker_id as u64));
            let initial =
                build_initial_schedule(problem, &pre, worker_id, worker_count, &mut initial_rng)?;
            let initial_score = score_schedule(problem, &pre, &initial);
            log!(
                worker_timer,
                "worker {} initial score: {:.3}",
                worker_id,
                initial_score
            );
            Ok(run_annealing_worker(
                problem,
                &pre,
                &initial,
                initial_score,
                deadline,
                worker_id,
                worker_timer,
            ))
        })
        .collect::<Vec<Result<AnnealingResult, String>>>();

    let mut best: Option<Vec<ScheduledBlock>> = None;
    let mut best_score = f64::INFINITY;
    for result in results {
        let result = result?;
        log!(
            timer,
            "[id={}] iter={}, best={:.3}, accepted={}, improved={}, current={:.3}, neighbors: {}",
            result.worker_id,
            result.iter,
            result.score,
            result.accepted,
            result.improved,
            result.current_score,
            format_neighbor_stats(&result.neighbor_stats),
        );
        if result.score + 1e-9 < best_score {
            best_score = result.score;
            best = Some(result.schedule);
        }
    }

    let best = best.ok_or_else(|| "no annealing worker result".to_string())?;
    Ok(schedule_to_solution(&best))
}

fn run_annealing_worker(
    problem: &Problem,
    pre: &Precompute,
    initial: &[ScheduledBlock],
    initial_score: f64,
    deadline: f64,
    worker_id: usize,
    timer: Timer,
) -> AnnealingResult {
    let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(worker_id as u64));
    let mut current = initial.to_vec();
    let mut current_score = initial_score;
    let mut best = current.clone();
    let mut best_score = current_score;
    let mut iter = 0usize;
    let mut accepted = 0usize;
    let mut improved = 0usize;
    let mut neighbor_stats = [NeighborStats::default(); NEIGHBOR_KIND_COUNT];

    loop {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= deadline {
            break;
        }
        iter += 1;
        let progress = (elapsed / deadline).clamp(0.0, 1.0);
        let temp = START_TEMP * (END_TEMP / START_TEMP).powf(progress);

        let neighbor = sample_neighbor(&mut rng);
        let neighbor_idx = neighbor_index(neighbor);
        neighbor_stats[neighbor_idx].selected += 1;

        let candidate = match neighbor {
            NeighborKind::LargeReconstruct => {
                try_large_reconstruct(problem, pre, &current, &mut rng)
            }
            NeighborKind::Move => try_move_neighbor(problem, pre, &current, &mut rng),
        };
        let Some(candidate) = candidate else {
            continue;
        };

        let score = score_schedule(problem, pre, &candidate);
        let delta = score - current_score;
        if delta <= 0.0 || rng.nextf() < (-delta / temp).exp() {
            current = candidate;
            current_score = score;
            accepted += 1;
            neighbor_stats[neighbor_idx].accepted += 1;

            if current_score + 1e-9 < best_score {
                log!(
                    timer,
                    "worker {} new best score: {:.3}",
                    worker_id,
                    current_score
                );
                best = current.clone();
                best_score = current_score;
                improved += 1;
            }
        }
    }

    AnnealingResult {
        worker_id,
        schedule: best,
        score: best_score,
        current_score,
        iter,
        accepted,
        improved,
        neighbor_stats,
    }
}

fn neighbor_index(kind: NeighborKind) -> usize {
    match kind {
        NeighborKind::LargeReconstruct => 0,
        NeighborKind::Move => 1,
    }
}

fn neighbor_name(kind: NeighborKind) -> &'static str {
    match kind {
        NeighborKind::LargeReconstruct => "large",
        NeighborKind::Move => "move",
    }
}

fn sample_neighbor<R: Random>(rng: &mut R) -> NeighborKind {
    let total = NEIGHBOR_PROBS.iter().map(|&(_, prob)| prob).sum::<f64>();
    debug_assert!(total > 0.0);

    let mut x = rng.nextf() * total;
    for &(kind, prob) in NEIGHBOR_PROBS {
        if x < prob {
            return kind;
        }
        x -= prob;
    }
    NEIGHBOR_PROBS.last().unwrap().0
}

fn format_neighbor_stats(stats: &[NeighborStats; NEIGHBOR_KIND_COUNT]) -> String {
    NEIGHBOR_PROBS
        .iter()
        .map(|&(kind, _)| {
            let stat = stats[neighbor_index(kind)];
            format!(
                "{}={}/{}",
                neighbor_name(kind),
                stat.accepted,
                stat.selected,
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn try_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    let k = rng
        .gen_range(MIN_REMOVED_BLOCKS, MAX_REMOVED_BLOCKS + 1)
        .min(problem.blocks.len());

    let removed = choose_removed_blocks(problem, pre, schedule, k, rng);
    if removed.is_empty() {
        return None;
    }

    let order_slack_weight = sample_order_slack_weight(rng);
    try_remove_reinsert(problem, pre, schedule, &removed, order_slack_weight, rng)
}

fn try_move_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];
    let block = &problem.blocks[old.block_id];

    let mut base = Vec::with_capacity(schedule.len() - 1);
    for (i, &s) in schedule.iter().enumerate() {
        if i != idx {
            base.push(s);
        }
    }

    let range = pre
        .collision
        .fit_range(old.bay_id, old.block_id, old.orient_idx)?;
    let original_tardiness = (old.exit_time - block.due_date).max(0);
    let min_t = block.release_time;
    let max_t = i64::MAX;

    for dist in (1..=MOVE_MAX_SHIFT_X + MOVE_MAX_SHIFT_Y).rev() {
        let min_abs_dx = (dist - MOVE_MAX_SHIFT_Y).max(0);
        let max_abs_dx = dist.min(MOVE_MAX_SHIFT_X);

        for abs_dx in min_abs_dx..=max_abs_dx {
            let abs_dy = dist - abs_dx;
            let x = old.x - abs_dx;
            let y = old.y - abs_dy;
            if !range.contains(x, y) {
                continue;
            }

            let tentative = ScheduledBlock {
                x,
                y,
                entry_time: 0,
                exit_time: block.processing_time,
                ..old
            };

            let Some(entry_time) = get_insert_t(pre, tentative, &base, min_t, max_t) else {
                continue;
            };

            let moved = ScheduledBlock {
                entry_time,
                exit_time: entry_time + block.processing_time,
                ..tentative
            };
            let new_tardiness = (moved.exit_time - block.due_date).max(0);
            if new_tardiness > original_tardiness {
                continue;
            }

            let mut candidate = base;
            candidate.push(moved);
            return Some(candidate);
        }
    }

    None
}

fn build_initial_schedule<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    worker_id: usize,
    worker_count: usize,
    rng: &mut R,
) -> Result<Vec<ScheduledBlock>, String> {
    fn initial_order_score(
        problem: &Problem,
        pre: &Precompute,
        block_id: usize,
        max_area: f64,
        max_due: i64,
        due_span: f64,
        area_weight: f64,
    ) -> f64 {
        let area_norm = pre.block_area[block_id] / max_area;
        let due_urgency = (max_due - problem.blocks[block_id].due_date) as f64 / due_span;
        due_urgency + area_weight * area_norm
    }

    fn sort_initial_order(
        problem: &Problem,
        pre: &Precompute,
        order: &mut [usize],
        worker_id: usize,
        worker_count: usize,
    ) {
        let max_area = pre.block_area.iter().copied().fold(0.0, f64::max).max(1.0);
        let min_due = problem
            .blocks
            .iter()
            .map(|block| block.due_date)
            .min()
            .unwrap_or(0);
        let max_due = problem
            .blocks
            .iter()
            .map(|block| block.due_date)
            .max()
            .unwrap_or(min_due);
        let due_span = (max_due - min_due).max(1) as f64;
        let w = worker_id as f64 / (worker_count - 1).max(1) as f64;
        let area_weight =
            INITIAL_AREA_WEIGHT_MIN + (INITIAL_AREA_WEIGHT_MAX - INITIAL_AREA_WEIGHT_MIN) * w;

        order.sort_by(|&a, &b| {
            let score_a =
                initial_order_score(problem, pre, a, max_area, max_due, due_span, area_weight);
            let score_b =
                initial_order_score(problem, pre, b, max_area, max_due, due_span, area_weight);
            score_b
                .total_cmp(&score_a)
                .then(problem.blocks[a].due_date.cmp(&problem.blocks[b].due_date))
                .then(pre.block_area[b].total_cmp(&pre.block_area[a]))
                .then(a.cmp(&b))
        });
    }

    let mut schedule = Vec::with_capacity(problem.blocks.len());
    let mut loads = vec![0.0; problem.bays.len()];
    let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
    sort_initial_order(problem, pre, &mut order, worker_id, worker_count);

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
        let scheduled = insert_greedy(
            problem,
            pre,
            original,
            &schedule,
            &loads,
            INSERT_PARAMS,
            rng,
        )
        .ok_or_else(|| format!("failed to place block {block_id} in initial schedule"))?;
        loads[scheduled.bay_id] += block.workload as f64;
        schedule.push(scheduled);
    }

    Ok(schedule)
}

fn sample_order_slack_weight<R: Random>(rng: &mut R) -> f64 {
    ORDER_SLACK_WEIGHT_MIN + (ORDER_SLACK_WEIGHT_MAX - ORDER_SLACK_WEIGHT_MIN) * rng.nextf()
}

fn block_slack(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    block.due_date - block.release_time - block.processing_time
}

fn insert_order_score(
    problem: &Problem,
    pre: &Precompute,
    s: ScheduledBlock,
    max_area: f64,
    max_slack: i64,
    slack_span: f64,
    slack_weight: f64,
) -> f64 {
    let area_norm = pre.block_area[s.block_id] / max_area;
    let slack_urgency = (max_slack - block_slack(problem, s.block_id)) as f64 / slack_span;
    area_norm + slack_weight * slack_urgency
}

fn sort_removed_by_area_slack(
    problem: &Problem,
    pre: &Precompute,
    removed: &mut [ScheduledBlock],
    slack_weight: f64,
) {
    let max_area = removed
        .iter()
        .map(|s| pre.block_area[s.block_id])
        .fold(0.0, f64::max)
        .max(1.0);
    let min_slack = removed
        .iter()
        .map(|s| block_slack(problem, s.block_id))
        .min()
        .unwrap_or(0);
    let max_slack = removed
        .iter()
        .map(|s| block_slack(problem, s.block_id))
        .max()
        .unwrap_or(min_slack);
    let slack_span = (max_slack - min_slack).max(1) as f64;

    removed.sort_by(|a, b| {
        let score_a = insert_order_score(
            problem,
            pre,
            *a,
            max_area,
            max_slack,
            slack_span,
            slack_weight,
        );
        let score_b = insert_order_score(
            problem,
            pre,
            *b,
            max_area,
            max_slack,
            slack_span,
            slack_weight,
        );
        score_b
            .total_cmp(&score_a)
            .then(block_slack(problem, a.block_id).cmp(&block_slack(problem, b.block_id)))
            .then(pre.block_area[b.block_id].total_cmp(&pre.block_area[a.block_id]))
            .then(
                problem.blocks[a.block_id]
                    .due_date
                    .cmp(&problem.blocks[b.block_id].due_date),
            )
            .then(a.block_id.cmp(&b.block_id))
    });
}

fn try_remove_reinsert<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    removed_ids: &[usize],
    order_slack_weight: f64,
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

    sort_removed_by_area_slack(problem, pre, &mut removed, order_slack_weight);

    let mut cur = base;
    let mut loads = vec![0.0; problem.bays.len()];
    for s in &cur {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }

    for old in removed {
        let scheduled = insert_greedy(problem, pre, old, &cur, &loads, INSERT_PARAMS, rng)?;
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        cur.push(scheduled);
    }

    Some(cur)
}

fn insert_greedy<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    schedule: &[ScheduledBlock],
    loads: &[f64],
    params: InsertSearchParams,
    _rng: &mut R,
) -> Option<ScheduledBlock> {
    let block_id = original.block_id;
    let block = &problem.blocks[block_id];
    let process_t = block.processing_time;
    let min_t = block.release_time;
    let max_t = i64::MAX;

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

        for &orient_idx in &pre.orientation_order_by_bbox[block_id] {
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            let mut anchor_x: Option<i64> = None;

            for x in (range.min_x..=range.max_x).step_by(params.x_step as usize) {
                if let Some(anchor_x) = anchor_x {
                    if x > anchor_x + INSERT_X_BUFFER {
                        break;
                    }
                }

                for y in (range.min_y..=range.max_y).step_by(params.y_step as usize) {
                    let tentative = ScheduledBlock {
                        block_id,
                        bay_id,
                        orient_idx,
                        x,
                        y,
                        entry_time: 0,
                        exit_time: process_t,
                    };
                    let Some(entry_time) = get_insert_t(pre, tentative, schedule, min_t, max_t)
                    else {
                        continue;
                    };
                    let scheduled = ScheduledBlock {
                        entry_time,
                        exit_time: entry_time + process_t,
                        ..tentative
                    };
                    let tardiness = (scheduled.exit_time - block.due_date).max(0);
                    let score_delta = problem.weights.w1 * tardiness as f64 + delta_obj23;

                    let bounds = pre.orientation_bbox_bounds[block_id][orient_idx];
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
                        if anchor_x.is_none() {
                            anchor_x = Some(x);
                        }
                    }
                }
            }
        }
    }

    best.map(|candidate| candidate.scheduled)
}

fn insert_candidate_better(a: &InsertCandidate, b: &InsertCandidate) -> bool {
    a.score_delta
        .total_cmp(&b.score_delta)
        .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
        .then(a.bbox_right.total_cmp(&b.bbox_right))
        .then(a.bbox_top.total_cmp(&b.bbox_top))
        .then(a.scheduled.block_id.cmp(&b.scheduled.block_id))
        .is_lt()
}

fn clamp_interval(l: i64, r: i64, min_val: i64, max_val: i64) -> Option<Interval> {
    let l = l.max(min_val);
    let r = r.min(max_val);
    if l <= r { Some((l, r)) } else { None }
}

fn add_forbidden_intervals_for_old(
    pre: &Precompute,
    new_block: ScheduledBlock,
    old: ScheduledBlock,
    min_t: i64,
    max_t: i64,
    forbidden: &mut Vec<Interval>,
) {
    let p = new_block.exit_time - new_block.entry_time;
    let a = old.entry_time;
    let b = old.exit_time;

    let Some((ol, or)) = clamp_interval(a - p + 1, b - 1, min_t, max_t) else {
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

    if !new_old_clear && !old_new_clear {
        forbidden.push((ol, or));
        return;
    }

    let (allow_l, allow_r) = if new_old_clear {
        ((a + 1).max(ol), (b - p - 1).min(or))
    } else {
        ((b - p + 1).max(ol), (a - 1).min(or))
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

fn first_feasible_time(mut forbidden: Vec<Interval>, min_t: i64, max_t: i64) -> Option<i64> {
    merge_intervals(&mut forbidden);
    let mut t = min_t;
    for (l, r) in forbidden {
        if t < l {
            return Some(t);
        }
        if t <= r {
            t = r + 1;
        }
        if t > max_t {
            return None;
        }
    }
    if t <= max_t { Some(t) } else { None }
}

fn get_insert_t(
    pre: &Precompute,
    new_block: ScheduledBlock,
    schedule: &[ScheduledBlock],
    min_t: i64,
    max_t: i64,
) -> Option<i64> {
    let mut forbidden = Vec::new();
    for &old in schedule.iter().filter(|old| old.bay_id == new_block.bay_id) {
        add_forbidden_intervals_for_old(pre, new_block, old, min_t, max_t, &mut forbidden);
    }
    first_feasible_time(forbidden, min_t, max_t)
}

fn scheduled_center(pre: &Precompute, s: ScheduledBlock) -> (f64, f64) {
    let (cx, cy) = pre.orientation_bbox_center[s.block_id][s.orient_idx];
    (s.x as f64 + cx, s.y as f64 + cy)
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
        let (sx, sy) = scheduled_center(pre, seed);
        let mut neighbors: Vec<(f64, usize)> = schedule
            .iter()
            .filter(|s| s.bay_id == seed.bay_id && !used[s.block_id])
            .map(|&s| {
                let (x, y) = scheduled_center(pre, s);
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
