use crate::{
    annealing::AnnealingParams,
    insert::{InsertSearchParams, insert_greedy, try_place_block},
    precompute::Precompute,
    preoptimize::{PreoptimizeParams, PreoptimizePrecompute, preoptimize},
    solver_util::{
        NeighborKind, bay_tardiness, block_pref_spread, block_slack, gen_rangef,
        schedule_to_solution, score_schedule, score13_block,
    },
    util::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
    *,
};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

macro_rules! log {
    ($timer:expr, $($arg:tt)*) => {
        eprintln!("[{:.4}] {}", $timer.elapsed_seconds(), format_args!($($arg)*))
    };
}

pub mod annealing;

pub use annealing::{BayAnnealing, BayOptimizeState, GlobalAnnealing};

const RNG_SEED: u64 = 1;
const MAX_WORKER_COUNT: usize = 4;

const LOCAL_SEARCH_TIME_BUFFER_SECONDS: f64 = 3.;
const GLOBAL_ANNEALING_REMAINING_SECONDS: f64 = 20.;

pub const PRECOMPUTE_ORIENTATION_NEIGHBOR_LIMIT: usize = 100;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_AREA_TOP_K: usize = 16;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_ALIGN_DELTA: i64 = 3;

const INITIAL_PREOPTIMIZE_ALPHA: f64 = 1.0;
const INITIAL_PREOPTIMIZE_BETA: f64 = 1.0;
const INITIAL_PREOPTIMIZE_TIME_RATIO: f64 = 0.1;
const INITIAL_PREOPTIMIZE_MAX_SECONDS: f64 = 10.0;
pub const PREOPTIMIZE_DISTANCE_WEIGHT: f64 = 0.0;
pub const PREOPTIMIZE_RNG_SEED: u64 = 2;
pub const PREOPTIMIZE_SWAP_PROBABILITY: f64 = 0.15;
pub const PREOPTIMIZE_BAD_BLOCK_SAMPLE_COUNT: usize = 8;
pub const PREOPTIMIZE_BAD_BLOCK_SELECT_PROBABILITY: f64 = 0.75;
pub const PREOPTIMIZE_MAX_RELOCATE_ATTEMPTS: usize = 8;
pub const PREOPTIMIZE_MAX_TIME_SHIFT: i64 = 10;
pub const PREOPTIMIZE_END_TEMPERATURE_RATIO: f64 = 1e-4;

const GLOBAL_ANNEALING_PARAMS: AnnealingParams = AnnealingParams {
    exchange_interval: 2048,
    start_temperature: 1e1,
    end_temperature: 1e-2,
    worker_temperature_scale: 10.,
    tabu_capacity: 4096,
};
const BAY_ANNEALING_PARAMS: AnnealingParams = AnnealingParams {
    exchange_interval: 2048,
    start_temperature: 1e-2,
    end_temperature: 1e-4,
    worker_temperature_scale: 1.,
    tabu_capacity: 4096,
};

const BAY_GREEDY_ORDER_TRIALS: usize = 32;
const BAY_GREEDY_MAX_DUPLICATE_TRIALS: usize = 128;
const NEIGHBOR_KINDS: &[&str] = &["Large", "Shift", "Move", "Rotate", "Swap"];

const GLOBAL_NEIGHBOR_PROBS: &[(NeighborKind, f64)] = &[
    (NeighborKind::LargeReconstruct, 0.2),
    (NeighborKind::Shift, 8.0),
    (NeighborKind::Move, 0.1),
    (NeighborKind::Rotate, 8.0),
    (NeighborKind::Swap, 3.0),
];
const BAY_NEIGHBOR_PROBS: &[(NeighborKind, f64)] = &[
    (NeighborKind::LargeReconstruct, 0.2),
    (NeighborKind::Shift, 8.0),
    (NeighborKind::Move, 0.1),
    (NeighborKind::Rotate, 8.0),
    (NeighborKind::Swap, 0.0),
];

pub struct NeighborParams {
    pub probabilities: &'static [(NeighborKind, f64)],
    pub min_removed_blocks: usize,
    pub max_removed_blocks: usize,
    pub remove_pool_factor: usize,
    pub remove_count_sample_power: f64,
    pub remove_seed_per_block: usize,
    pub remove_random_seed_ratio: f64,
    pub remove_x_distance_weight_max: f64,
    pub remove_y_distance_weight_max: f64,
    pub reconstruct_workload_weight_range: (f64, f64),
    pub reconstruct_area_weight_range: (f64, f64),
    pub reconstruct_pref_spread_weight_range: (f64, f64),
    pub reconstruct_due_urgency_weight_range: (f64, f64),
    pub reconstruct_slack_urgency_weight_range: (f64, f64),
    pub reconstruct_order_random_weight_range: (f64, f64),
    pub insert_y_buffer: i64,
    pub shift_max_x: i64,
    pub shift_max_y: i64,
    pub rotate_max_shift_delta: i64,
    pub swap_neighbor_top_k: usize,
    pub swap_max_shift_delta: i64,
    pub move_sample_blocks: usize,
    pub move_small_pool_size: usize,
}

const GLOBAL_NEIGHBOR_PARAMS: NeighborParams = NeighborParams {
    probabilities: GLOBAL_NEIGHBOR_PROBS,
    min_removed_blocks: 7,
    max_removed_blocks: 13,
    remove_pool_factor: 4,
    remove_count_sample_power: 2.0,
    remove_seed_per_block: 4,
    remove_random_seed_ratio: 0.25,
    remove_x_distance_weight_max: 3.0,
    remove_y_distance_weight_max: 3.0,
    reconstruct_workload_weight_range: (0.0, 1.0),
    reconstruct_area_weight_range: (-0.2, 1.0),
    reconstruct_pref_spread_weight_range: (0.0, 1.0),
    reconstruct_due_urgency_weight_range: (0.0, 1.0),
    reconstruct_slack_urgency_weight_range: (0.0, 1.0),
    reconstruct_order_random_weight_range: (0.0, 0.5),
    insert_y_buffer: 10,
    shift_max_x: 5,
    shift_max_y: 5,
    rotate_max_shift_delta: 2,
    swap_neighbor_top_k: 32,
    swap_max_shift_delta: 2,
    move_sample_blocks: 16,
    move_small_pool_size: 8,
};

const BAY_NEIGHBOR_PARAMS: NeighborParams = NeighborParams {
    probabilities: BAY_NEIGHBOR_PROBS,
    ..GLOBAL_NEIGHBOR_PARAMS
};

fn sample_reconstruct_order_weights(
    rng: &mut impl Random,
    params: &NeighborParams,
) -> BlockOrderWeights {
    BlockOrderWeights {
        workload: gen_rangef(rng, params.reconstruct_workload_weight_range),
        area: gen_rangef(rng, params.reconstruct_area_weight_range),
        pref_spread: gen_rangef(rng, params.reconstruct_pref_spread_weight_range),
        due_urgency: gen_rangef(rng, params.reconstruct_due_urgency_weight_range),
        slack_urgency: gen_rangef(rng, params.reconstruct_slack_urgency_weight_range),
        random: gen_rangef(rng, params.reconstruct_order_random_weight_range),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PreoptimizedBlock {
    pub bay_id: usize,
    pub entry_time: i64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PreoptimizeState {
    pub score: f64,
    pub blocks: Vec<PreoptimizedBlock>,
}

impl crate::annealing::AnnealingState for PreoptimizeState {
    fn annealing_score(&self) -> f64 {
        self.score
    }

    fn tabu_key(&self) -> Option<u64> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct OptimizeState {
    pub score: f64,
    pub blocks: Vec<ScheduledBlock>,
}

#[derive(Clone, Copy)]
struct BlockOrderWeights {
    workload: f64,
    area: f64,
    pref_spread: f64,
    due_urgency: f64,
    slack_urgency: f64,
    random: f64,
}

struct BlockOrderContext {
    max_workload: f64,
    max_area: f64,
    max_pref_spread: f64,
    max_due: i64,
    due_span: f64,
    max_slack: i64,
    slack_span: f64,
}

struct PrecedenceConstraints {
    befores: Vec<Vec<usize>>,
    afters: Vec<Vec<usize>>,
}

pub fn to_preoptimize_state(state: &OptimizeState) -> PreoptimizeState {
    let mut ordered = state.blocks.clone();
    ordered.sort_unstable_by_key(|scheduled| scheduled.block_id);
    PreoptimizeState {
        score: state.score,
        blocks: ordered
            .into_iter()
            .map(|scheduled| PreoptimizedBlock {
                bay_id: scheduled.bay_id,
                entry_time: scheduled.entry_time,
            })
            .collect(),
    }
}

fn preoptimize_time_limit(timelimit: f64, ratio: f64, max_seconds: f64) -> f64 {
    (timelimit * ratio).min(max_seconds).max(1e-4)
}

pub fn solve(problem: &Problem, timelimit: f64, timer: Timer) -> Result<Solution, String> {
    let deadline = timelimit - LOCAL_SEARCH_TIME_BUFFER_SECONDS;

    log!(timer, "building preoptimize precompute...");
    let preoptimize_pre = PreoptimizePrecompute::build(problem)?;
    log!(timer, "preoptimize precompute built");
    log!(timer, "building precompute...");
    let pre = Precompute::build(problem);
    log!(timer, "precompute built");

    let initial_time_limit = preoptimize_time_limit(
        timelimit,
        INITIAL_PREOPTIMIZE_TIME_RATIO,
        INITIAL_PREOPTIMIZE_MAX_SECONDS,
    )
    .min((deadline - timer.elapsed_seconds()).max(1e-4));
    let initial_abstract = preoptimize(
        problem,
        &preoptimize_pre,
        None,
        PreoptimizeParams {
            alpha: INITIAL_PREOPTIMIZE_ALPHA,
            beta: INITIAL_PREOPTIMIZE_BETA,
            time_limit: initial_time_limit,
        },
    )?;
    log!(
        timer,
        "initial abstract score: {:.3}",
        initial_abstract.score
    );
    let initial = build_optimize_state(problem, &pre, &initial_abstract)
        .ok_or_else(|| "failed to build initial optimize state".to_string())?;
    log!(timer, "initial optimize score: {:.3}", initial.score);

    let bay_deadline = deadline - GLOBAL_ANNEALING_REMAINING_SECONDS;
    let initial = BayAnnealing::new(problem, &pre, &initial_abstract, timer).run(
        initial,
        bay_deadline,
        BAY_ANNEALING_PARAMS,
        RNG_SEED,
    );
    log!(timer, "bay annealing score: {:.3}", initial.score);

    let best = GlobalAnnealing::new(problem, &pre, timer).run(
        initial,
        deadline,
        GLOBAL_ANNEALING_PARAMS,
        RNG_SEED,
    );
    Ok(schedule_to_solution(&best.blocks))
}

/// TODO: rayによる並列化
pub fn build_optimize_state(
    problem: &Problem,
    pre: &Precompute,
    state: &PreoptimizeState,
) -> Option<OptimizeState> {
    let constraints = build_precedence_constraints(problem, state);
    let mut blocks_by_bay = vec![Vec::new(); problem.bays.len()];
    for (block_id, block) in state.blocks.iter().enumerate() {
        blocks_by_bay[block.bay_id].push(block_id);
    }

    let mut blocks = Vec::with_capacity(problem.blocks.len());
    for (bay_id, bay_block_ids) in blocks_by_bay.iter().enumerate() {
        if bay_block_ids.is_empty() {
            continue;
        }
        let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(20_000).wrapping_add(bay_id as u64));
        let mut seen_order_hashes = HashSet::new();
        let mut duplicate_trials = 0;
        let mut best: Option<(i64, Vec<ScheduledBlock>)> = None;
        while seen_order_hashes.len() < BAY_GREEDY_ORDER_TRIALS
            && duplicate_trials < BAY_GREEDY_MAX_DUPLICATE_TRIALS
        {
            let weights = sample_reconstruct_order_weights(&mut rng, &BAY_NEIGHBOR_PARAMS);
            let order = build_topological_order(
                problem,
                pre,
                bay_block_ids,
                &constraints,
                weights,
                &mut rng,
            );
            if !seen_order_hashes.insert(hash_order(&order)) {
                duplicate_trials += 1;
                continue;
            }
            let Some(schedule) = build_bay_schedule(
                problem,
                pre,
                bay_id,
                &order,
                &constraints,
                &BAY_NEIGHBOR_PARAMS,
            ) else {
                continue;
            };
            let tardiness = bay_tardiness(problem, &schedule);
            if best
                .as_ref()
                .is_none_or(|(best_tardiness, _)| tardiness < *best_tardiness)
            {
                best = Some((tardiness, schedule));
            }
        }
        blocks.extend(best?.1);
    }

    let score = score_schedule(problem, pre, &blocks);
    Some(OptimizeState { score, blocks })
}

fn hash_order(order: &[usize]) -> u64 {
    let mut hash = 1469598103934665603u64;
    for &block_id in order {
        hash ^= block_id as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn try_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng, params).min(problem.blocks.len());

    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng, params);
    if removed_ids.is_empty() {
        return None;
    }
    let w = sample_reconstruct_order_weights(rng, params);
    sort_block_order(problem, pre, &mut removed_ids, w, rng);

    let mut removed_ordered = vec![None; removed_ids.len()];
    let mut base = Vec::with_capacity(schedule.len() - removed_ids.len());
    for &s in schedule {
        if let Some(pos) = removed_ids.iter().position(|&id| id == s.block_id) {
            removed_ordered[pos] = Some(s);
        } else {
            base.push(s);
        }
    }

    let mut cur = base;
    let mut loads = vec![0.0; problem.bays.len()];
    let mut fixed_score13 = 0.0;
    for &s in &cur {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
        fixed_score13 += score13_block(problem, pre, s);
    }

    for old in removed_ordered {
        if fixed_score13 > accept_threshold + 1e-9 {
            return None;
        }
        let old = old?;
        let scheduled = insert_greedy(
            problem,
            pre,
            old,
            i64::MIN,
            i64::MAX,
            &cur,
            &loads,
            InsertSearchParams {
                y_buffer: params.insert_y_buffer,
            },
            &pre.bay_order_by_pref[old.block_id],
        )?;
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        fixed_score13 += score13_block(problem, pre, scheduled);
        cur.push(scheduled);
    }

    Some(cur)
}

fn scheduled_by_id(problem: &Problem, schedule: &[ScheduledBlock]) -> Vec<Option<ScheduledBlock>> {
    let mut result = vec![None; problem.blocks.len()];
    for &scheduled in schedule {
        result[scheduled.block_id] = Some(scheduled);
    }
    result
}

fn precedence_entry_time_range(
    problem: &Problem,
    constraints: &PrecedenceConstraints,
    scheduled_by_id: &[Option<ScheduledBlock>],
    block_id: usize,
) -> Option<(i64, i64)> {
    let block = &problem.blocks[block_id];
    let mut min_entry_time = block.release_time;
    for &before in &constraints.befores[block_id] {
        min_entry_time = min_entry_time.max(scheduled_by_id[before]?.exit_time);
    }

    let mut max_entry_time = i64::MAX;
    for &after in &constraints.afters[block_id] {
        if let Some(after) = scheduled_by_id[after] {
            max_entry_time = max_entry_time.min(after.entry_time - block.processing_time);
        }
    }
    (min_entry_time <= max_entry_time).then_some((min_entry_time, max_entry_time))
}

fn try_bay_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    constraints: &PrecedenceConstraints,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    bay_id: usize,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng, params).min(schedule.len());
    let removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng, params);
    if removed_ids.is_empty() {
        return None;
    }

    let weights = sample_reconstruct_order_weights(rng, params);
    let order = build_topological_order(problem, pre, &removed_ids, constraints, weights, rng);
    let original_by_id = scheduled_by_id(problem, schedule);
    let mut removed = vec![false; problem.blocks.len()];
    for &block_id in &removed_ids {
        removed[block_id] = true;
    }

    let mut cur = Vec::with_capacity(schedule.len());
    let mut loads = vec![0.0; problem.bays.len()];
    let mut fixed_tardiness = 0.0;
    for &scheduled in schedule {
        if removed[scheduled.block_id] {
            continue;
        }
        loads[bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        fixed_tardiness +=
            (scheduled.exit_time - problem.blocks[scheduled.block_id].due_date).max(0) as f64;
        cur.push(scheduled);
    }
    let mut current_by_id = scheduled_by_id(problem, &cur);
    let bay_order = [bay_id];

    for block_id in order {
        if fixed_tardiness > accept_threshold + 1e-9 {
            return None;
        }
        let old = original_by_id[block_id]?;
        let (min_entry_time, max_entry_time) =
            precedence_entry_time_range(problem, constraints, &current_by_id, block_id)?;
        let scheduled = insert_greedy(
            problem,
            pre,
            old,
            min_entry_time,
            max_entry_time,
            &cur,
            &loads,
            InsertSearchParams {
                y_buffer: params.insert_y_buffer,
            },
            &bay_order,
        )?;
        loads[bay_id] += problem.blocks[block_id].workload as f64;
        fixed_tardiness += (scheduled.exit_time - problem.blocks[block_id].due_date).max(0) as f64;
        current_by_id[block_id] = Some(scheduled);
        cur.push(scheduled);
    }

    Some(cur)
}

fn update_best(
    problem: &Problem,
    best: &mut Option<(i64, i64, ScheduledBlock)>,
    candidate: ScheduledBlock,
) {
    let block = &problem.blocks[candidate.block_id];
    let tardiness = (candidate.exit_time - block.due_date).max(0);
    if best
        .as_ref()
        .map_or(true, |&(best_tardiness, best_entry_time, _)| {
            (tardiness, candidate.entry_time) < (best_tardiness, best_entry_time)
        })
    {
        *best = Some((tardiness, candidate.entry_time, candidate));
    }
}

fn try_shift_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    constraints: Option<&PrecedenceConstraints>,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];

    let mut base = Vec::with_capacity(schedule.len());
    for (i, &s) in schedule.iter().enumerate() {
        if i != idx {
            base.push(s);
        }
    }

    let (min_entry_time, max_entry_time) = if let Some(constraints) = constraints {
        let by_id = scheduled_by_id(problem, &base);
        precedence_entry_time_range(problem, constraints, &by_id, old.block_id)?
    } else {
        (i64::MIN, i64::MAX)
    };
    let range = pre
        .collision
        .fit_range(old.bay_id, old.block_id, old.orient_idx)?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;

    for dist in (1..=params.shift_max_x + params.shift_max_y).rev() {
        let min_abs_dx = (dist - params.shift_max_y).max(0);
        let max_abs_dx = dist.min(params.shift_max_x);

        for abs_dx in min_abs_dx..=max_abs_dx {
            let abs_dy = dist - abs_dx;
            let x = old.x - abs_dx;
            let y = old.y - abs_dy;
            if !range.contains(x, y) {
                continue;
            }

            let Some(moved) = try_place_block(
                problem,
                pre,
                &base,
                old.block_id,
                old.bay_id,
                old.orient_idx,
                x,
                y,
                min_entry_time,
                max_entry_time,
            ) else {
                continue;
            };
            if moved == old {
                continue;
            }
            update_best(problem, &mut best, moved);
        }
    }

    let moved = best?.2;
    base.push(moved);
    Some(base)
}

fn try_rotate_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    constraints: Option<&PrecedenceConstraints>,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];
    let block = &problem.blocks[old.block_id];
    if block.shape.len() <= 1 {
        return None;
    }

    let mut base = Vec::with_capacity(schedule.len());
    for (i, &s) in schedule.iter().enumerate() {
        if i != idx {
            base.push(s);
        }
    }

    let (min_entry_time, max_entry_time) = if let Some(constraints) = constraints {
        let by_id = scheduled_by_id(problem, &base);
        precedence_entry_time_range(problem, constraints, &by_id, old.block_id)?
    } else {
        (i64::MIN, i64::MAX)
    };
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;
    for &(orient_idx, dx, dy) in &pre.orientation_neighbors[old.block_id][old.orient_idx] {
        let Some(range) = pre
            .collision
            .fit_range(old.bay_id, old.block_id, orient_idx)
        else {
            continue;
        };

        for ddx in -params.rotate_max_shift_delta..=params.rotate_max_shift_delta {
            for ddy in -params.rotate_max_shift_delta..=params.rotate_max_shift_delta {
                let x = old.x + dx + ddx;
                let y = old.y + dy + ddy;
                if !range.contains(x, y) {
                    continue;
                }

                let Some(rotated) = try_place_block(
                    problem,
                    pre,
                    &base,
                    old.block_id,
                    old.bay_id,
                    orient_idx,
                    x,
                    y,
                    min_entry_time,
                    max_entry_time,
                ) else {
                    continue;
                };
                if rotated == old {
                    continue;
                }

                update_best(problem, &mut best, rotated);
            }
        }
    }

    let rotated = best?.2;
    base.push(rotated);
    Some(base)
}

fn try_swap_place(
    problem: &Problem,
    pre: &Precompute,
    old: ScheduledBlock,
    schedule: &[ScheduledBlock],
    bay_id: usize,
    orient_idx: usize,
    base_x: i64,
    base_y: i64,
    params: &NeighborParams,
) -> Option<ScheduledBlock> {
    let range = pre.collision.fit_range(bay_id, old.block_id, orient_idx)?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;

    for ddx in -params.swap_max_shift_delta..=params.swap_max_shift_delta {
        for ddy in -params.swap_max_shift_delta..=params.swap_max_shift_delta {
            let x = base_x + ddx;
            let y = base_y + ddy;
            if !range.contains(x, y) {
                continue;
            }

            let Some(scheduled) = try_place_block(
                problem,
                pre,
                schedule,
                old.block_id,
                bay_id,
                orient_idx,
                x,
                y,
                i64::MIN,
                i64::MAX,
            ) else {
                continue;
            };
            update_best(problem, &mut best, scheduled);
        }
    }

    Some(best?.2)
}

fn try_swap_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.len() < 2 {
        return None;
    }

    let a_idx = rng.gen_index(schedule.len());
    let a_old = schedule[a_idx];
    let candidates = &pre.other_block_neighbors[a_old.block_id][a_old.orient_idx];
    if candidates.is_empty() {
        return None;
    }
    let candidate_count = params.swap_neighbor_top_k.min(candidates.len());
    let candidate = candidates[rng.gen_index(candidate_count)];
    let b_idx = schedule
        .iter()
        .position(|s| s.block_id == candidate.block_id)?;
    if a_idx == b_idx {
        return None;
    }
    let b_old = schedule[b_idx];

    let mut cur = Vec::with_capacity(schedule.len());
    for (idx, &s) in schedule.iter().enumerate() {
        if idx != a_idx && idx != b_idx {
            cur.push(s);
        }
    }

    let a_base_x = b_old.x - candidate.dx;
    let a_base_y = b_old.y - candidate.dy;
    let a_new = try_swap_place(
        problem,
        pre,
        a_old,
        &cur,
        b_old.bay_id,
        a_old.orient_idx,
        a_base_x,
        a_base_y,
        params,
    )?;
    cur.push(a_new);

    let b_base_x = a_old.x + candidate.dx;
    let b_base_y = a_old.y + candidate.dy;
    let b_new = try_swap_place(
        problem,
        pre,
        b_old,
        &cur,
        a_old.bay_id,
        candidate.orient_idx,
        b_base_x,
        b_base_y,
        params,
    )?;
    cur.push(b_new);

    Some(cur)
}

fn try_move_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    constraints: Option<&PrecedenceConstraints>,
    fixed_bay_id: Option<usize>,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    fn move_obj13(problem: &Problem, pre: &Precompute, s: ScheduledBlock) -> f64 {
        let block = &problem.blocks[s.block_id];
        let tardiness = (s.exit_time - block.due_date).max(0) as f64;
        problem.weights.w1 * tardiness
            + problem.weights.w3 * pre.pref_penalty[s.block_id][s.bay_id] as f64
    }

    if schedule.is_empty() {
        return None;
    }

    let sample_count = params.move_sample_blocks.min(schedule.len());
    let mut indices: Vec<usize> = (0..schedule.len()).collect();
    rng.shuffle(&mut indices);
    indices.truncate(sample_count);
    indices.sort_by(|&a, &b| {
        pre.block_area[schedule[a].block_id]
            .total_cmp(&pre.block_area[schedule[b].block_id])
            .then(schedule[a].block_id.cmp(&schedule[b].block_id))
    });
    indices.truncate(params.move_small_pool_size.min(indices.len()));

    let idx = indices.into_iter().max_by(|&a, &b| {
        let sa = move_obj13(problem, pre, schedule[a]);
        let sb = move_obj13(problem, pre, schedule[b]);
        sa.total_cmp(&sb).then(
            pre.block_area[schedule[b].block_id].total_cmp(&pre.block_area[schedule[a].block_id]),
        )
    })?;
    let old = schedule[idx];

    let mut base = Vec::with_capacity(schedule.len());
    let mut loads = vec![0.0; problem.bays.len()];
    for (i, &s) in schedule.iter().enumerate() {
        if i == idx {
            continue;
        }
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
        base.push(s);
    }

    let (min_entry_time, max_entry_time) = if let Some(constraints) = constraints {
        let by_id = scheduled_by_id(problem, &base);
        precedence_entry_time_range(problem, constraints, &by_id, old.block_id)?
    } else {
        (i64::MIN, i64::MAX)
    };
    let fixed_bay_order = fixed_bay_id.map(|bay_id| [bay_id]);
    let bay_order = fixed_bay_order
        .as_ref()
        .map_or(pre.bay_order_by_pref[old.block_id].as_slice(), |order| {
            order.as_slice()
        });
    let scheduled = insert_greedy(
        problem,
        pre,
        old,
        min_entry_time,
        max_entry_time,
        &base,
        &loads,
        InsertSearchParams {
            y_buffer: params.insert_y_buffer,
        },
        bay_order,
    )?;
    if scheduled == old {
        return None;
    }

    base.push(scheduled);
    Some(base)
}

fn sample_removed_count<R: Random>(rng: &mut R, params: &NeighborParams) -> usize {
    let span = params.max_removed_blocks - params.min_removed_blocks + 1;
    let u = rng.nextf().powf(params.remove_count_sample_power);
    params.min_removed_blocks + ((u * span as f64) as usize).min(span - 1)
}

fn choose_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    k: usize,
    rng: &mut R,
    params: &NeighborParams,
) -> Vec<usize> {
    fn scheduled_center(pre: &Precompute, s: ScheduledBlock) -> (f64, f64) {
        let (cx, cy) = pre.orientation_bbox_center[s.block_id][s.orient_idx];
        (s.x as f64 + cx, s.y as f64 + cy)
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

    let remove_x_distance_weight = rng.gen_rangef(0.0, params.remove_x_distance_weight_max);
    let remove_y_distance_weight = rng.gen_rangef(0.0, params.remove_y_distance_weight_max);

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
    let pool_len = (k * params.remove_pool_factor).min(ids.len()).max(k);
    ids.truncate(pool_len);
    let bad_pool = ids;

    let mut selected = Vec::with_capacity(k);
    let mut used = vec![false; problem.blocks.len()];
    let seed_count = k.div_ceil(params.remove_seed_per_block);
    let mut random_seed_count = 0;
    for _ in 0..seed_count {
        if rng.nextf() < params.remove_random_seed_ratio {
            random_seed_count += 1;
        }
    }
    let bad_seed_count = seed_count - random_seed_count;

    let mut seed_pool = bad_pool.clone();
    rng.shuffle(&mut seed_pool);
    for &block_id in seed_pool.iter().take(bad_seed_count) {
        push_selected(&mut selected, &mut used, block_id, k);
    }

    let mut random_seed_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut random_seed_pool);
    let random_seed_end = selected.len() + random_seed_count;
    for block_id in random_seed_pool {
        if selected.len() >= random_seed_end {
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
        let st = seed.entry_time as f64;
        let mut neighbors: Vec<(f64, usize)> = schedule
            .iter()
            .filter(|s| s.bay_id == seed.bay_id && !used[s.block_id])
            .map(|&s| {
                let (x, y) = scheduled_center(pre, s);
                let dx = x - sx;
                let dy = y - sy;
                let dt = s.entry_time as f64 - st;
                (
                    remove_x_distance_weight * dx * dx
                        + remove_y_distance_weight * dy * dy
                        + dt * dt,
                    s.block_id,
                )
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, block_id) in neighbors.into_iter().take(need) {
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

fn build_block_order_context(
    problem: &Problem,
    pre: &Precompute,
    order: &[usize],
) -> BlockOrderContext {
    let max_workload = order
        .iter()
        .map(|&block_id| problem.blocks[block_id].workload as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let max_area = order
        .iter()
        .map(|&block_id| pre.block_area[block_id])
        .fold(0.0, f64::max)
        .max(1.0);
    let max_pref_spread = order
        .iter()
        .map(|&block_id| block_pref_spread(problem, block_id) as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let min_due = order
        .iter()
        .map(|&block_id| problem.blocks[block_id].due_date)
        .min()
        .unwrap_or(0);
    let max_due = order
        .iter()
        .map(|&block_id| problem.blocks[block_id].due_date)
        .max()
        .unwrap_or(min_due);
    let min_slack = order
        .iter()
        .map(|&block_id| block_slack(problem, block_id))
        .min()
        .unwrap_or(0);
    let max_slack = order
        .iter()
        .map(|&block_id| block_slack(problem, block_id))
        .max()
        .unwrap_or(min_slack);

    BlockOrderContext {
        max_workload,
        max_area,
        max_pref_spread,
        max_due,
        due_span: (max_due - min_due).max(1) as f64,
        max_slack,
        slack_span: (max_slack - min_slack).max(1) as f64,
    }
}

fn block_order_score(
    problem: &Problem,
    pre: &Precompute,
    ctx: &BlockOrderContext,
    weights: BlockOrderWeights,
    block_id: usize,
) -> f64 {
    let block = &problem.blocks[block_id];
    let workload_norm = block.workload as f64 / ctx.max_workload;
    let area_norm = pre.block_area[block_id] / ctx.max_area;
    let pref_spread_norm = block_pref_spread(problem, block_id) as f64 / ctx.max_pref_spread;
    let due_urgency = (ctx.max_due - block.due_date) as f64 / ctx.due_span;
    let slack_urgency = (ctx.max_slack - block_slack(problem, block_id)) as f64 / ctx.slack_span;

    weights.workload * workload_norm
        + weights.area * area_norm
        + weights.pref_spread * pref_spread_norm
        + weights.due_urgency * due_urgency
        + weights.slack_urgency * slack_urgency
}

fn sort_block_order<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    order: &mut [usize],
    weights: BlockOrderWeights,
    rng: &mut R,
) {
    let ctx = build_block_order_context(problem, pre, order);
    let mut random_scores = vec![0.0; problem.blocks.len()];
    for &block_id in order.iter() {
        random_scores[block_id] = rng.gen_rangef(0., weights.random);
    }
    order.sort_by(|&a, &b| {
        let score_a = block_order_score(problem, pre, &ctx, weights, a) + random_scores[a];
        let score_b = block_order_score(problem, pre, &ctx, weights, b) + random_scores[b];
        score_b
            .total_cmp(&score_a)
            .then(block_slack(problem, a).cmp(&block_slack(problem, b)))
            .then(problem.blocks[a].due_date.cmp(&problem.blocks[b].due_date))
            .then(pre.block_area[b].total_cmp(&pre.block_area[a]))
            .then(problem.blocks[b].workload.cmp(&problem.blocks[a].workload))
            .then(block_pref_spread(problem, b).cmp(&block_pref_spread(problem, a)))
            .then(a.cmp(&b))
    });
}

fn build_precedence_constraints(
    problem: &Problem,
    state: &PreoptimizeState,
) -> PrecedenceConstraints {
    let mut befores = vec![Vec::new(); problem.blocks.len()];
    let mut afters = vec![Vec::new(); problem.blocks.len()];
    for i in 0..problem.blocks.len() {
        for j in i + 1..problem.blocks.len() {
            let si = state.blocks[i];
            let sj = state.blocks[j];
            if si.bay_id != sj.bay_id {
                continue;
            }
            let end_i = si.entry_time + problem.blocks[i].processing_time;
            let end_j = sj.entry_time + problem.blocks[j].processing_time;
            let edge = if end_i <= sj.entry_time {
                Some((i, j))
            } else if end_j <= si.entry_time {
                Some((j, i))
            } else {
                None
            };
            if let Some((before, after)) = edge {
                afters[before].push(after);
                befores[after].push(before);
            }
        }
    }
    PrecedenceConstraints { befores, afters }
}

fn build_topological_order<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    bay_block_ids: &[usize],
    constraints: &PrecedenceConstraints,
    weights: BlockOrderWeights,
    rng: &mut R,
) -> Vec<usize> {
    let mut priority_order = bay_block_ids.to_vec();
    sort_block_order(problem, pre, &mut priority_order, weights, rng);
    let mut priority_rank = vec![usize::MAX; problem.blocks.len()];
    for (rank, &block_id) in priority_order.iter().enumerate() {
        priority_rank[block_id] = rank;
    }

    let mut included = vec![false; problem.blocks.len()];
    for &block_id in bay_block_ids {
        included[block_id] = true;
    }
    let mut indegree = vec![0usize; problem.blocks.len()];
    let mut ready = BinaryHeap::new();
    for &block_id in bay_block_ids {
        indegree[block_id] = constraints.befores[block_id]
            .iter()
            .filter(|&&before| included[before])
            .count();
        if indegree[block_id] == 0 {
            ready.push(Reverse((priority_rank[block_id], block_id)));
        }
    }

    let mut order = Vec::with_capacity(bay_block_ids.len());
    while let Some(Reverse((_, block_id))) = ready.pop() {
        order.push(block_id);
        for &after in &constraints.afters[block_id] {
            if !included[after] {
                continue;
            }
            indegree[after] -= 1;
            if indegree[after] == 0 {
                ready.push(Reverse((priority_rank[after], after)));
            }
        }
    }
    debug_assert_eq!(order.len(), bay_block_ids.len());
    order
}

fn build_bay_schedule(
    problem: &Problem,
    pre: &Precompute,
    bay_id: usize,
    order: &[usize],
    constraints: &PrecedenceConstraints,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    let mut schedule = Vec::with_capacity(order.len());
    let mut scheduled_by_id: Vec<Option<ScheduledBlock>> = vec![None; problem.blocks.len()];
    let mut loads = vec![0.0; problem.bays.len()];
    let bay_order = [bay_id];

    for &block_id in order {
        let block = &problem.blocks[block_id];
        let min_entry_time = constraints.befores[block_id]
            .iter()
            .map(|&before| scheduled_by_id[before].unwrap().exit_time)
            .fold(block.release_time, i64::max);
        let original = ScheduledBlock {
            block_id,
            bay_id,
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
            min_entry_time,
            i64::MAX,
            &schedule,
            &loads,
            InsertSearchParams {
                y_buffer: params.insert_y_buffer,
            },
            &bay_order,
        )?;
        loads[bay_id] += block.workload as f64;
        scheduled_by_id[block_id] = Some(scheduled);
        schedule.push(scheduled);
    }
    Some(schedule)
}
