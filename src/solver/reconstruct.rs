use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    sync::Mutex,
};

use rayon::prelude::*;

use crate::{
    Problem, ScheduledBlock,
    params::{InsertParams, ReconstructNeighborParams},
    utils::{
        random::{RandPcg64Mcg, Random, sample_weighted_index},
        time::Timer,
    },
};

use super::{
    PreoptimizeState,
    insert::{ScheduleView, Scratch as InsertScratch, insert_greedy},
    objective::{ScheduleScore, schedule_tardiness, score_schedule, score13_block},
    optimize::OptimizeState,
    precompute::Precompute,
};

#[derive(Clone, Copy)]
enum RemoveSeedMethod {
    Badness,
    Fluidity,
    Random,
}

#[derive(Clone, Copy)]
struct RemoveSeed {
    block_id: usize,
    remove_count: usize,
}

struct RemoveScratch {
    badness: Vec<i64>,
    fluidity: Vec<f64>,
    bad_pool: Vec<usize>,
    fluid_pool: Vec<usize>,
    seed_sizes: Vec<usize>,
    seed_methods: Vec<RemoveSeedMethod>,
    used: Vec<bool>,
    seeds: Vec<RemoveSeed>,
    entry_bases: Vec<usize>,
    seed_pool: Vec<usize>,
    selected: Vec<usize>,
    neighbors: Vec<(f64, usize)>,
    fill_pool: Vec<usize>,
}

impl RemoveScratch {
    fn new() -> Self {
        Self {
            badness: Vec::new(),
            fluidity: Vec::new(),
            bad_pool: Vec::new(),
            fluid_pool: Vec::new(),
            seed_sizes: Vec::new(),
            seed_methods: Vec::new(),
            used: Vec::new(),
            seeds: Vec::new(),
            entry_bases: Vec::new(),
            seed_pool: Vec::new(),
            selected: Vec::new(),
            neighbors: Vec::new(),
            fill_pool: Vec::new(),
        }
    }
}

struct OrderScratch {
    random_scores: Vec<f64>,
    priority_order: Vec<usize>,
    priority_rank: Vec<usize>,
    included: Vec<bool>,
    indegree: Vec<usize>,
    ready: BinaryHeap<Reverse<(usize, usize)>>,
    order: Vec<usize>,
}

impl OrderScratch {
    fn new() -> Self {
        Self {
            random_scores: Vec::new(),
            priority_order: Vec::new(),
            priority_rank: Vec::new(),
            included: Vec::new(),
            indegree: Vec::new(),
            ready: BinaryHeap::new(),
            order: Vec::new(),
        }
    }
}

pub(super) struct Scratch<'a> {
    insert: InsertScratch<'a>,
    remove: RemoveScratch,
    order: OrderScratch,
    bay_members: Vec<Vec<ScheduledBlock>>,
    removed: Vec<bool>,
    loads: Vec<f64>,
    current_penalties: Vec<f64>,
    original_by_id: Vec<Option<ScheduledBlock>>,
    current_by_id: Vec<Option<ScheduledBlock>>,
}

impl<'a> Scratch<'a> {
    pub(super) fn new() -> Self {
        Self {
            insert: InsertScratch::new(),
            remove: RemoveScratch::new(),
            order: OrderScratch::new(),
            bay_members: Vec::new(),
            removed: Vec::new(),
            loads: Vec::new(),
            current_penalties: Vec::new(),
            original_by_id: Vec::new(),
            current_by_id: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct BlockOrderWeights {
    workload: f64,
    volume: f64,
    pref_spread: f64,
    limit_time_urgency: f64,
    slack_tightness: f64,
    random: f64,
}

struct BlockOrderContext {
    max_workload: f64,
    max_volume: f64,
    max_pref_spread: f64,
    max_limit_time: i64,
    limit_time_span: f64,
    max_slack: i64,
    slack_span: f64,
}

pub(super) struct HeuristicPrecedence {
    pub(super) predecessors: Vec<Vec<usize>>,
    pub(super) successors: Vec<Vec<usize>>,
}

#[derive(Clone, Copy)]
pub(super) struct EntryTimeBounds {
    pub(super) min: i64,
    pub(super) max: i64,
}

pub(super) fn sample_reconstruct_order_weights(
    rng: &mut impl Random,
    params: &ReconstructNeighborParams,
) -> BlockOrderWeights {
    BlockOrderWeights {
        workload: rng.gen_range_f64(
            params.workload_weight_range.0,
            params.workload_weight_range.1,
        ),
        volume: rng.gen_range_f64(params.volume_weight_range.0, params.volume_weight_range.1),
        pref_spread: rng.gen_range_f64(
            params.pref_spread_weight_range.0,
            params.pref_spread_weight_range.1,
        ),
        limit_time_urgency: rng.gen_range_f64(
            params.limit_time_urgency_weight_range.0,
            params.limit_time_urgency_weight_range.1,
        ),
        slack_tightness: rng.gen_range_f64(
            params.slack_tightness_weight_range.0,
            params.slack_tightness_weight_range.1,
        ),
        random: rng.gen_range_f64(
            params.order_random_weight_range.0,
            params.order_random_weight_range.1,
        ),
    }
}

pub(super) fn build_optimize_state(
    problem: &Problem,
    pre: &Precompute,
    state: &PreoptimizeState,
    time_limit: f64,
    timer: Timer,
    max_worker_count: usize,
    seed: u64,
    neighbor_params: &ReconstructNeighborParams,
    insert_params: &InsertParams,
    precedence_margin: i64,
) -> Option<OptimizeState> {
    let constraints = build_heuristic_precedence(problem, state, precedence_margin);

    let mut blocks_by_bay = vec![Vec::new(); problem.bays.len()];
    for (block_id, block) in state.blocks.iter().enumerate() {
        blocks_by_bay[block.bay_id].push(block_id);
    }
    let active_bays: Vec<_> = blocks_by_bay
        .iter()
        .enumerate()
        .filter_map(|(bay_id, blocks)| (!blocks.is_empty()).then_some(bay_id))
        .collect();
    if active_bays.is_empty() {
        return Some(OptimizeState {
            objective: 0.0,
            total_tardiness: 0,
            schedule: Vec::new(),
        });
    }

    let build_deadline = timer.elapsed_seconds() + time_limit;
    let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
    let seen_order_hashes: Vec<_> = (0..problem.bays.len())
        .map(|_| Mutex::new(HashSet::new()))
        .collect();
    let bests: Vec<Mutex<Option<(i64, Vec<ScheduledBlock>)>>> =
        (0..problem.bays.len()).map(|_| Mutex::new(None)).collect();

    let worker_trials: Vec<_> = (0..worker_count)
        .into_par_iter()
        .map(|worker_id| {
            let mut rng =
                RandPcg64Mcg::new(seed.wrapping_add(1 << 16).wrapping_add(worker_id as u64));
            let mut turn = worker_id;
            let mut trials = 0usize;
            let mut order_scratch = OrderScratch::new();
            let mut insert_scratch = InsertScratch::new();
            let mut schedule = Vec::new();
            let mut scheduled_by_id = Vec::new();
            let mut loads = Vec::new();

            while timer.elapsed_seconds() < build_deadline {
                let bay_id = active_bays[turn % active_bays.len()];
                turn += 1;

                let weights = sample_reconstruct_order_weights(&mut rng, neighbor_params);
                fill_topological_order(
                    problem,
                    pre,
                    &blocks_by_bay[bay_id],
                    &constraints,
                    weights,
                    None,
                    0.0,
                    &mut rng,
                    &mut order_scratch,
                );
                let order_hash = hash_order(&order_scratch.order);
                if !seen_order_hashes[bay_id].lock().unwrap().insert(order_hash) {
                    continue;
                }

                if !build_bay_schedule(
                    problem,
                    pre,
                    bay_id,
                    &order_scratch.order,
                    &constraints,
                    insert_params,
                    &mut rng,
                    &mut schedule,
                    &mut scheduled_by_id,
                    &mut loads,
                    &mut insert_scratch,
                ) {
                    continue;
                }
                trials += 1;

                let tardiness = schedule_tardiness(problem, &schedule);
                let mut best = bests[bay_id].lock().unwrap();
                if best
                    .as_ref()
                    .is_none_or(|(best_tardiness, _)| tardiness < *best_tardiness)
                {
                    if let Some((best_tardiness, best_schedule)) = best.as_mut() {
                        *best_tardiness = tardiness;
                        best_schedule.clone_from(&schedule);
                    } else {
                        *best = Some((tardiness, schedule.clone()));
                    }
                    log!(
                        "[{:.4}] [build] best: worker={}, bay={}, tardiness={}",
                        timer.elapsed_seconds(),
                        worker_id,
                        bay_id,
                        tardiness,
                    );
                }
            }
            trials
        })
        .collect();

    for (worker_id, trials) in worker_trials.into_iter().enumerate() {
        log!(
            "[{:.4}] [build worker={}] trials={}",
            timer.elapsed_seconds(),
            worker_id,
            trials,
        );
    }

    let mut bests: Vec<_> = bests
        .into_iter()
        .map(|best| best.into_inner().unwrap())
        .collect();
    let mut blocks = Vec::with_capacity(problem.blocks.len());
    for bay_id in active_bays {
        let (tardiness, schedule) = bests[bay_id].take()?;
        log!(
            "[{:.4}] [build] result: bay={}, tardiness={}",
            timer.elapsed_seconds(),
            bay_id,
            tardiness,
        );
        blocks.extend(schedule);
    }

    let ScheduleScore {
        objective,
        total_tardiness,
    } = score_schedule(problem, pre, &blocks);
    log!(
        "[{:.4}] [build] finished: score={:.3}",
        timer.elapsed_seconds(),
        objective,
    );
    Some(OptimizeState {
        objective,
        total_tardiness,
        schedule: blocks,
    })
}

fn hash_order(order: &[usize]) -> u64 {
    let mut hash = 1469598103934665603u64;
    for &block_id in order {
        hash ^= block_id as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn build_reconstruct_base(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    scratch: &mut Scratch<'_>,
) -> (Vec<ScheduledBlock>, f64) {
    scratch.removed.resize(problem.blocks.len(), false);
    scratch.removed.fill(false);
    for &block_id in &scratch.remove.selected {
        scratch.removed[block_id] = true;
    }

    scratch.loads.resize(problem.bays.len(), 0.0);
    scratch.loads.fill(0.0);
    scratch
        .bay_members
        .resize_with(problem.bays.len(), Vec::new);
    for members in &mut scratch.bay_members {
        members.clear();
    }

    let mut remaining = Vec::with_capacity(schedule.len());
    let mut weighted_z1_z3 = 0.0;
    for &scheduled in schedule {
        if scratch.removed[scheduled.block_id] {
            continue;
        }
        scratch.loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        scratch.bay_members[scheduled.bay_id].push(scheduled);
        weighted_z1_z3 += score13_block(problem, pre, scheduled);
        remaining.push(scheduled);
    }

    (remaining, weighted_z1_z3)
}

pub(super) fn try_large_reconstruct<'a, R: Random>(
    problem: &Problem,
    pre: &'a Precompute,
    constraints: Option<&HeuristicPrecedence>,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    params: &ReconstructNeighborParams,
    insert_params: &InsertParams,
    scratch: &mut Scratch<'a>,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng, params).min(problem.blocks.len());

    scratch.original_by_id.resize(problem.blocks.len(), None);
    scratch.original_by_id.fill(None);
    for &scheduled in schedule {
        scratch.original_by_id[scheduled.block_id] = Some(scheduled);
    }
    if !choose_removed_blocks(
        problem,
        pre,
        schedule,
        &scratch.original_by_id,
        k,
        rng,
        params,
        &mut scratch.loads,
        &mut scratch.remove,
    ) {
        return None;
    }
    let weights = sample_reconstruct_order_weights(rng, params);
    let current_penalty_weight = rng.gen_range_f64(
        params.current_penalty_weight_range.0,
        params.current_penalty_weight_range.1,
    );
    scratch.current_penalties.resize(problem.blocks.len(), 0.0);
    scratch.current_penalties.fill(0.0);
    for &scheduled in schedule {
        scratch.current_penalties[scheduled.block_id] = score13_block(problem, pre, scheduled);
    }
    if let Some(constraints) = constraints {
        fill_topological_order(
            problem,
            pre,
            &scratch.remove.selected,
            constraints,
            weights,
            Some(&scratch.current_penalties),
            current_penalty_weight,
            rng,
            &mut scratch.order,
        );
    } else {
        scratch.order.order.clear();
        scratch
            .order
            .order
            .extend_from_slice(&scratch.remove.selected);
        sort_block_order_with_scratch(
            problem,
            &pre.max_footprint_area,
            &pre.pref_spread,
            &mut scratch.order.order,
            weights,
            Some(&scratch.current_penalties),
            current_penalty_weight,
            rng,
            &mut scratch.order.random_scores,
        );
    }

    let (mut cur, mut fixed_score13) = build_reconstruct_base(problem, pre, schedule, scratch);
    if constraints.is_some() {
        scratch.current_by_id.resize(problem.blocks.len(), None);
        scratch.current_by_id.fill(None);
        for &scheduled in &cur {
            scratch.current_by_id[scheduled.block_id] = Some(scheduled);
        }
    }

    for order_index in 0..scratch.order.order.len() {
        let block_id = scratch.order.order[order_index];
        if fixed_score13 > accept_threshold + 1e-9 {
            return None;
        }
        let old = scratch.original_by_id[block_id]?;
        let EntryTimeBounds {
            min: min_entry_time,
            max: max_entry_time,
        } = if let Some(constraints) = constraints {
            precedence_entry_time_bounds(problem, constraints, &scratch.current_by_id, block_id)?
        } else {
            EntryTimeBounds {
                min: i64::MIN,
                max: i64::MAX,
            }
        };
        let scheduled = insert_greedy(
            problem,
            pre,
            old,
            min_entry_time,
            max_entry_time,
            ScheduleView::ByBay(&scratch.bay_members),
            &scratch.loads,
            insert_params,
            &pre.bay_order_by_pref[old.block_id],
            params.insert_candidate_top_k,
            params.insert_candidate_select_p,
            rng,
            &mut scratch.insert,
        )?;
        scratch.loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        scratch.bay_members[scheduled.bay_id].push(scheduled);
        fixed_score13 += score13_block(problem, pre, scheduled);
        if constraints.is_some() {
            scratch.current_by_id[block_id] = Some(scheduled);
        }
        cur.push(scheduled);
    }

    Some(cur)
}

pub(super) fn scheduled_by_id(
    problem: &Problem,
    schedule: &[ScheduledBlock],
) -> Vec<Option<ScheduledBlock>> {
    let mut result = vec![None; problem.blocks.len()];
    for &scheduled in schedule {
        result[scheduled.block_id] = Some(scheduled);
    }
    result
}

pub(super) fn precedence_entry_time_bounds(
    problem: &Problem,
    constraints: &HeuristicPrecedence,
    scheduled_by_id: &[Option<ScheduledBlock>],
    block_id: usize,
) -> Option<EntryTimeBounds> {
    let block = &problem.blocks[block_id];
    let mut min_entry_time = block.release_time;
    for &before in &constraints.predecessors[block_id] {
        min_entry_time = min_entry_time.max(scheduled_by_id[before]?.exit_time);
    }

    let mut max_entry_time = i64::MAX;
    for &after in &constraints.successors[block_id] {
        if let Some(after) = scheduled_by_id[after] {
            max_entry_time = max_entry_time.min(after.entry_time - block.processing_time);
        }
    }
    (min_entry_time <= max_entry_time).then_some(EntryTimeBounds {
        min: min_entry_time,
        max: max_entry_time,
    })
}

pub(super) fn sample_removed_count<R: Random>(
    rng: &mut R,
    params: &ReconstructNeighborParams,
) -> usize {
    let span = params.max_removed_blocks - params.min_removed_blocks + 1;
    let u = rng.next_f64().powf(params.remove_count_sample_power);
    params.min_removed_blocks + ((u * span as f64) as usize).min(span - 1)
}

fn scheduled_center(pre: &Precompute, s: ScheduledBlock) -> (f64, f64) {
    let (cx, cy) = pre.orientation_bbox_center[s.block_id][s.orient_idx];
    (s.x as f64 + cx, s.y as f64 + cy)
}

fn push_removed_block(selected: &mut Vec<usize>, used: &mut [bool], block_id: usize, limit: usize) {
    if selected.len() < limit && !used[block_id] {
        selected.push(block_id);
        used[block_id] = true;
    }
}

fn fill_remove_badness(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    loads: &mut Vec<f64>,
    badness: &mut Vec<i64>,
) {
    loads.resize(problem.bays.len(), 0.0);
    loads.fill(0.0);
    for s in schedule {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }
    let heavy_bay = loads
        .iter()
        .enumerate()
        .max_by(|&(a, a_load), &(b, b_load)| {
            (a_load * pre.bay_load_scale[a]).total_cmp(&(b_load * pre.bay_load_scale[b]))
        })
        .map(|(bay_id, _)| bay_id);

    badness.resize(problem.blocks.len(), 0);
    badness.fill(0);
    for s in schedule {
        let block = &problem.blocks[s.block_id];
        let tardiness = (s.exit_time - block.due_date).max(0);
        let pref_penalty = pre.pref_penalty[s.block_id][s.bay_id];
        let score = tardiness as f64 * problem.weights.w1
            + pref_penalty as f64 * problem.weights.w3
            + if Some(s.bay_id) == heavy_bay {
                problem.weights.w2
            } else {
                0.0
            };
        badness[s.block_id] = score as i64;
    }
}

fn choose_local_proximity_seeds<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    by_block: &[Option<ScheduledBlock>],
    k: usize,
    rng: &mut R,
    params: &ReconstructNeighborParams,
    scratch: &mut RemoveScratch,
) -> bool {
    let pool_len = (k * params.remove_pool_factor).min(schedule.len()).max(k);
    let slack_weight = rng.gen_range_f64(
        params.remove_fluidity_slack_weight_range.0,
        params.remove_fluidity_slack_weight_range.1,
    );
    let pref_spread_weight = rng.gen_range_f64(
        params.remove_fluidity_pref_spread_weight_range.0,
        params.remove_fluidity_pref_spread_weight_range.1,
    );
    let max_slack = schedule
        .iter()
        .map(|scheduled| {
            let block = &problem.blocks[scheduled.block_id];
            (block.due_date - block.release_time - block.processing_time).max(0) as f64
        })
        .fold(0.0, f64::max)
        .max(1.0);
    let max_pref_spread = schedule
        .iter()
        .map(|scheduled| pre.pref_spread[scheduled.block_id] as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    scratch.fluidity.resize(problem.blocks.len(), 0.0);
    scratch.fluidity.fill(0.0);
    for scheduled in schedule {
        let block = &problem.blocks[scheduled.block_id];
        let slack = (block.due_date - block.release_time - block.processing_time).max(0) as f64;
        let pref_spread = pre.pref_spread[scheduled.block_id] as f64;
        scratch.fluidity[scheduled.block_id] = slack_weight * slack / max_slack
            + pref_spread_weight * (1.0 - pref_spread / max_pref_spread);
    }
    scratch.fluid_pool.clear();
    scratch
        .fluid_pool
        .extend(schedule.iter().map(|s| s.block_id));
    rng.shuffle(&mut scratch.fluid_pool);
    scratch
        .fluid_pool
        .sort_by(|&a, &b| scratch.fluidity[b].total_cmp(&scratch.fluidity[a]));
    scratch.fluid_pool.truncate(pool_len);

    scratch.seed_sizes.clear();
    let mut seed_size_sum = 0;
    while seed_size_sum < k {
        let seed_size = rng
            .gen_range(
                params.remove_seed_per_block.0,
                params.remove_seed_per_block.1 + 1,
            )
            .min(k - seed_size_sum);
        scratch.seed_sizes.push(seed_size);
        seed_size_sum += seed_size;
    }
    let seed_count = scratch.seed_sizes.len();
    let entry_base_interval =
        sample_weighted_index(rng, &params.remove_entry_base_interval_weights) + 1;
    let base_count = (1 + (seed_count - 1) / entry_base_interval).min(seed_count);
    let method_weights = params.remove_seed_method_weights;
    let total_method_weight =
        method_weights.badness + method_weights.fluidity + method_weights.random;
    scratch.seed_methods.clear();
    scratch.seed_methods.extend((0..seed_count).map(|_| {
        let value = rng.next_f64() * total_method_weight;
        if value < method_weights.badness {
            RemoveSeedMethod::Badness
        } else if value < method_weights.badness + method_weights.fluidity {
            RemoveSeedMethod::Fluidity
        } else {
            RemoveSeedMethod::Random
        }
    }));

    scratch.used.resize(problem.blocks.len(), false);
    scratch.used.fill(false);
    scratch.seeds.clear();
    scratch.entry_bases.clear();

    for seed_index in 0..base_count {
        scratch.seed_pool.clear();
        match scratch.seed_methods[seed_index] {
            RemoveSeedMethod::Badness => scratch.seed_pool.extend_from_slice(&scratch.bad_pool),
            RemoveSeedMethod::Fluidity => scratch.seed_pool.extend_from_slice(&scratch.fluid_pool),
            RemoveSeedMethod::Random => scratch
                .seed_pool
                .extend(schedule.iter().map(|s| s.block_id)),
        }
        scratch
            .seed_pool
            .retain(|&block_id| !scratch.used[block_id]);
        rng.shuffle(&mut scratch.seed_pool);
        let Some(&seed_id) = scratch.seed_pool.first() else {
            return false;
        };
        scratch.used[seed_id] = true;
        scratch.seeds.push(RemoveSeed {
            block_id: seed_id,
            remove_count: scratch.seed_sizes[seed_index],
        });
        scratch.entry_bases.push(seed_id);
    }

    for seed_index in base_count..seed_count {
        let base_id = scratch.entry_bases[rng.gen_range(0, scratch.entry_bases.len())];
        let base = by_block[base_id].unwrap();
        scratch.seed_pool.clear();
        match scratch.seed_methods[seed_index] {
            RemoveSeedMethod::Badness => scratch.seed_pool.extend_from_slice(&scratch.bad_pool),
            RemoveSeedMethod::Fluidity => scratch.seed_pool.extend_from_slice(&scratch.fluid_pool),
            RemoveSeedMethod::Random => scratch
                .seed_pool
                .extend(schedule.iter().map(|s| s.block_id)),
        }
        scratch
            .seed_pool
            .retain(|&block_id| !scratch.used[block_id]);
        rng.shuffle(&mut scratch.seed_pool);
        scratch.seed_pool.sort_by_key(|&block_id| {
            let candidate = by_block[block_id].unwrap();
            (
                candidate.entry_time.abs_diff(base.entry_time),
                candidate.bay_id == base.bay_id,
            )
        });
        scratch.seed_pool.truncate(
            params
                .remove_entry_seed_candidate_count
                .min(scratch.seed_pool.len()),
        );
        rng.shuffle(&mut scratch.seed_pool);
        let Some(&seed_id) = scratch.seed_pool.first() else {
            return false;
        };
        scratch.used[seed_id] = true;
        scratch.seeds.push(RemoveSeed {
            block_id: seed_id,
            remove_count: scratch.seed_sizes[seed_index],
        });
    }

    true
}

fn collect_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    by_block: &[Option<ScheduledBlock>],
    k: usize,
    seeds: &[RemoveSeed],
    bad_pool: &[usize],
    remove_x_distance_weight: f64,
    remove_y_distance_weight: f64,
    remove_t_distance_weight: f64,
    rng: &mut R,
    selected: &mut Vec<usize>,
    used: &mut Vec<bool>,
    neighbors: &mut Vec<(f64, usize)>,
    fill_pool: &mut Vec<usize>,
) {
    selected.clear();
    used.resize(problem.blocks.len(), false);
    used.fill(false);
    for seed in seeds {
        push_removed_block(selected, used, seed.block_id, k);
    }

    for seed in seeds {
        let scheduled = by_block[seed.block_id].unwrap();
        let (sx, sy) = scheduled_center(pre, scheduled);
        let st = scheduled.entry_time as f64;
        neighbors.clear();
        neighbors.extend(
            schedule
                .iter()
                .filter(|s| s.bay_id == scheduled.bay_id && !used[s.block_id])
                .map(|&candidate| {
                    let (x, y) = scheduled_center(pre, candidate);
                    let dx = x - sx;
                    let dy = y - sy;
                    let dt = candidate.entry_time as f64 - st;
                    (
                        remove_x_distance_weight * dx * dx
                            + remove_y_distance_weight * dy * dy
                            + remove_t_distance_weight * dt * dt,
                        candidate.block_id,
                    )
                }),
        );
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(_, block_id) in neighbors.iter().take(seed.remove_count - 1) {
            push_removed_block(selected, used, block_id, k);
        }
    }

    fill_pool.clear();
    fill_pool.extend_from_slice(bad_pool);
    rng.shuffle(fill_pool);
    for &block_id in fill_pool.iter() {
        push_removed_block(selected, used, block_id, k);
    }
}

fn choose_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    by_block: &[Option<ScheduledBlock>],
    k: usize,
    rng: &mut R,
    params: &ReconstructNeighborParams,
    loads: &mut Vec<f64>,
    scratch: &mut RemoveScratch,
) -> bool {
    if schedule.is_empty() || k == 0 {
        return false;
    }

    let k = k.min(schedule.len());
    let remove_x_distance_weight = rng.gen_range_f64(
        params.remove_x_distance_weight_range.0,
        params.remove_x_distance_weight_range.1,
    );
    let remove_y_distance_weight = rng.gen_range_f64(
        params.remove_y_distance_weight_range.0,
        params.remove_y_distance_weight_range.1,
    );
    let remove_t_distance_weight = rng.gen_range_f64(
        params.remove_t_distance_weight_range.0,
        params.remove_t_distance_weight_range.1,
    );
    fill_remove_badness(problem, pre, schedule, loads, &mut scratch.badness);

    scratch.bad_pool.clear();
    scratch.bad_pool.extend(schedule.iter().map(|s| s.block_id));
    rng.shuffle(&mut scratch.bad_pool);
    scratch
        .bad_pool
        .sort_by_key(|&block_id| Reverse(scratch.badness[block_id]));
    let pool_len = (k * params.remove_pool_factor).min(schedule.len()).max(k);
    scratch.bad_pool.truncate(pool_len);

    if !choose_local_proximity_seeds(problem, pre, schedule, by_block, k, rng, params, scratch) {
        return false;
    }
    let RemoveScratch {
        seeds,
        bad_pool,
        selected,
        used,
        neighbors,
        fill_pool,
        ..
    } = scratch;
    collect_removed_blocks(
        problem,
        pre,
        schedule,
        by_block,
        k,
        seeds,
        bad_pool,
        remove_x_distance_weight,
        remove_y_distance_weight,
        remove_t_distance_weight,
        rng,
        selected,
        used,
        neighbors,
        fill_pool,
    );
    true
}

fn block_volume(problem: &Problem, block_areas: &[f64], block_id: usize) -> f64 {
    block_areas[block_id] * problem.blocks[block_id].processing_time as f64
}

fn block_limit_time(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    block.due_date - block.processing_time
}

fn block_slack(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    (block.due_date - block.processing_time - block.release_time).max(0)
}

fn build_block_order_context(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &[usize],
) -> BlockOrderContext {
    let max_workload = order
        .iter()
        .map(|&block_id| problem.blocks[block_id].workload as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let max_volume = order
        .iter()
        .map(|&block_id| block_volume(problem, block_areas, block_id))
        .fold(0.0, f64::max)
        .max(1.0);
    let max_pref_spread = order
        .iter()
        .map(|&block_id| pref_spread[block_id] as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let min_limit_time = order
        .iter()
        .map(|&block_id| block_limit_time(problem, block_id))
        .min()
        .unwrap_or(0);
    let max_limit_time = order
        .iter()
        .map(|&block_id| block_limit_time(problem, block_id))
        .max()
        .unwrap_or(min_limit_time);
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
        max_volume,
        max_pref_spread,
        max_limit_time,
        limit_time_span: (max_limit_time - min_limit_time).max(1) as f64,
        max_slack,
        slack_span: (max_slack - min_slack).max(1) as f64,
    }
}

fn block_order_score(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    ctx: &BlockOrderContext,
    weights: BlockOrderWeights,
    current_penalties: Option<&[f64]>,
    current_penalty_weight: f64,
    max_current_penalty: f64,
    block_id: usize,
) -> f64 {
    let block = &problem.blocks[block_id];
    let workload_norm = block.workload as f64 / ctx.max_workload;
    let volume_norm = block_volume(problem, block_areas, block_id) / ctx.max_volume;
    let pref_spread_norm = pref_spread[block_id] as f64 / ctx.max_pref_spread;
    let limit_time_urgency =
        (ctx.max_limit_time - block_limit_time(problem, block_id)) as f64 / ctx.limit_time_span;
    let slack_tightness = (ctx.max_slack - block_slack(problem, block_id)) as f64 / ctx.slack_span;
    let current_penalty = current_penalties
        .map(|penalties| penalties[block_id] / max_current_penalty)
        .unwrap_or(0.0);

    weights.workload * workload_norm
        + weights.volume * volume_norm
        + weights.pref_spread * pref_spread_norm
        + weights.limit_time_urgency * limit_time_urgency
        + weights.slack_tightness * slack_tightness
        + current_penalty_weight * current_penalty
}

fn sort_block_order_with_scratch<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &mut [usize],
    weights: BlockOrderWeights,
    current_penalties: Option<&[f64]>,
    current_penalty_weight: f64,
    rng: &mut R,
    random_scores: &mut Vec<f64>,
) {
    let ctx = build_block_order_context(problem, block_areas, pref_spread, order);
    let max_current_penalty = current_penalties
        .map(|penalties| {
            order
                .iter()
                .map(|&block_id| penalties[block_id])
                .fold(0.0, f64::max)
                .max(1.0)
        })
        .unwrap_or(1.0);
    random_scores.resize(problem.blocks.len(), 0.0);
    random_scores.fill(0.0);
    for &block_id in order.iter() {
        random_scores[block_id] = rng.gen_range_f64(0., weights.random);
    }
    order.sort_by(|&a, &b| {
        let score_a = block_order_score(
            problem,
            block_areas,
            pref_spread,
            &ctx,
            weights,
            current_penalties,
            current_penalty_weight,
            max_current_penalty,
            a,
        ) + random_scores[a];
        let score_b = block_order_score(
            problem,
            block_areas,
            pref_spread,
            &ctx,
            weights,
            current_penalties,
            current_penalty_weight,
            max_current_penalty,
            b,
        ) + random_scores[b];
        score_b
            .total_cmp(&score_a)
            .then(block_limit_time(problem, a).cmp(&block_limit_time(problem, b)))
            .then(
                block_volume(problem, block_areas, b).total_cmp(&block_volume(
                    problem,
                    block_areas,
                    a,
                )),
            )
            .then(problem.blocks[b].workload.cmp(&problem.blocks[a].workload))
            .then(pref_spread[b].cmp(&pref_spread[a]))
            .then(a.cmp(&b))
    });
}

pub(super) fn sort_block_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &mut [usize],
    weights: BlockOrderWeights,
    current_penalties: Option<&[f64]>,
    current_penalty_weight: f64,
    rng: &mut R,
) {
    let mut random_scores = Vec::new();
    sort_block_order_with_scratch(
        problem,
        block_areas,
        pref_spread,
        order,
        weights,
        current_penalties,
        current_penalty_weight,
        rng,
        &mut random_scores,
    );
}

pub(super) fn sort_default_reconstruct_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &mut [usize],
    rng: &mut R,
    params: &ReconstructNeighborParams,
) {
    let weights = sample_reconstruct_order_weights(rng, params);
    sort_block_order(
        problem,
        block_areas,
        pref_spread,
        order,
        weights,
        None,
        0.0,
        rng,
    );
}

pub(super) fn build_heuristic_precedence(
    problem: &Problem,
    state: &PreoptimizeState,
    precedence_margin: i64,
) -> HeuristicPrecedence {
    let block_count = problem.blocks.len();
    let start_times: Vec<_> = state.blocks.iter().map(|block| block.entry_time).collect();
    let end_times: Vec<_> = state
        .blocks
        .iter()
        .enumerate()
        .map(|(block_id, block)| block.entry_time + problem.blocks[block_id].processing_time)
        .collect();
    let mut predecessors = vec![Vec::new(); block_count];
    let mut successors = vec![Vec::new(); block_count];

    for after in 0..block_count {
        let latest_start = (0..block_count)
            .filter(|&block_id| {
                block_id != after
                    && state.blocks[block_id].bay_id == state.blocks[after].bay_id
                    && end_times[block_id].saturating_add(precedence_margin) <= start_times[after]
            })
            .map(|block_id| start_times[block_id])
            .max();

        for before in 0..block_count {
            if before == after
                || state.blocks[before].bay_id != state.blocks[after].bay_id
                || end_times[before].saturating_add(precedence_margin) > start_times[after]
            {
                continue;
            }
            if latest_start.is_some_and(|start_time| {
                end_times[before].saturating_add(precedence_margin) <= start_time
            }) {
                continue;
            }
            successors[before].push(after);
            predecessors[after].push(before);
        }
    }

    HeuristicPrecedence {
        predecessors,
        successors,
    }
}

fn fill_topological_order<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    bay_block_ids: &[usize],
    constraints: &HeuristicPrecedence,
    weights: BlockOrderWeights,
    current_penalties: Option<&[f64]>,
    current_penalty_weight: f64,
    rng: &mut R,
    scratch: &mut OrderScratch,
) {
    scratch.priority_order.clear();
    scratch.priority_order.extend_from_slice(bay_block_ids);
    sort_block_order_with_scratch(
        problem,
        &pre.max_footprint_area,
        &pre.pref_spread,
        &mut scratch.priority_order,
        weights,
        current_penalties,
        current_penalty_weight,
        rng,
        &mut scratch.random_scores,
    );
    scratch
        .priority_rank
        .resize(problem.blocks.len(), usize::MAX);
    scratch.priority_rank.fill(usize::MAX);
    for (rank, &block_id) in scratch.priority_order.iter().enumerate() {
        scratch.priority_rank[block_id] = rank;
    }

    scratch.included.resize(problem.blocks.len(), false);
    scratch.included.fill(false);
    for &block_id in bay_block_ids {
        scratch.included[block_id] = true;
    }
    scratch.indegree.resize(problem.blocks.len(), 0);
    scratch.indegree.fill(0);
    scratch.ready.clear();
    for &block_id in bay_block_ids {
        scratch.indegree[block_id] = constraints.predecessors[block_id]
            .iter()
            .filter(|&&before| scratch.included[before])
            .count();
        if scratch.indegree[block_id] == 0 {
            scratch
                .ready
                .push(Reverse((scratch.priority_rank[block_id], block_id)));
        }
    }

    scratch.order.clear();
    while let Some(Reverse((_, block_id))) = scratch.ready.pop() {
        scratch.order.push(block_id);
        for &after in &constraints.successors[block_id] {
            if !scratch.included[after] {
                continue;
            }
            scratch.indegree[after] -= 1;
            if scratch.indegree[after] == 0 {
                scratch
                    .ready
                    .push(Reverse((scratch.priority_rank[after], after)));
            }
        }
    }
    debug_assert_eq!(scratch.order.len(), bay_block_ids.len());
}

fn build_bay_schedule<'a, R: Random>(
    problem: &Problem,
    pre: &'a Precompute,
    bay_id: usize,
    order: &[usize],
    constraints: &HeuristicPrecedence,
    params: &InsertParams,
    rng: &mut R,
    schedule: &mut Vec<ScheduledBlock>,
    scheduled_by_id: &mut Vec<Option<ScheduledBlock>>,
    loads: &mut Vec<f64>,
    scratch: &mut InsertScratch<'a>,
) -> bool {
    schedule.clear();
    scheduled_by_id.resize(problem.blocks.len(), None);
    scheduled_by_id.fill(None);
    loads.resize(problem.bays.len(), 0.0);
    loads.fill(0.0);
    let bay_order = [bay_id];

    for &block_id in order {
        let block = &problem.blocks[block_id];
        let min_entry_time = constraints.predecessors[block_id]
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
        let Some(scheduled) = insert_greedy(
            problem,
            pre,
            original,
            min_entry_time,
            i64::MAX,
            ScheduleView::Flat(schedule),
            &loads,
            params,
            &bay_order,
            1,
            1.0,
            rng,
            scratch,
        ) else {
            return false;
        };
        loads[bay_id] += block.workload as f64;
        scheduled_by_id[block_id] = Some(scheduled);
        schedule.push(scheduled);
    }
    true
}
