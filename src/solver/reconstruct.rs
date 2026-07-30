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
    insert::insert_greedy,
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

#[derive(Clone, Copy)]
pub(super) struct BlockOrderWeights {
    workload: f64,
    volume: f64,
    pref_spread: f64,
    pref_density: f64,
    limit_time_urgency: f64,
    random: f64,
}

struct BlockOrderContext {
    max_workload: f64,
    max_volume: f64,
    max_pref_spread: f64,
    max_pref_density: f64,
    max_limit_time: i64,
    limit_time_span: f64,
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
        pref_density: rng.gen_range_f64(
            params.pref_density_weight_range.0,
            params.pref_density_weight_range.1,
        ),
        limit_time_urgency: rng.gen_range_f64(
            params.limit_time_urgency_weight_range.0,
            params.limit_time_urgency_weight_range.1,
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

            while timer.elapsed_seconds() < build_deadline {
                let bay_id = active_bays[turn % active_bays.len()];
                turn += 1;

                let weights = sample_reconstruct_order_weights(&mut rng, neighbor_params);
                let order = build_topological_order(
                    problem,
                    pre,
                    &blocks_by_bay[bay_id],
                    &constraints,
                    weights,
                    &mut rng,
                );
                let order_hash = hash_order(&order);
                if !seen_order_hashes[bay_id].lock().unwrap().insert(order_hash) {
                    continue;
                }

                let Some(schedule) = build_bay_schedule(
                    problem,
                    pre,
                    bay_id,
                    &order,
                    &constraints,
                    insert_params,
                    &mut rng,
                ) else {
                    continue;
                };
                trials += 1;

                let tardiness = schedule_tardiness(problem, &schedule);
                let mut best = bests[bay_id].lock().unwrap();
                if best
                    .as_ref()
                    .is_none_or(|(best_tardiness, _)| tardiness < *best_tardiness)
                {
                    *best = Some((tardiness, schedule));
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

pub(super) struct ReconstructBase {
    pub(super) schedule: Vec<ScheduledBlock>,
    pub(super) loads: Vec<f64>,
    pub(super) weighted_z1_z3: f64,
}

pub(super) fn build_reconstruct_base(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    removed_ids: &[usize],
) -> ReconstructBase {
    let mut removed = vec![false; problem.blocks.len()];
    for &block_id in removed_ids {
        removed[block_id] = true;
    }

    let mut remaining = Vec::with_capacity(schedule.len());
    let mut loads = vec![0.0; problem.bays.len()];
    let mut weighted_z1_z3 = 0.0;
    for &scheduled in schedule {
        if removed[scheduled.block_id] {
            continue;
        }
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        weighted_z1_z3 += score13_block(problem, pre, scheduled);
        remaining.push(scheduled);
    }

    ReconstructBase {
        schedule: remaining,
        loads,
        weighted_z1_z3,
    }
}

pub(super) fn try_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    constraints: Option<&HeuristicPrecedence>,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    params: &ReconstructNeighborParams,
    insert_params: &InsertParams,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng, params).min(problem.blocks.len());

    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng, params)?;
    if removed_ids.is_empty() {
        return None;
    }
    let weights = sample_reconstruct_order_weights(rng, params);
    let order = if let Some(constraints) = constraints {
        build_topological_order(problem, pre, &removed_ids, constraints, weights, rng)
    } else {
        sort_block_order(
            problem,
            &pre.max_footprint_area,
            &pre.pref_spread,
            &mut removed_ids,
            weights,
            rng,
        );
        removed_ids.clone()
    };

    let original_by_id = scheduled_by_id(problem, schedule);
    let ReconstructBase {
        schedule: mut cur,
        mut loads,
        weighted_z1_z3: mut fixed_score13,
    } = build_reconstruct_base(problem, pre, schedule, &removed_ids);
    let mut current_by_id = constraints.map(|_| scheduled_by_id(problem, &cur));

    for block_id in order {
        if fixed_score13 > accept_threshold + 1e-9 {
            return None;
        }
        let old = original_by_id[block_id]?;
        let EntryTimeBounds {
            min: min_entry_time,
            max: max_entry_time,
        } = if let Some(constraints) = constraints {
            precedence_entry_time_bounds(
                problem,
                constraints,
                current_by_id.as_ref().unwrap(),
                block_id,
            )?
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
            &cur,
            &loads,
            insert_params,
            &pre.bay_order_by_pref[old.block_id],
            params.insert_candidate_top_k,
            rng,
        )?;
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        fixed_score13 += score13_block(problem, pre, scheduled);
        if let Some(current_by_id) = &mut current_by_id {
            current_by_id[block_id] = Some(scheduled);
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

fn occupancy_interval_distance(a: ScheduledBlock, b: ScheduledBlock) -> i64 {
    if a.entry_time < b.exit_time && b.entry_time < a.exit_time {
        0
    } else if a.exit_time <= b.entry_time {
        b.entry_time - a.exit_time + 1
    } else {
        a.entry_time - b.exit_time + 1
    }
}

fn push_removed_block(selected: &mut Vec<usize>, used: &mut [bool], block_id: usize, limit: usize) {
    if selected.len() < limit && !used[block_id] {
        selected.push(block_id);
        used[block_id] = true;
    }
}

fn remove_badness(problem: &Problem, pre: &Precompute, schedule: &[ScheduledBlock]) -> Vec<i64> {
    let mut loads = vec![0.0; problem.bays.len()];
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
                0.0
            };
        badness[s.block_id] = score as i64;
    }
    badness
}

fn choose_local_proximity_seeds<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    by_block: &[Option<ScheduledBlock>],
    k: usize,
    bad_pool: &[usize],
    rng: &mut R,
    params: &ReconstructNeighborParams,
) -> Option<Vec<RemoveSeed>> {
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
    let mut fluidity = vec![0.0; problem.blocks.len()];
    for scheduled in schedule {
        let block = &problem.blocks[scheduled.block_id];
        let slack = (block.due_date - block.release_time - block.processing_time).max(0) as f64;
        let pref_spread = pre.pref_spread[scheduled.block_id] as f64;
        fluidity[scheduled.block_id] = slack_weight * slack / max_slack
            + pref_spread_weight * (1.0 - pref_spread / max_pref_spread);
    }
    let mut fluid_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut fluid_pool);
    fluid_pool.sort_by(|&a, &b| fluidity[b].total_cmp(&fluidity[a]));
    fluid_pool.truncate(pool_len);

    let mut seed_sizes = Vec::new();
    let mut seed_size_sum = 0;
    while seed_size_sum < k {
        let seed_size = rng
            .gen_range(
                params.remove_seed_per_block.0,
                params.remove_seed_per_block.1 + 1,
            )
            .min(k - seed_size_sum);
        seed_sizes.push(seed_size);
        seed_size_sum += seed_size;
    }
    let seed_count = seed_sizes.len();
    let entry_base_interval =
        sample_weighted_index(rng, &params.remove_entry_base_interval_weights) + 1;
    let base_count = (1 + (seed_count - 1) / entry_base_interval).min(seed_count);
    let method_weights = params.remove_seed_method_weights;
    let total_method_weight =
        method_weights.badness + method_weights.fluidity + method_weights.random;
    let seed_methods: Vec<_> = (0..seed_count)
        .map(|_| {
            let value = rng.next_f64() * total_method_weight;
            if value < method_weights.badness {
                RemoveSeedMethod::Badness
            } else if value < method_weights.badness + method_weights.fluidity {
                RemoveSeedMethod::Fluidity
            } else {
                RemoveSeedMethod::Random
            }
        })
        .collect();

    let mut used = vec![false; problem.blocks.len()];
    let mut seeds = Vec::with_capacity(seed_count);
    let mut entry_bases = Vec::with_capacity(base_count);

    for seed_index in 0..base_count {
        let mut seed_pool: Vec<usize> = match seed_methods[seed_index] {
            RemoveSeedMethod::Badness => bad_pool.to_vec(),
            RemoveSeedMethod::Fluidity => fluid_pool.clone(),
            RemoveSeedMethod::Random => schedule.iter().map(|s| s.block_id).collect(),
        };
        seed_pool.retain(|&block_id| !used[block_id]);
        rng.shuffle(&mut seed_pool);
        let seed_id = *seed_pool.get(0)?;
        used[seed_id] = true;
        seeds.push(RemoveSeed {
            block_id: seed_id,
            remove_count: seed_sizes[seed_index],
        });
        entry_bases.push(seed_id);
    }

    for seed_index in base_count..seed_count {
        let base_id = entry_bases[rng.gen_range(0, entry_bases.len())];
        let base = by_block[base_id].unwrap();
        let mut seed_pool: Vec<usize> = match seed_methods[seed_index] {
            RemoveSeedMethod::Badness => bad_pool.to_vec(),
            RemoveSeedMethod::Fluidity => fluid_pool.clone(),
            RemoveSeedMethod::Random => schedule.iter().map(|s| s.block_id).collect(),
        };
        seed_pool.retain(|&block_id| !used[block_id]);
        rng.shuffle(&mut seed_pool);
        seed_pool.sort_by_key(|&block_id| {
            let candidate = by_block[block_id].unwrap();
            (
                occupancy_interval_distance(base, candidate),
                candidate.bay_id == base.bay_id,
            )
        });
        seed_pool.truncate(
            params
                .remove_entry_seed_candidate_count
                .min(seed_pool.len()),
        );
        rng.shuffle(&mut seed_pool);
        let seed_id = *seed_pool.get(0)?;
        used[seed_id] = true;
        seeds.push(RemoveSeed {
            block_id: seed_id,
            remove_count: seed_sizes[seed_index],
        });
    }

    Some(seeds)
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
) -> Vec<usize> {
    let mut selected = Vec::with_capacity(k);
    let mut used = vec![false; problem.blocks.len()];
    for seed in seeds {
        push_removed_block(&mut selected, &mut used, seed.block_id, k);
    }

    for seed in seeds {
        let scheduled = by_block[seed.block_id].unwrap();
        let (sx, sy) = scheduled_center(pre, scheduled);
        let mut neighbors: Vec<(f64, usize)> = schedule
            .iter()
            .filter(|s| s.bay_id == scheduled.bay_id && !used[s.block_id])
            .map(|&candidate| {
                let (x, y) = scheduled_center(pre, candidate);
                let dx = x - sx;
                let dy = y - sy;
                let dt = occupancy_interval_distance(scheduled, candidate) as f64;
                (
                    remove_x_distance_weight * dx * dx
                        + remove_y_distance_weight * dy * dy
                        + remove_t_distance_weight * dt * dt,
                    candidate.block_id,
                )
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, block_id) in neighbors.into_iter().take(seed.remove_count - 1) {
            push_removed_block(&mut selected, &mut used, block_id, k);
        }
    }

    let mut fill_pool = bad_pool.to_vec();
    rng.shuffle(&mut fill_pool);
    for block_id in fill_pool {
        push_removed_block(&mut selected, &mut used, block_id, k);
    }
    selected
}

pub(super) fn choose_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    k: usize,
    rng: &mut R,
    params: &ReconstructNeighborParams,
) -> Option<Vec<usize>> {
    if schedule.is_empty() || k == 0 {
        return None;
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
    let badness = remove_badness(problem, pre, schedule);
    let mut by_block = vec![None; problem.blocks.len()];
    for &scheduled in schedule {
        by_block[scheduled.block_id] = Some(scheduled);
    }

    let mut bad_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut bad_pool);
    bad_pool.sort_by_key(|&block_id| Reverse(badness[block_id]));
    let pool_len = (k * params.remove_pool_factor).min(schedule.len()).max(k);
    bad_pool.truncate(pool_len);

    let seeds =
        choose_local_proximity_seeds(problem, pre, schedule, &by_block, k, &bad_pool, rng, params)?;
    let blocks = collect_removed_blocks(
        problem,
        pre,
        schedule,
        &by_block,
        k,
        &seeds,
        &bad_pool,
        remove_x_distance_weight,
        remove_y_distance_weight,
        remove_t_distance_weight,
        rng,
    );
    Some(blocks)
}

fn block_volume(problem: &Problem, block_areas: &[f64], block_id: usize) -> f64 {
    block_areas[block_id] * problem.blocks[block_id].processing_time as f64
}

fn block_pref_density(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    block_id: usize,
) -> f64 {
    pref_spread[block_id] as f64 / block_volume(problem, block_areas, block_id).max(1.0)
}

fn block_limit_time(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    block.due_date - block.processing_time
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
    let max_pref_density = order
        .iter()
        .map(|&block_id| block_pref_density(problem, block_areas, pref_spread, block_id))
        .fold(0.0, f64::max)
        .max(1e-9);
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

    BlockOrderContext {
        max_workload,
        max_volume,
        max_pref_spread,
        max_pref_density,
        max_limit_time,
        limit_time_span: (max_limit_time - min_limit_time).max(1) as f64,
    }
}

fn block_order_score(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    ctx: &BlockOrderContext,
    weights: BlockOrderWeights,
    block_id: usize,
) -> f64 {
    let block = &problem.blocks[block_id];
    let workload_norm = block.workload as f64 / ctx.max_workload;
    let volume_norm = block_volume(problem, block_areas, block_id) / ctx.max_volume;
    let pref_spread_norm = pref_spread[block_id] as f64 / ctx.max_pref_spread;
    let pref_density_norm =
        block_pref_density(problem, block_areas, pref_spread, block_id) / ctx.max_pref_density;
    let limit_time_urgency =
        (ctx.max_limit_time - block_limit_time(problem, block_id)) as f64 / ctx.limit_time_span;

    weights.workload * workload_norm
        + weights.volume * volume_norm
        + weights.pref_spread * pref_spread_norm
        + weights.pref_density * pref_density_norm
        + weights.limit_time_urgency * limit_time_urgency
}

pub(super) fn sort_block_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &mut [usize],
    weights: BlockOrderWeights,
    rng: &mut R,
) {
    let ctx = build_block_order_context(problem, block_areas, pref_spread, order);
    let mut random_scores = vec![0.0; problem.blocks.len()];
    for &block_id in order.iter() {
        random_scores[block_id] = rng.gen_range_f64(0., weights.random);
    }
    order.sort_by(|&a, &b| {
        let score_a = block_order_score(problem, block_areas, pref_spread, &ctx, weights, a)
            + random_scores[a];
        let score_b = block_order_score(problem, block_areas, pref_spread, &ctx, weights, b)
            + random_scores[b];
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

pub(super) fn sort_default_reconstruct_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &mut [usize],
    rng: &mut R,
    params: &ReconstructNeighborParams,
) {
    let weights = sample_reconstruct_order_weights(rng, params);
    sort_block_order(problem, block_areas, pref_spread, order, weights, rng);
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

pub(super) fn build_topological_order<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    bay_block_ids: &[usize],
    constraints: &HeuristicPrecedence,
    weights: BlockOrderWeights,
    rng: &mut R,
) -> Vec<usize> {
    let mut priority_order = bay_block_ids.to_vec();
    sort_block_order(
        problem,
        &pre.max_footprint_area,
        &pre.pref_spread,
        &mut priority_order,
        weights,
        rng,
    );
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
        indegree[block_id] = constraints.predecessors[block_id]
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
        for &after in &constraints.successors[block_id] {
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

fn build_bay_schedule<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    bay_id: usize,
    order: &[usize],
    constraints: &HeuristicPrecedence,
    params: &InsertParams,
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    let mut schedule = Vec::with_capacity(order.len());
    let mut scheduled_by_id: Vec<Option<ScheduledBlock>> = vec![None; problem.blocks.len()];
    let mut loads = vec![0.0; problem.bays.len()];
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
        let scheduled = insert_greedy(
            problem,
            pre,
            original,
            min_entry_time,
            i64::MAX,
            &schedule,
            &loads,
            params,
            &bay_order,
            1,
            rng,
        )?;
        loads[bay_id] += block.workload as f64;
        scheduled_by_id[block_id] = Some(scheduled);
        schedule.push(scheduled);
    }
    Some(schedule)
}
