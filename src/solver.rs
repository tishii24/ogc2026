use crate::{
    insert::{insert_greedy, try_place_block},
    log,
    params::{InsertParams, NeighborParams, SolverParams},
    precompute::Precompute,
    preoptimize::{PreoptimizePrecompute, preoptimize},
    solver_util::{
        NeighborKind, block_pref_spread, gen_rangef, schedule_tardiness, schedule_to_solution,
        score_schedule, score13_block,
    },
    util::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
    *,
};
use rayon::prelude::*;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    sync::Mutex,
};

pub mod optimize;

pub use optimize::GlobalAnnealing;

const NEIGHBOR_KINDS: &[&str] = &["Large", "Shift", "Move", "Rotate", "Swap"];

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

#[derive(Clone, Debug)]
pub struct OptimizeState {
    pub score: f64,
    pub blocks: Vec<ScheduledBlock>,
}

#[derive(Clone, Copy)]
struct BlockOrderWeights {
    workload: f64,
    volume: f64,
    pref_spread: f64,
    limit_time_urgency: f64,
    random: f64,
}

struct BlockOrderContext {
    max_workload: f64,
    max_volume: f64,
    max_pref_spread: f64,
    max_limit_time: i64,
    limit_time_span: f64,
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

fn sample_reconstruct_order_weights(
    rng: &mut impl Random,
    params: &NeighborParams,
) -> BlockOrderWeights {
    BlockOrderWeights {
        workload: gen_rangef(rng, params.reconstruct_workload_weight_range),
        volume: gen_rangef(rng, params.reconstruct_volume_weight_range),
        pref_spread: gen_rangef(rng, params.reconstruct_pref_spread_weight_range),
        limit_time_urgency: gen_rangef(rng, params.reconstruct_limit_time_urgency_weight_range),
        random: gen_rangef(rng, params.reconstruct_order_random_weight_range),
    }
}

fn phase_time_limit(timelimit: f64, ratio: f64, max_seconds: f64) -> f64 {
    (timelimit * ratio).min(max_seconds).max(1e-4)
}

pub fn solve(
    problem: &Problem,
    timelimit: f64,
    timer: Timer,
    params: &SolverParams,
) -> Result<Solution, String> {
    let deadline = timelimit - params.runtime.local_search_time_buffer_seconds;

    log!("[{:.4}] building precompute...", timer.elapsed_seconds());
    let pre = Precompute::build(problem, &params.precompute);
    log!("[{:.4}] precompute built", timer.elapsed_seconds());

    log!(
        "[{:.4}] building preoptimize precompute...",
        timer.elapsed_seconds()
    );
    let preoptimize_pre = PreoptimizePrecompute::build(problem)?;
    log!(
        "[{:.4}] preoptimize precompute built",
        timer.elapsed_seconds()
    );
    let preoptimize_time_limit = phase_time_limit(
        timelimit,
        params.phases.initial_preoptimize.time_ratio,
        params.phases.initial_preoptimize.max_seconds,
    )
    .min((deadline - timer.elapsed_seconds()).max(1e-4));
    let initial_abstract = preoptimize(
        problem,
        &preoptimize_pre,
        &params.preoptimize,
        &params.global_neighbor,
        preoptimize_time_limit,
        params.runtime.worker_count,
        params.runtime.preoptimize_seed,
    )?;
    log!(
        "[{:.4}] initial abstract score: {:.3}",
        timer.elapsed_seconds(),
        initial_abstract.score
    );
    let build_time_limit = phase_time_limit(
        timelimit,
        params.phases.initial_build.time_ratio,
        params.phases.initial_build.max_seconds,
    )
    .min((deadline - timer.elapsed_seconds()).max(1e-4));
    let initial = build_optimize_state(
        problem,
        &pre,
        &initial_abstract,
        build_time_limit,
        timer,
        params.runtime.worker_count,
        params.runtime.solver_seed,
        &params.global_neighbor,
        &params.insert,
        params.preoptimize.precedence_margin,
    )
    .ok_or_else(|| "failed to build initial optimize state".to_string())?;
    log!(
        "[{:.4}] initial optimize score: {:.3}",
        timer.elapsed_seconds(),
        initial.score
    );

    let global = GlobalAnnealing::new(
        problem,
        &pre,
        &initial_abstract,
        params.preoptimize.precedence_margin,
        timer,
    );
    let global_start = timer.elapsed_seconds();
    let constrained_time_limit = phase_time_limit(
        (deadline - global_start).max(0.0),
        params.phases.global_constrained.time_ratio,
        params.phases.global_constrained.max_seconds,
    );
    let constrained_deadline = global_start + constrained_time_limit;
    let initial = global.run(
        initial,
        constrained_deadline,
        &params.global_optimize,
        &params.global_neighbor,
        &params.insert,
        true,
        params.runtime.solver_seed,
        params.runtime.worker_count,
    );
    log!(
        "[{:.4}] constrained global annealing score: {:.3}",
        timer.elapsed_seconds(),
        initial.score
    );

    let best = global.run(
        initial,
        deadline,
        &params.global_optimize,
        &params.global_neighbor,
        &params.insert,
        false,
        params.runtime.solver_seed.wrapping_add(1 << 32),
        params.runtime.worker_count,
    );
    Ok(schedule_to_solution(&best.blocks))
}

pub fn build_optimize_state(
    problem: &Problem,
    pre: &Precompute,
    state: &PreoptimizeState,
    time_limit: f64,
    timer: Timer,
    max_worker_count: usize,
    seed: u64,
    neighbor_params: &NeighborParams,
    insert_params: &InsertParams,
    precedence_margin: i64,
) -> Option<OptimizeState> {
    let constraints = build_precedence_constraints(problem, state, precedence_margin);

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
            score: 0.0,
            blocks: Vec::new(),
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

    let score = score_schedule(problem, pre, &blocks);
    log!(
        "[{:.4}] [build] finished: score={:.3}",
        timer.elapsed_seconds(),
        score,
    );
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
    constraints: Option<&PrecedenceConstraints>,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    params: &NeighborParams,
    insert_params: &InsertParams,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng, params).min(problem.blocks.len());

    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng, params);
    if removed_ids.is_empty() {
        return None;
    }
    let weights = sample_reconstruct_order_weights(rng, params);
    let order = if let Some(constraints) = constraints {
        build_topological_order(problem, pre, &removed_ids, constraints, weights, rng)
    } else {
        sort_block_order(problem, &pre.block_area, &mut removed_ids, weights, rng);
        removed_ids.clone()
    };

    let original_by_id = scheduled_by_id(problem, schedule);
    let mut removed = vec![false; problem.blocks.len()];
    for &block_id in &removed_ids {
        removed[block_id] = true;
    }
    let mut cur = Vec::with_capacity(schedule.len());
    let mut loads = vec![0.0; problem.bays.len()];
    let mut fixed_score13 = 0.0;
    for &scheduled in schedule {
        if removed[scheduled.block_id] {
            continue;
        }
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        fixed_score13 += score13_block(problem, pre, scheduled);
        cur.push(scheduled);
    }
    let mut current_by_id = constraints.map(|_| scheduled_by_id(problem, &cur));

    for block_id in order {
        if fixed_score13 > accept_threshold + 1e-9 {
            return None;
        }
        let old = original_by_id[block_id]?;
        let (min_entry_time, max_entry_time) = if let Some(constraints) = constraints {
            precedence_entry_time_range(
                problem,
                constraints,
                current_by_id.as_ref().unwrap(),
                block_id,
            )?
        } else {
            (i64::MIN, i64::MAX)
        };
        let k = schedule.len() - cur.len();
        let y_sample_ratio = insert_params
            .y_sample_ratio_base
            .powi((k - 1) as i32)
            .clamp(insert_params.y_sample_ratio_min, 1.0);
        let scheduled = insert_greedy(
            problem,
            pre,
            old,
            min_entry_time,
            max_entry_time,
            &cur,
            &loads,
            insert_params,
            y_sample_ratio,
            &pre.bay_order_by_pref[old.block_id],
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

    for dist in (0..=params.shift_max_x + params.shift_max_y).rev() {
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
    min_entry_time: i64,
    max_entry_time: i64,
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
                min_entry_time,
                max_entry_time,
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
    constraints: Option<&PrecedenceConstraints>,
    same_bay_only: bool,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.len() < 2 {
        return None;
    }

    let a_idx = rng.gen_index(schedule.len());
    let a_old = schedule[a_idx];
    let candidates: Vec<_> = pre.other_block_neighbors[a_old.block_id][a_old.orient_idx]
        .iter()
        .filter_map(|&candidate| {
            let b_idx = schedule
                .iter()
                .position(|s| s.block_id == candidate.block_id)?;
            (!same_bay_only || schedule[b_idx].bay_id == a_old.bay_id).then_some((candidate, b_idx))
        })
        .take(params.swap_neighbor_top_k)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let (candidate, b_idx) = candidates[rng.gen_index(candidates.len())];
    let b_old = schedule[b_idx];

    let mut cur = Vec::with_capacity(schedule.len());
    for (idx, &s) in schedule.iter().enumerate() {
        if idx != a_idx && idx != b_idx {
            cur.push(s);
        }
    }

    let mut targets = [
        (
            a_old,
            b_old.bay_id,
            a_old.orient_idx,
            b_old.x - candidate.dx,
            b_old.y - candidate.dy,
        ),
        (
            b_old,
            a_old.bay_id,
            candidate.orient_idx,
            a_old.x + candidate.dx,
            a_old.y + candidate.dy,
        ),
    ];
    if constraints
        .is_some_and(|constraints| constraints.befores[a_old.block_id].contains(&b_old.block_id))
    {
        targets.swap(0, 1);
    }

    for (old, bay_id, orient_idx, base_x, base_y) in targets {
        let (min_entry_time, max_entry_time) = if let Some(constraints) = constraints {
            let by_id = scheduled_by_id(problem, &cur);
            precedence_entry_time_range(problem, constraints, &by_id, old.block_id)?
        } else {
            (i64::MIN, i64::MAX)
        };
        let new = try_swap_place(
            problem,
            pre,
            old,
            &cur,
            bay_id,
            orient_idx,
            base_x,
            base_y,
            min_entry_time,
            max_entry_time,
            params,
        )?;
        cur.push(new);
    }

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
    insert_params: &InsertParams,
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
        insert_params,
        1.0,
        bay_order,
        rng,
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

fn block_volume(problem: &Problem, block_areas: &[f64], block_id: usize) -> f64 {
    block_areas[block_id] * problem.blocks[block_id].processing_time as f64
}

fn block_limit_time(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    block.due_date - block.processing_time
}

fn build_block_order_context(
    problem: &Problem,
    block_areas: &[f64],
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
        .map(|&block_id| block_pref_spread(problem, block_id) as f64)
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

    BlockOrderContext {
        max_workload,
        max_volume,
        max_pref_spread,
        max_limit_time,
        limit_time_span: (max_limit_time - min_limit_time).max(1) as f64,
    }
}

fn block_order_score(
    problem: &Problem,
    block_areas: &[f64],
    ctx: &BlockOrderContext,
    weights: BlockOrderWeights,
    block_id: usize,
) -> f64 {
    let block = &problem.blocks[block_id];
    let workload_norm = block.workload as f64 / ctx.max_workload;
    let volume_norm = block_volume(problem, block_areas, block_id) / ctx.max_volume;
    let pref_spread_norm = block_pref_spread(problem, block_id) as f64 / ctx.max_pref_spread;
    let limit_time_urgency =
        (ctx.max_limit_time - block_limit_time(problem, block_id)) as f64 / ctx.limit_time_span;

    weights.workload * workload_norm
        + weights.volume * volume_norm
        + weights.pref_spread * pref_spread_norm
        + weights.limit_time_urgency * limit_time_urgency
}

fn sort_block_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    order: &mut [usize],
    weights: BlockOrderWeights,
    rng: &mut R,
) {
    let ctx = build_block_order_context(problem, block_areas, order);
    let mut random_scores = vec![0.0; problem.blocks.len()];
    for &block_id in order.iter() {
        random_scores[block_id] = rng.gen_rangef(0., weights.random);
    }
    order.sort_by(|&a, &b| {
        let score_a = block_order_score(problem, block_areas, &ctx, weights, a) + random_scores[a];
        let score_b = block_order_score(problem, block_areas, &ctx, weights, b) + random_scores[b];
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
            .then(block_pref_spread(problem, b).cmp(&block_pref_spread(problem, a)))
            .then(a.cmp(&b))
    });
}

pub fn sort_default_reconstruct_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    order: &mut [usize],
    rng: &mut R,
    params: &NeighborParams,
) {
    let weights = sample_reconstruct_order_weights(rng, params);
    sort_block_order(problem, block_areas, order, weights, rng);
}

fn build_precedence_constraints(
    problem: &Problem,
    state: &PreoptimizeState,
    precedence_margin: i64,
) -> PrecedenceConstraints {
    let block_count = problem.blocks.len();
    let start_times: Vec<_> = state.blocks.iter().map(|block| block.entry_time).collect();
    let end_times: Vec<_> = state
        .blocks
        .iter()
        .enumerate()
        .map(|(block_id, block)| block.entry_time + problem.blocks[block_id].processing_time)
        .collect();
    let mut befores = vec![Vec::new(); block_count];
    let mut afters = vec![Vec::new(); block_count];

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
            afters[before].push(after);
            befores[after].push(before);
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
    sort_block_order(problem, &pre.block_area, &mut priority_order, weights, rng);
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

fn build_bay_schedule<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    bay_id: usize,
    order: &[usize],
    constraints: &PrecedenceConstraints,
    params: &InsertParams,
    rng: &mut R,
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
            params,
            1.0,
            &bay_order,
            rng,
        )?;
        loads[bay_id] += block.workload as f64;
        scheduled_by_id[block_id] = Some(scheduled);
        schedule.push(scheduled);
    }
    Some(schedule)
}
