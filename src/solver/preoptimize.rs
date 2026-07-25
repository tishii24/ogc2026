use std::sync::Mutex;

use crate::{
    Bay, Orientation, Problem, log,
    solver::{PreoptimizeState, PreoptimizedBlock, sort_default_reconstruct_order},
    utils::annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    utils::base::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
    utils::params::{AnnealingParamsConfig, NeighborParams, PreoptimizeSolverParams},
    utils::precompute::{build_pref_spread, orientation_union},
};
use geo::Area;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
struct OrientationArea {
    union_area: f64,
    bbox_area: f64,
    perimeter: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

#[derive(Debug)]
pub struct PreoptimizePrecompute {
    orientation_areas: Vec<Vec<OrientationArea>>,
    bay_load_scale: Vec<f64>,
    pref_penalty: Vec<Vec<i64>>,
    pref_spread: Vec<i64>,
    min_time: i64,
    search_horizon: i64,
}

impl PreoptimizePrecompute {
    pub fn build(problem: &Problem) -> Result<Self, String> {
        if problem.bays.is_empty() {
            return Err("bays must be non-empty".to_string());
        }
        for (block_id, block) in problem.blocks.iter().enumerate() {
            if block.processing_time <= 0 {
                return Err(format!("block {block_id} has non-positive processing_time"));
            }
            if block.bay_preferences.len() != problem.bays.len() {
                return Err(format!(
                    "block {block_id} has {} preferences, but there are {} bays",
                    block.bay_preferences.len(),
                    problem.bays.len()
                ));
            }
        }

        let orientation_areas = problem
            .blocks
            .iter()
            .enumerate()
            .map(|(block_id, block)| {
                block
                    .shape
                    .iter()
                    .map(orientation_area)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|err| format!("block {block_id}: {err}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bay_areas: Vec<f64> = problem
            .bays
            .iter()
            .map(|bay| (bay.width * bay.height) as f64)
            .collect();
        let avg_bay_area = bay_areas.iter().sum::<f64>() / bay_areas.len() as f64;
        let bay_load_scale = bay_areas.iter().map(|&area| avg_bay_area / area).collect();
        let pref_penalty = problem
            .blocks
            .iter()
            .map(|block| {
                let max_pref = block.bay_preferences.iter().copied().max().unwrap_or(0);
                block
                    .bay_preferences
                    .iter()
                    .map(|&pref| max_pref - pref)
                    .collect()
            })
            .collect();
        let pref_spread = build_pref_spread(problem);

        if problem.blocks.is_empty() {
            return Ok(Self {
                orientation_areas,
                bay_load_scale,
                pref_penalty,
                pref_spread,
                min_time: 0,
                search_horizon: 0,
            });
        }

        let min_time = problem
            .blocks
            .iter()
            .map(|block| block.release_time)
            .min()
            .unwrap();
        let max_release = problem
            .blocks
            .iter()
            .map(|block| block.release_time)
            .max()
            .unwrap();
        let total_processing = problem.blocks.iter().try_fold(0i64, |sum, block| {
            sum.checked_add(block.processing_time)
                .ok_or_else(|| "sum of processing times overflowed i64".to_string())
        })?;
        let search_horizon = max_release
            .checked_add(total_processing)
            .ok_or_else(|| "preoptimize search horizon overflowed i64".to_string())?;

        Ok(Self {
            orientation_areas,
            bay_load_scale,
            pref_penalty,
            pref_spread,
            min_time,
            search_horizon,
        })
    }
}

struct PreoptimizeContext<'a> {
    pre: &'a PreoptimizePrecompute,
    params: &'a PreoptimizeSolverParams,
    reconstruct_order_params: &'a NeighborParams,
    occupancy: Vec<Vec<Option<f64>>>,
    block_areas: Vec<f64>,
    bay_capacities: Vec<f64>,
    max_time: i64,
}

#[derive(Clone)]
struct PreoptimizeAnnealingState {
    schedule: Vec<PreoptimizedBlock>,
    used_area: Vec<Vec<f64>>,
    loads: Vec<f64>,
    z1: f64,
    z2: f64,
    z3: f64,
    official_objective: f64,
    congestion: f64,
    objective: f64,
}

impl AnnealingState for PreoptimizeAnnealingState {
    fn annealing_score(&self) -> f64 {
        self.objective
    }

    fn has_tardiness(&self) -> bool {
        self.z1 > 0.0
    }

    fn tabu_key(&self) -> Option<u64> {
        None
    }
}

fn normalized_imbalance(loads: &[f64], bay_load_scale: &[f64]) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }
    let mut min_load = f64::INFINITY;
    let mut max_load = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let normalized = bay_load_scale[bay_id] * load;
        min_load = min_load.min(normalized);
        max_load = max_load.max(normalized);
    }
    (max_load - min_load).floor()
}

fn evaluate_schedule(
    problem: &Problem,
    pref_penalty: &[Vec<i64>],
    bay_load_scale: &[f64],
    schedule: &[PreoptimizedBlock],
) -> (f64, f64, f64, f64) {
    let mut z1 = 0.0;
    let mut z3 = 0.0;
    let mut loads = vec![0.0; problem.bays.len()];
    for (block_id, selected) in schedule.iter().enumerate() {
        let block = &problem.blocks[block_id];
        z1 += (selected.entry_time + block.processing_time - block.due_date).max(0) as f64;
        z3 += pref_penalty[block_id][selected.bay_id] as f64;
        loads[selected.bay_id] += block.workload as f64;
    }
    let z2 = normalized_imbalance(&loads, bay_load_scale);
    let objective = problem.weights.w1 * z1 + problem.weights.w2 * z2 + problem.weights.w3 * z3;
    (objective, z1, z2, z3)
}

fn orientation_area(orientation: &Orientation) -> Result<OrientationArea, String> {
    let union = orientation_union(orientation)
        .ok_or_else(|| "orientation must contain at least one layer".to_string())?;

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for layer in &orientation.layers {
        for &[x, y] in layer {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    let perimeter = union
        .0
        .iter()
        .flat_map(|polygon| std::iter::once(polygon.exterior()).chain(polygon.interiors().iter()))
        .flat_map(|ring| ring.0.windows(2))
        .map(|edge| {
            let dx = edge[1].x - edge[0].x;
            let dy = edge[1].y - edge[0].y;
            dx.hypot(dy)
        })
        .sum();

    Ok(OrientationArea {
        union_area: union.unsigned_area(),
        bbox_area: (max_x - min_x) * (max_y - min_y),
        perimeter,
        min_x,
        min_y,
        max_x,
        max_y,
    })
}

fn fits_bay(area: OrientationArea, bay: &Bay) -> bool {
    let min_integer_x = (-area.min_x).ceil() as i64;
    let max_integer_x = (bay.width as f64 - area.max_x).floor() as i64;
    let min_integer_y = (-area.min_y).ceil() as i64;
    let max_integer_y = (bay.height as f64 - area.max_y).floor() as i64;
    min_integer_x <= max_integer_x && min_integer_y <= max_integer_y
}

fn build_occupancy(
    problem: &Problem,
    pre: &PreoptimizePrecompute,
    params: &PreoptimizeSolverParams,
) -> Result<Vec<Vec<Option<f64>>>, String> {
    let occupancy: Vec<Vec<Option<f64>>> = pre
        .orientation_areas
        .iter()
        .map(|areas| {
            problem
                .bays
                .iter()
                .map(|bay| {
                    areas
                        .iter()
                        .copied()
                        .filter(|&area| fits_bay(area, bay))
                        .map(|area| {
                            area.union_area
                                + params.alpha * (area.bbox_area - area.union_area)
                                + params.beta * area.perimeter
                        })
                        .min_by(f64::total_cmp)
                })
                .collect()
        })
        .collect();
    for (block_id, row) in occupancy.iter().enumerate() {
        if row.iter().all(Option::is_none) {
            return Err(format!("block {block_id} has no eligible bay"));
        }
    }
    Ok(occupancy)
}

fn try_build_initial_state(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    order: &[usize],
) -> Option<PreoptimizeAnnealingState> {
    let time_count: usize = (context.pre.search_horizon - context.pre.min_time)
        .try_into()
        .ok()?;
    let mut used_area: Vec<Vec<f64>> = vec![vec![0.0; time_count]; problem.bays.len()];
    let mut congestion = 0.0;
    let mut loads: Vec<f64> = vec![0.0; problem.bays.len()];
    let mut schedule = vec![None; problem.blocks.len()];

    for &block_id in order {
        let block = &problem.blocks[block_id];
        let current_imbalance = normalized_imbalance(&loads, &context.pre.bay_load_scale);
        let mut best: Option<(f64, i64, usize)> = None;
        for (bay_id, occupied_area) in context.occupancy[block_id].iter().copied().enumerate() {
            let Some(occupied_area) = occupied_area else {
                continue;
            };
            let last_entry = context.pre.search_horizon - block.processing_time;
            let entry_time = (block.release_time..=last_entry).find(|&entry_time| {
                (entry_time..entry_time + block.processing_time).all(|time| {
                    used_area[bay_id][(time - context.pre.min_time) as usize] + occupied_area
                        <= context.bay_capacities[bay_id] + 1e-9
                })
            });
            let Some(entry_time) = entry_time else {
                continue;
            };

            let mut next_loads = loads.clone();
            next_loads[bay_id] += block.workload as f64;
            let tardiness = (entry_time + block.processing_time - block.due_date).max(0);
            let bay_capacity = context.bay_capacities[bay_id];
            let ratio = occupied_area / bay_capacity;
            let congestion_delta = (entry_time..entry_time + block.processing_time)
                .map(|time| {
                    ratio * used_area[bay_id][(time - context.pre.min_time) as usize] / bay_capacity
                })
                .sum::<f64>();
            let score = problem.weights.w1 * tardiness as f64
                + problem.weights.w2
                    * (normalized_imbalance(&next_loads, &context.pre.bay_load_scale)
                        - current_imbalance)
                + problem.weights.w3 * context.pre.pref_penalty[block_id][bay_id] as f64
                + problem.weights.w1 * context.params.congestion_weight * congestion_delta;
            let candidate = (score, entry_time, bay_id);
            if best.as_ref().is_none_or(|best| {
                candidate
                    .0
                    .total_cmp(&best.0)
                    .then(candidate.1.cmp(&best.1))
                    .then(candidate.2.cmp(&best.2))
                    .is_lt()
            }) {
                best = Some(candidate);
            }
        }

        let (_, entry_time, bay_id) = best?;
        let selected = PreoptimizedBlock { bay_id, entry_time };
        add_used_area(
            problem,
            context,
            &mut used_area,
            &mut congestion,
            block_id,
            selected,
            1.0,
        );
        loads[bay_id] += block.workload as f64;
        schedule[block_id] = Some(selected);
    }

    let schedule: Vec<_> = schedule.into_iter().collect::<Option<_>>()?;
    let (objective, z1, z2, z3) = evaluate_schedule(
        problem,
        &context.pre.pref_penalty,
        &context.pre.bay_load_scale,
        &schedule,
    );
    Some(PreoptimizeAnnealingState {
        schedule,
        used_area,
        loads,
        z1,
        z2,
        z3,
        official_objective: objective,
        congestion,
        objective: objective + problem.weights.w1 * context.params.congestion_weight * congestion,
    })
}

fn build_initial_state(
    problem: &Problem,
    context: &mut PreoptimizeContext<'_>,
    time_limit: f64,
    timer: Timer,
    max_worker_count: usize,
    seed: u64,
) -> Result<PreoptimizeAnnealingState, String> {
    let deadline = timer.elapsed_seconds() + time_limit;
    let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
    let best = Mutex::new(None::<PreoptimizeAnnealingState>);
    let worker_trials: Vec<_> = (0..worker_count)
        .into_par_iter()
        .map(|worker_id| {
            let mut rng =
                RandPcg64Mcg::new(seed.wrapping_add(1 << 20).wrapping_add(worker_id as u64));
            let mut trials = 0usize;
            let mut completed = 0usize;
            while timer.elapsed_seconds() < deadline {
                trials += 1;
                let mut order: Vec<_> = (0..problem.blocks.len()).collect();
                sort_default_reconstruct_order(
                    problem,
                    &context.block_areas,
                    &context.pre.pref_spread,
                    &mut order,
                    &mut rng,
                    context.reconstruct_order_params,
                );
                let Some(candidate) = try_build_initial_state(problem, context, &order) else {
                    continue;
                };
                completed += 1;

                let mut best = best.lock().unwrap();
                if best
                    .as_ref()
                    .is_none_or(|best| candidate.objective < best.objective)
                {
                    log!(
                        "[{:.4}] [preopt-build] best: worker={}, trial={}, score={:.3}, official={:.3}, tardiness={:.0}",
                        timer.elapsed_seconds(),
                        worker_id,
                        trials,
                        candidate.objective,
                        candidate.official_objective,
                        candidate.z1,
                    );
                    *best = Some(candidate);
                }
            }
            (trials, completed)
        })
        .collect();

    for (worker_id, (trials, completed)) in worker_trials.into_iter().enumerate() {
        log!(
            "[{:.4}] [preopt-build worker={}] trials={}, completed={}",
            timer.elapsed_seconds(),
            worker_id,
            trials,
            completed,
        );
    }

    let mut best = best
        .into_inner()
        .unwrap()
        .ok_or_else(|| "failed to build preoptimize initial state".to_string())?;
    let initial_max_exit = best
        .schedule
        .iter()
        .enumerate()
        .map(|(block_id, selected)| selected.entry_time + problem.blocks[block_id].processing_time)
        .max()
        .unwrap();
    let max_due = problem
        .blocks
        .iter()
        .map(|block| block.due_date)
        .max()
        .unwrap();
    context.max_time = initial_max_exit.max(max_due.min(context.pre.search_horizon));
    let time_count = (context.max_time - context.pre.min_time) as usize;
    for row in &mut best.used_area {
        row.truncate(time_count);
    }
    log!(
        "[{:.4}] [preopt-build] result: score={:.3}, official={:.3}, tardiness={:.0}",
        timer.elapsed_seconds(),
        best.objective,
        best.official_objective,
        best.z1,
    );
    Ok(best)
}

fn block_z1(problem: &Problem, block_id: usize, selected: PreoptimizedBlock) -> f64 {
    let block = &problem.blocks[block_id];
    (selected.entry_time + block.processing_time - block.due_date).max(0) as f64
}

fn block_z3(context: &PreoptimizeContext<'_>, block_id: usize, selected: PreoptimizedBlock) -> f64 {
    context.pre.pref_penalty[block_id][selected.bay_id] as f64
}

fn add_used_area(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    used_area: &mut [Vec<f64>],
    congestion: &mut f64,
    block_id: usize,
    selected: PreoptimizedBlock,
    sign: f64,
) {
    let area = context.occupancy[block_id][selected.bay_id].unwrap();
    let bay_capacity = context.bay_capacities[selected.bay_id];
    let ratio = area / bay_capacity;
    let block = &problem.blocks[block_id];
    for time in selected.entry_time..selected.entry_time + block.processing_time {
        let used = &mut used_area[selected.bay_id][(time - context.pre.min_time) as usize];
        let used_ratio = *used / bay_capacity;
        if sign > 0.0 {
            *congestion += ratio * used_ratio;
        } else {
            *congestion -= ratio * (used_ratio - ratio);
        }
        *used += sign * area;
    }
}

fn can_place(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    used_area: &[Vec<f64>],
    block_id: usize,
    selected: PreoptimizedBlock,
) -> bool {
    let block = &problem.blocks[block_id];
    if selected.entry_time < block.release_time
        || selected.entry_time + block.processing_time > context.max_time
    {
        return false;
    }
    let Some(area) = context.occupancy[block_id][selected.bay_id] else {
        return false;
    };
    (selected.entry_time..selected.entry_time + block.processing_time).all(|time| {
        used_area[selected.bay_id][(time - context.pre.min_time) as usize] + area
            <= context.bay_capacities[selected.bay_id] + 1e-9
    })
}

fn select_block(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> usize {
    if rng.nextf() >= context.params.neighbor.bad_block_select_probability {
        return rng.gen_index(problem.blocks.len());
    }
    let mut best = rng.gen_index(problem.blocks.len());
    let mut best_cost = f64::NEG_INFINITY;
    for _ in 0..context.params.neighbor.bad_block_sample_count {
        let block_id = rng.gen_index(problem.blocks.len());
        let selected = state.schedule[block_id];
        let cost = problem.weights.w1 * block_z1(problem, block_id, selected)
            + problem.weights.w3 * block_z3(context, block_id, selected);
        if cost > best_cost {
            best = block_id;
            best_cost = cost;
        }
    }
    best
}

fn sample_entry_time(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    block_id: usize,
    current: i64,
    rng: &mut impl Random,
) -> i64 {
    let block = &problem.blocks[block_id];
    let min_time = block.release_time;
    let max_time = context.max_time - block.processing_time;
    match rng.gen_range(0, 4) {
        0 => min_time,
        1 => (block.due_date - block.processing_time).clamp(min_time, max_time),
        2 => {
            let max_shift = context.params.neighbor.max_time_shift;
            let delta = rng.gen_range(0, (2 * max_shift + 1) as usize) as i64 - max_shift;
            current.saturating_add(delta).clamp(min_time, max_time)
        }
        _ => min_time + rng.gen_range(0, (max_time - min_time + 1) as usize) as i64,
    }
}

fn try_relocate(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> bool {
    let block_id = select_block(problem, context, state, rng);
    let old = state.schedule[block_id];
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        block_id,
        old,
        -1.0,
    );

    let mut candidate = None;
    for _ in 0..context.params.neighbor.max_relocate_attempts {
        let bay_id = rng.gen_index(problem.bays.len());
        let entry_time = sample_entry_time(problem, context, block_id, old.entry_time, rng);
        let selected = PreoptimizedBlock { bay_id, entry_time };
        if selected != old && can_place(problem, context, &state.used_area, block_id, selected) {
            candidate = Some(selected);
            break;
        }
    }
    let Some(selected) = candidate else {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            block_id,
            old,
            1.0,
        );
        return false;
    };

    let old_z1 = block_z1(problem, block_id, old);
    let new_z1 = block_z1(problem, block_id, selected);
    let old_z3 = block_z3(context, block_id, old);
    let new_z3 = block_z3(context, block_id, selected);
    let mut next_loads = state.loads.clone();
    let workload = problem.blocks[block_id].workload as f64;
    next_loads[old.bay_id] -= workload;
    next_loads[selected.bay_id] += workload;
    let new_z2 = normalized_imbalance(&next_loads, &context.pre.bay_load_scale);
    let raw_delta = problem.weights.w1 * (new_z1 - old_z1)
        + problem.weights.w2 * (new_z2 - state.z2)
        + problem.weights.w3 * (new_z3 - old_z3);

    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        block_id,
        selected,
        1.0,
    );
    state.schedule[block_id] = selected;
    state.loads = next_loads;
    state.z1 += new_z1 - old_z1;
    state.z2 = new_z2;
    state.z3 += new_z3 - old_z3;
    state.official_objective += raw_delta;
    state.objective = state.official_objective
        + problem.weights.w1 * context.params.congestion_weight * state.congestion;
    true
}

fn try_swap(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> bool {
    if problem.blocks.len() < 2 {
        return false;
    }
    let a = select_block(problem, context, state, rng);
    let mut b = rng.gen_index(problem.blocks.len() - 1);
    if b >= a {
        b += 1;
    }
    let old_a = state.schedule[a];
    let old_b = state.schedule[b];
    let (entry_a, entry_b) = if rng.nextf() < 0.5 {
        (old_a.entry_time, old_b.entry_time)
    } else {
        (old_b.entry_time, old_a.entry_time)
    };
    let new_a = PreoptimizedBlock {
        bay_id: old_b.bay_id,
        entry_time: entry_a,
    };
    let new_b = PreoptimizedBlock {
        bay_id: old_a.bay_id,
        entry_time: entry_b,
    };
    if new_a == old_a && new_b == old_b {
        return false;
    }

    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        a,
        old_a,
        -1.0,
    );
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        b,
        old_b,
        -1.0,
    );
    if !can_place(problem, context, &state.used_area, a, new_a) {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            a,
            old_a,
            1.0,
        );
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            b,
            old_b,
            1.0,
        );
        return false;
    }
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        a,
        new_a,
        1.0,
    );
    if !can_place(problem, context, &state.used_area, b, new_b) {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            a,
            new_a,
            -1.0,
        );
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            a,
            old_a,
            1.0,
        );
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            b,
            old_b,
            1.0,
        );
        return false;
    }
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        b,
        new_b,
        1.0,
    );

    let old_z1 = block_z1(problem, a, old_a) + block_z1(problem, b, old_b);
    let new_z1 = block_z1(problem, a, new_a) + block_z1(problem, b, new_b);
    let old_z3 = block_z3(context, a, old_a) + block_z3(context, b, old_b);
    let new_z3 = block_z3(context, a, new_a) + block_z3(context, b, new_b);
    let mut next_loads = state.loads.clone();
    next_loads[old_a.bay_id] -= problem.blocks[a].workload as f64;
    next_loads[old_b.bay_id] -= problem.blocks[b].workload as f64;
    next_loads[new_a.bay_id] += problem.blocks[a].workload as f64;
    next_loads[new_b.bay_id] += problem.blocks[b].workload as f64;
    let new_z2 = normalized_imbalance(&next_loads, &context.pre.bay_load_scale);
    let raw_delta = problem.weights.w1 * (new_z1 - old_z1)
        + problem.weights.w2 * (new_z2 - state.z2)
        + problem.weights.w3 * (new_z3 - old_z3);

    state.schedule[a] = new_a;
    state.schedule[b] = new_b;
    state.loads = next_loads;
    state.z1 += new_z1 - old_z1;
    state.z2 = new_z2;
    state.z3 += new_z3 - old_z3;
    state.official_objective += raw_delta;
    state.objective = state.official_objective
        + problem.weights.w1 * context.params.congestion_weight * state.congestion;
    true
}

fn sample_large_reconstruct_count(
    context: &PreoptimizeContext<'_>,
    rng: &mut impl Random,
) -> usize {
    let params = &context.params.neighbor;
    let span = params.max_removed_blocks - params.min_removed_blocks + 1;
    let u = rng.nextf().powf(params.remove_count_sample_power);
    params.min_removed_blocks + ((u * span as f64) as usize).min(span - 1)
}

fn choose_large_reconstruct_blocks(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &PreoptimizeAnnealingState,
    k: usize,
    rng: &mut impl Random,
) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let seed = select_block(problem, context, state, rng);
    let mut remaining: Vec<_> = (0..problem.blocks.len())
        .filter(|&block_id| block_id != seed)
        .collect();
    rng.shuffle(&mut remaining);
    let mut selected = Vec::with_capacity(k);
    selected.push(seed);
    selected.extend(remaining.into_iter().take(k - 1));
    selected
}

fn try_large_reconstruct(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> bool {
    let k = sample_large_reconstruct_count(context, rng).min(problem.blocks.len());
    if k < 2 {
        return false;
    }
    let mut removed_ids = choose_large_reconstruct_blocks(problem, context, state, k, rng);
    for &block_id in &removed_ids {
        let selected = state.schedule[block_id];
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            block_id,
            selected,
            -1.0,
        );
        state.loads[selected.bay_id] -= problem.blocks[block_id].workload as f64;
    }

    sort_default_reconstruct_order(
        problem,
        &context.block_areas,
        &context.pre.pref_spread,
        &mut removed_ids,
        rng,
        context.reconstruct_order_params,
    );
    let mut changed = false;
    for block_id in removed_ids {
        let old = state.schedule[block_id];
        let mut candidate = None;
        for _ in 0..context.params.neighbor.max_relocate_attempts {
            let selected = PreoptimizedBlock {
                bay_id: rng.gen_index(problem.bays.len()),
                entry_time: sample_entry_time(problem, context, block_id, old.entry_time, rng),
            };
            if can_place(problem, context, &state.used_area, block_id, selected) {
                candidate = Some(selected);
                break;
            }
        }
        let Some(selected) = candidate else {
            return false;
        };
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            block_id,
            selected,
            1.0,
        );
        state.loads[selected.bay_id] += problem.blocks[block_id].workload as f64;
        state.schedule[block_id] = selected;
        changed |= selected != old;
    }
    if !changed {
        return false;
    }

    let (objective, z1, z2, z3) = evaluate_schedule(
        problem,
        &context.pre.pref_penalty,
        &context.pre.bay_load_scale,
        &state.schedule,
    );
    state.official_objective = objective;
    state.objective =
        objective + problem.weights.w1 * context.params.congestion_weight * state.congestion;
    state.z1 = z1;
    state.z2 = z2;
    state.z3 = z3;
    true
}

const PREOPTIMIZE_NEIGHBOR_KINDS: &[&str] = &["Relocate", "Swap", "LargeReconstruct"];

struct PreoptimizeAnnealingDelegate<'a> {
    problem: &'a Problem,
    context: PreoptimizeContext<'a>,
    initial: PreoptimizeAnnealingState,
}

impl AnnealingDelegate for PreoptimizeAnnealingDelegate<'_> {
    type State = PreoptimizeAnnealingState;
    type Output = PreoptimizeState;

    fn initial_states(&self) -> Vec<Self::State> {
        return vec![self.initial.clone()];
    }

    fn name(&self) -> &'static str {
        "preopt"
    }

    fn neighbor_kinds(&self) -> &'static [&'static str] {
        PREOPTIMIZE_NEIGHBOR_KINDS
    }

    fn propose(
        &self,
        _domain: usize,
        current: &Self::State,
        _accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let mut candidate = current.clone();
        let probabilities = &self.context.params.neighbor_probabilities;
        let total = probabilities.relocate + probabilities.swap + probabilities.large_reconstruct;
        let x = rng.nextf() * total;
        let (neighbor_kind, succeeded) = if x < probabilities.large_reconstruct {
            (
                2,
                try_large_reconstruct(self.problem, &self.context, &mut candidate, rng),
            )
        } else if x < probabilities.large_reconstruct + probabilities.swap {
            (
                1,
                try_swap(self.problem, &self.context, &mut candidate, rng),
            )
        } else {
            (
                0,
                try_relocate(self.problem, &self.context, &mut candidate, rng),
            )
        };
        AnnealingAttempt {
            neighbor_kind,
            candidate: succeeded.then_some(candidate),
        }
    }

    fn is_finished(&self, _domain: usize, _state: &Self::State) -> bool {
        false
    }

    fn finish(&self, mut states: Vec<Self::State>) -> Self::Output {
        let state = states.pop().unwrap();
        let score = evaluate_schedule(
            self.problem,
            &self.context.pre.pref_penalty,
            &self.context.pre.bay_load_scale,
            &state.schedule,
        )
        .0;
        log!(
            "[preopt] objective: official={score:.3}, congestion={:.3}, search={:.3}",
            state.congestion,
            state.objective,
        );
        PreoptimizeState {
            score,
            blocks: state.schedule,
        }
    }
}

pub fn preoptimize(
    problem: &Problem,
    pre: &PreoptimizePrecompute,
    params: &PreoptimizeSolverParams,
    annealing: &AnnealingParamsConfig,
    reconstruct_order_params: &NeighborParams,
    time_limit: f64,
    max_worker_count: usize,
    seed: u64,
) -> Result<PreoptimizeState, String> {
    let occupancy = build_occupancy(problem, pre, params)?;
    let bay_capacities = problem
        .bays
        .iter()
        .map(|bay| {
            (bay.width as f64 - params.bay_padding) * (bay.height as f64 - params.bay_padding)
        })
        .collect();
    let block_areas = occupancy
        .iter()
        .map(|areas| {
            areas
                .iter()
                .flatten()
                .copied()
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let mut context = PreoptimizeContext {
        pre,
        params,
        reconstruct_order_params,
        occupancy,
        block_areas,
        bay_capacities,
        max_time: pre.search_horizon,
    };
    let timer = Timer::start(1.0);
    let initial_build_time_limit =
        (time_limit * params.initial_build.time_ratio).min(params.initial_build.max_seconds);
    let initial = build_initial_state(
        problem,
        &mut context,
        initial_build_time_limit,
        timer,
        max_worker_count,
        seed,
    )?;
    let annealing_params = annealing.make(problem, initial.annealing_score());
    let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
    let delegate = PreoptimizeAnnealingDelegate {
        problem,
        context,
        initial,
    };
    Ok(Annealer::new(time_limit, worker_count, seed, annealing_params, delegate).run(timer))
}
