use crate::{
    Bay, Orientation, Problem,
    util::rand::{RandPcg64Mcg, Random},
};
use geo::{Area, BooleanOps, Coord, LineString, MultiPolygon, Polygon};
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct PreoptimizeParams {
    pub alpha: f64,
    pub beta: f64,
    pub time_limit: f64,
    pub distance_weight: f64,
    pub rng_seed: u64,
    pub swap_probability: f64,
    pub bad_block_sample_count: usize,
    pub bad_block_select_probability: f64,
    pub max_relocate_attempts: usize,
    pub max_time_shift: i64,
    pub end_temperature_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PreoptimizedBlock {
    pub bay_id: usize,
    pub entry_time: i64,
}

#[derive(Debug, Serialize)]
pub struct PreoptimizeResult {
    pub objective: f64,
    pub regularized_objective: f64,
    pub z1: f64,
    pub z2: f64,
    pub z3: f64,
    pub bay_distance: usize,
    pub blocks: Vec<PreoptimizedBlock>,
}

#[derive(Clone, Copy, Debug)]
struct OrientationArea {
    union_area: f64,
    bbox_area: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

#[derive(Debug)]
pub struct PreoptimizePrecompute {
    orientation_areas: Vec<Vec<OrientationArea>>,
    bay_areas: Vec<f64>,
    bay_load_scale: Vec<f64>,
    pref_penalty: Vec<Vec<i64>>,
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

        if problem.blocks.is_empty() {
            return Ok(Self {
                orientation_areas,
                bay_areas,
                bay_load_scale,
                pref_penalty,
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
            bay_areas,
            bay_load_scale,
            pref_penalty,
            min_time,
            search_horizon,
        })
    }
}

struct PreoptimizeContext<'a> {
    pre: &'a PreoptimizePrecompute,
    reference: Option<&'a [PreoptimizedBlock]>,
    occupancy: Vec<Vec<Option<f64>>>,
    max_time: i64,
}

struct AnnealingState {
    schedule: Vec<PreoptimizedBlock>,
    used_area: Vec<Vec<f64>>,
    loads: Vec<f64>,
    z1: f64,
    z2: f64,
    z3: f64,
    objective: f64,
    bay_distance: usize,
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

fn layer_polygon(layer: &[[f64; 2]]) -> Polygon<f64> {
    let mut coords: Vec<_> = layer.iter().map(|&[x, y]| Coord { x, y }).collect();
    if coords.first() != coords.last() {
        coords.push(coords[0]);
    }
    Polygon::new(LineString::from(coords), vec![])
}

fn orientation_area(orientation: &Orientation) -> Result<OrientationArea, String> {
    let mut polygons = orientation.layers.iter().map(|layer| layer_polygon(layer));
    let first = polygons
        .next()
        .ok_or_else(|| "orientation must contain at least one layer".to_string())?;
    let mut union = MultiPolygon(vec![first]);
    for polygon in polygons {
        union = union.union(&polygon);
    }

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

    Ok(OrientationArea {
        union_area: union.unsigned_area(),
        bbox_area: (max_x - min_x) * (max_y - min_y),
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
    params: PreoptimizeParams,
) -> Result<Vec<Vec<Option<f64>>>, String> {
    let occupancy: Vec<Vec<Option<f64>>> = pre
        .orientation_areas
        .iter()
        .map(|areas| {
            problem
                .bays
                .iter()
                .enumerate()
                .map(|(bay_id, bay)| {
                    let extra = params.beta * bay.width.min(bay.height) as f64;
                    areas
                        .iter()
                        .copied()
                        .filter(|&area| fits_bay(area, bay))
                        .map(|area| {
                            area.union_area
                                + params.alpha * (area.bbox_area - area.union_area)
                                + extra
                        })
                        .filter(|&area| area <= pre.bay_areas[bay_id] + 1e-9)
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

fn build_initial_state(
    problem: &Problem,
    context: &mut PreoptimizeContext<'_>,
    params: PreoptimizeParams,
) -> Result<AnnealingState, String> {
    let time_count: usize = (context.pre.search_horizon - context.pre.min_time)
        .try_into()
        .map_err(|_| "greedy time range is too large".to_string())?;
    let mut used_area = vec![vec![0.0; time_count]; problem.bays.len()];
    let mut loads = vec![0.0; problem.bays.len()];
    let mut schedule = vec![None; problem.blocks.len()];
    let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
    order.sort_by(|&a, &b| {
        let block_a = &problem.blocks[a];
        let block_b = &problem.blocks[b];
        let slack_a = block_a.due_date - block_a.release_time - block_a.processing_time;
        let slack_b = block_b.due_date - block_b.release_time - block_b.processing_time;
        let area_a = context.occupancy[a]
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let area_b = context.occupancy[b]
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min);
        slack_a
            .cmp(&slack_b)
            .then(block_a.due_date.cmp(&block_b.due_date))
            .then(area_b.total_cmp(&area_a))
            .then(a.cmp(&b))
    });

    for block_id in order {
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
                        <= context.pre.bay_areas[bay_id] + 1e-9
                })
            });
            let Some(entry_time) = entry_time else {
                continue;
            };

            let mut next_loads = loads.clone();
            next_loads[bay_id] += block.workload as f64;
            let tardiness = (entry_time + block.processing_time - block.due_date).max(0);
            let distance = context
                .reference
                .is_some_and(|reference| reference[block_id].bay_id != bay_id);
            let score = problem.weights.w1 * tardiness as f64
                + problem.weights.w2
                    * (normalized_imbalance(&next_loads, &context.pre.bay_load_scale)
                        - current_imbalance)
                + problem.weights.w3 * context.pre.pref_penalty[block_id][bay_id] as f64
                + params.distance_weight * distance as usize as f64;
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

        let (_, entry_time, bay_id) =
            best.ok_or_else(|| format!("failed to greedily schedule block {block_id}"))?;
        let occupied_area = context.occupancy[block_id][bay_id].unwrap();
        for time in entry_time..entry_time + block.processing_time {
            used_area[bay_id][(time - context.pre.min_time) as usize] += occupied_area;
        }
        loads[bay_id] += block.workload as f64;
        schedule[block_id] = Some(PreoptimizedBlock { bay_id, entry_time });
    }

    let schedule: Vec<_> = schedule
        .into_iter()
        .enumerate()
        .map(|(block_id, selected)| {
            selected.ok_or_else(|| format!("block {block_id} was not greedily scheduled"))
        })
        .collect::<Result<_, _>>()?;
    let initial_max_exit = schedule
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
    for row in &mut used_area {
        row.truncate(time_count);
    }

    let (objective, z1, z2, z3) = evaluate_schedule(
        problem,
        &context.pre.pref_penalty,
        &context.pre.bay_load_scale,
        &schedule,
    );
    let bay_distance = context.reference.map_or(0, |reference| {
        schedule
            .iter()
            .zip(reference)
            .filter(|(selected, reference)| selected.bay_id != reference.bay_id)
            .count()
    });
    Ok(AnnealingState {
        schedule,
        used_area,
        loads,
        z1,
        z2,
        z3,
        objective,
        bay_distance,
    })
}

fn block_z1(problem: &Problem, block_id: usize, selected: PreoptimizedBlock) -> f64 {
    let block = &problem.blocks[block_id];
    (selected.entry_time + block.processing_time - block.due_date).max(0) as f64
}

fn block_z3(context: &PreoptimizeContext<'_>, block_id: usize, selected: PreoptimizedBlock) -> f64 {
    context.pre.pref_penalty[block_id][selected.bay_id] as f64
}

fn block_distance(
    context: &PreoptimizeContext<'_>,
    block_id: usize,
    selected: PreoptimizedBlock,
) -> usize {
    context
        .reference
        .is_some_and(|reference| reference[block_id].bay_id != selected.bay_id) as usize
}

fn add_used_area(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    used_area: &mut [Vec<f64>],
    block_id: usize,
    selected: PreoptimizedBlock,
    sign: f64,
) {
    let area = context.occupancy[block_id][selected.bay_id].unwrap();
    let block = &problem.blocks[block_id];
    for time in selected.entry_time..selected.entry_time + block.processing_time {
        used_area[selected.bay_id][(time - context.pre.min_time) as usize] += sign * area;
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
            <= context.pre.bay_areas[selected.bay_id] + 1e-9
    })
}

fn select_block(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &AnnealingState,
    params: PreoptimizeParams,
    rng: &mut impl Random,
) -> usize {
    if rng.nextf() >= params.bad_block_select_probability {
        return rng.gen_index(problem.blocks.len());
    }
    let mut best = rng.gen_index(problem.blocks.len());
    let mut best_cost = f64::NEG_INFINITY;
    for _ in 0..params.bad_block_sample_count {
        let block_id = rng.gen_index(problem.blocks.len());
        let selected = state.schedule[block_id];
        let cost = problem.weights.w1 * block_z1(problem, block_id, selected)
            + problem.weights.w3 * block_z3(context, block_id, selected)
            + params.distance_weight * block_distance(context, block_id, selected) as f64;
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
    max_time_shift: i64,
    rng: &mut impl Random,
) -> i64 {
    let block = &problem.blocks[block_id];
    let min_time = block.release_time;
    let max_time = context.max_time - block.processing_time;
    match rng.gen_range(0, 4) {
        0 => min_time,
        1 => (block.due_date - block.processing_time).clamp(min_time, max_time),
        2 => {
            let delta = rng.gen_range(0, (2 * max_time_shift + 1) as usize) as i64 - max_time_shift;
            current.saturating_add(delta).clamp(min_time, max_time)
        }
        _ => min_time + rng.gen_range(0, (max_time - min_time + 1) as usize) as i64,
    }
}

fn accept(delta: f64, temperature: f64, rng: &mut impl Random) -> bool {
    delta <= 0.0 || rng.nextf() < (-delta / temperature).exp()
}

fn try_relocate(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut AnnealingState,
    params: PreoptimizeParams,
    temperature: f64,
    rng: &mut impl Random,
) -> bool {
    let block_id = select_block(problem, context, state, params, rng);
    let old = state.schedule[block_id];
    add_used_area(problem, context, &mut state.used_area, block_id, old, -1.0);

    let mut candidate = None;
    for _ in 0..params.max_relocate_attempts {
        let bay_id = rng.gen_index(problem.bays.len());
        let entry_time = sample_entry_time(
            problem,
            context,
            block_id,
            old.entry_time,
            params.max_time_shift,
            rng,
        );
        let selected = PreoptimizedBlock { bay_id, entry_time };
        if selected != old && can_place(problem, context, &state.used_area, block_id, selected) {
            candidate = Some(selected);
            break;
        }
    }
    let Some(selected) = candidate else {
        add_used_area(problem, context, &mut state.used_area, block_id, old, 1.0);
        return false;
    };

    let old_z1 = block_z1(problem, block_id, old);
    let new_z1 = block_z1(problem, block_id, selected);
    let old_z3 = block_z3(context, block_id, old);
    let new_z3 = block_z3(context, block_id, selected);
    let old_distance = block_distance(context, block_id, old);
    let new_distance = block_distance(context, block_id, selected);
    let mut next_loads = state.loads.clone();
    let workload = problem.blocks[block_id].workload as f64;
    next_loads[old.bay_id] -= workload;
    next_loads[selected.bay_id] += workload;
    let new_z2 = normalized_imbalance(&next_loads, &context.pre.bay_load_scale);
    let raw_delta = problem.weights.w1 * (new_z1 - old_z1)
        + problem.weights.w2 * (new_z2 - state.z2)
        + problem.weights.w3 * (new_z3 - old_z3);
    let distance_delta = new_distance as i64 - old_distance as i64;
    let delta = raw_delta + params.distance_weight * distance_delta as f64;

    if accept(delta, temperature, rng) {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            block_id,
            selected,
            1.0,
        );
        state.schedule[block_id] = selected;
        state.loads = next_loads;
        state.z1 += new_z1 - old_z1;
        state.z2 = new_z2;
        state.z3 += new_z3 - old_z3;
        state.objective += raw_delta;
        state.bay_distance = (state.bay_distance as i64 + distance_delta) as usize;
        true
    } else {
        add_used_area(problem, context, &mut state.used_area, block_id, old, 1.0);
        false
    }
}

fn try_swap(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut AnnealingState,
    params: PreoptimizeParams,
    temperature: f64,
    rng: &mut impl Random,
) -> bool {
    if problem.blocks.len() < 2 {
        return false;
    }
    let a = select_block(problem, context, state, params, rng);
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

    add_used_area(problem, context, &mut state.used_area, a, old_a, -1.0);
    add_used_area(problem, context, &mut state.used_area, b, old_b, -1.0);
    if !can_place(problem, context, &state.used_area, a, new_a) {
        add_used_area(problem, context, &mut state.used_area, a, old_a, 1.0);
        add_used_area(problem, context, &mut state.used_area, b, old_b, 1.0);
        return false;
    }
    add_used_area(problem, context, &mut state.used_area, a, new_a, 1.0);
    if !can_place(problem, context, &state.used_area, b, new_b) {
        add_used_area(problem, context, &mut state.used_area, a, new_a, -1.0);
        add_used_area(problem, context, &mut state.used_area, a, old_a, 1.0);
        add_used_area(problem, context, &mut state.used_area, b, old_b, 1.0);
        return false;
    }
    add_used_area(problem, context, &mut state.used_area, b, new_b, 1.0);

    let old_z1 = block_z1(problem, a, old_a) + block_z1(problem, b, old_b);
    let new_z1 = block_z1(problem, a, new_a) + block_z1(problem, b, new_b);
    let old_z3 = block_z3(context, a, old_a) + block_z3(context, b, old_b);
    let new_z3 = block_z3(context, a, new_a) + block_z3(context, b, new_b);
    let old_distance = block_distance(context, a, old_a) + block_distance(context, b, old_b);
    let new_distance = block_distance(context, a, new_a) + block_distance(context, b, new_b);
    let mut next_loads = state.loads.clone();
    next_loads[old_a.bay_id] -= problem.blocks[a].workload as f64;
    next_loads[old_b.bay_id] -= problem.blocks[b].workload as f64;
    next_loads[new_a.bay_id] += problem.blocks[a].workload as f64;
    next_loads[new_b.bay_id] += problem.blocks[b].workload as f64;
    let new_z2 = normalized_imbalance(&next_loads, &context.pre.bay_load_scale);
    let raw_delta = problem.weights.w1 * (new_z1 - old_z1)
        + problem.weights.w2 * (new_z2 - state.z2)
        + problem.weights.w3 * (new_z3 - old_z3);
    let distance_delta = new_distance as i64 - old_distance as i64;
    let delta = raw_delta + params.distance_weight * distance_delta as f64;

    if accept(delta, temperature, rng) {
        state.schedule[a] = new_a;
        state.schedule[b] = new_b;
        state.loads = next_loads;
        state.z1 += new_z1 - old_z1;
        state.z2 = new_z2;
        state.z3 += new_z3 - old_z3;
        state.objective += raw_delta;
        state.bay_distance = (state.bay_distance as i64 + distance_delta) as usize;
        true
    } else {
        add_used_area(problem, context, &mut state.used_area, a, new_a, -1.0);
        add_used_area(problem, context, &mut state.used_area, b, new_b, -1.0);
        add_used_area(problem, context, &mut state.used_area, a, old_a, 1.0);
        add_used_area(problem, context, &mut state.used_area, b, old_b, 1.0);
        false
    }
}

pub fn preoptimize(
    problem: &Problem,
    pre: &PreoptimizePrecompute,
    reference: Option<&[PreoptimizedBlock]>,
    params: PreoptimizeParams,
) -> Result<PreoptimizeResult, String> {
    let occupancy = build_occupancy(problem, pre, params)?;
    let mut context = PreoptimizeContext {
        pre,
        reference,
        occupancy,
        max_time: pre.search_horizon,
    };
    let mut state = build_initial_state(problem, &mut context, params)?;
    let mut best = state.schedule.clone();
    let mut best_regularized = state.objective + params.distance_weight * state.bay_distance as f64;
    let mut rng = RandPcg64Mcg::new(params.rng_seed);
    let start_temperature = (best_regularized / problem.blocks.len() as f64)
        .max(problem.weights.w1)
        .max(problem.weights.w3)
        .max(params.distance_weight)
        .max(1.0);
    let end_temperature = start_temperature * params.end_temperature_ratio;
    let solve_start = Instant::now();
    let mut iter = 0usize;

    loop {
        let elapsed = solve_start.elapsed().as_secs_f64();
        if elapsed >= params.time_limit {
            break;
        }
        iter += 1;
        let progress = (elapsed / params.time_limit).clamp(0.0, 1.0);
        let temperature = start_temperature * (end_temperature / start_temperature).powf(progress);
        if rng.nextf() < params.swap_probability {
            try_swap(problem, &context, &mut state, params, temperature, &mut rng);
        } else {
            try_relocate(problem, &context, &mut state, params, temperature, &mut rng);
        }
        let regularized = state.objective + params.distance_weight * state.bay_distance as f64;
        if regularized + 1e-9 < best_regularized {
            best_regularized = regularized;
            best.clone_from(&state.schedule);
            eprintln!(
                "[{:.4}] annealing new best: iter={:8}, score={:.3}, z1={:.3}, z2={:.3}, z3={:.3}",
                solve_start.elapsed().as_secs_f64(),
                iter,
                regularized,
                state.z1,
                state.z2,
                state.z3,
            );
        }
    }

    let (objective, z1, z2, z3) = evaluate_schedule(
        problem,
        &context.pre.pref_penalty,
        &context.pre.bay_load_scale,
        &best,
    );
    let bay_distance = reference.map_or(0, |reference| {
        best.iter()
            .zip(reference)
            .filter(|(selected, reference)| selected.bay_id != reference.bay_id)
            .count()
    });
    Ok(PreoptimizeResult {
        objective,
        regularized_objective: objective + params.distance_weight * bay_distance as f64,
        z1,
        z2,
        z3,
        bay_distance,
        blocks: best,
    })
}
