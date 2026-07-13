use std::{num::NonZeroU32, time::Instant};

use geo::{Area, BooleanOps, Coord, LineString, MultiPolygon, Polygon};
use highs::{Col, HighsModelStatus, HighsSolutionStatus, RowProblem, Sense};
use serde::Serialize;

use crate::{Bay, Orientation, Problem};

#[derive(Clone, Copy, Debug)]
pub struct PreoptimizeParams {
    pub alpha: f64,
    pub beta: f64,
    pub time_limit: f64,
    pub horizon_margin: i64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PreoptimizedBlock {
    pub bay_id: usize,
    pub entry_time: i64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub enum PreoptimizeStatus {
    Optimal,
    TimeLimit,
}

#[derive(Debug, Serialize)]
pub struct PreoptimizeResult {
    pub status: PreoptimizeStatus,
    pub objective: f64,
    pub initial_objective: f64,
    pub z1: f64,
    pub z2: f64,
    pub z3: f64,
    pub blocks: Vec<PreoptimizedBlock>,
    pub horizon: i64,
    pub variable_count: usize,
    pub constraint_count: usize,
    pub mip_gap: Option<f64>,
    pub model_build_seconds: f64,
    pub solve_seconds: f64,
}

#[derive(Clone, Copy)]
struct OrientationArea {
    union_area: f64,
    bbox_area: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

#[derive(Clone, Copy)]
struct StartVariable {
    col: Col,
    col_index: usize,
    bay_id: usize,
    entry_time: i64,
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
    max_load - min_load
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

fn build_greedy_schedule(
    problem: &Problem,
    occupancy: &[Vec<Option<f64>>],
    pref_penalty: &[Vec<i64>],
    bay_load_scale: &[f64],
    min_time: i64,
    search_horizon: i64,
) -> Result<Vec<PreoptimizedBlock>, String> {
    let time_count: usize = (search_horizon - min_time)
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
        let area_a = occupancy[a]
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let area_b = occupancy[b]
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
        let current_imbalance = normalized_imbalance(&loads, bay_load_scale);
        let mut best: Option<(f64, i64, usize)> = None;
        for (bay_id, occupied_area) in occupancy[block_id].iter().copied().enumerate() {
            let Some(occupied_area) = occupied_area else {
                continue;
            };
            let bay_area = (problem.bays[bay_id].width * problem.bays[bay_id].height) as f64;
            let last_entry = search_horizon - block.processing_time;
            let entry_time = (block.release_time..=last_entry).find(|&entry_time| {
                (entry_time..entry_time + block.processing_time).all(|time| {
                    used_area[bay_id][(time - min_time) as usize] + occupied_area <= bay_area + 1e-9
                })
            });
            let Some(entry_time) = entry_time else {
                continue;
            };

            let mut next_loads = loads.clone();
            next_loads[bay_id] += block.workload as f64;
            let tardiness = (entry_time + block.processing_time - block.due_date).max(0);
            let score = problem.weights.w1 * tardiness as f64
                + problem.weights.w2
                    * (normalized_imbalance(&next_loads, bay_load_scale) - current_imbalance)
                + problem.weights.w3 * pref_penalty[block_id][bay_id] as f64;
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
        let occupied_area = occupancy[block_id][bay_id].unwrap();
        for time in entry_time..entry_time + block.processing_time {
            used_area[bay_id][(time - min_time) as usize] += occupied_area;
        }
        loads[bay_id] += block.workload as f64;
        schedule[block_id] = Some(PreoptimizedBlock { bay_id, entry_time });
    }

    schedule
        .into_iter()
        .enumerate()
        .map(|(block_id, selected)| {
            selected.ok_or_else(|| format!("block {block_id} was not greedily scheduled"))
        })
        .collect()
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
    params: PreoptimizeParams,
) -> Result<Vec<Vec<Option<f64>>>, String> {
    let orientation_areas: Vec<Vec<OrientationArea>> = problem
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
        .collect::<Result<_, _>>()?;

    Ok(orientation_areas
        .iter()
        .map(|areas| {
            problem
                .bays
                .iter()
                .map(|bay| {
                    let bay_area = (bay.width * bay.height) as f64;
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
                        .filter(|&area| area <= bay_area + 1e-9)
                        .min_by(f64::total_cmp)
                })
                .collect()
        })
        .collect())
}

pub fn preoptimize(
    problem: &Problem,
    params: PreoptimizeParams,
) -> Result<PreoptimizeResult, String> {
    let build_start = Instant::now();
    if problem.bays.is_empty() {
        return Err("bays must be non-empty".to_string());
    }
    if !params.alpha.is_finite() || params.alpha < 0.0 {
        return Err("alpha must be a finite non-negative number".to_string());
    }
    if !params.beta.is_finite() || params.beta < 0.0 {
        return Err("beta must be a finite non-negative number".to_string());
    }
    if !params.time_limit.is_finite() || params.time_limit <= 0.0 {
        return Err("time_limit must be a finite positive number".to_string());
    }
    if params.horizon_margin < 0 {
        return Err("horizon_margin must be non-negative".to_string());
    }
    if problem.blocks.is_empty() {
        return Ok(PreoptimizeResult {
            status: PreoptimizeStatus::Optimal,
            objective: 0.0,
            initial_objective: 0.0,
            z1: 0.0,
            z2: 0.0,
            z3: 0.0,
            blocks: Vec::new(),
            horizon: 0,
            variable_count: 0,
            constraint_count: 0,
            mip_gap: Some(0.0),
            model_build_seconds: build_start.elapsed().as_secs_f64(),
            solve_seconds: 0.0,
        });
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

    let occupancy = build_occupancy(problem, params)?;
    for (block_id, row) in occupancy.iter().enumerate() {
        if row.iter().all(Option::is_none) {
            return Err(format!("block {block_id} has no eligible bay"));
        }
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

    let bay_areas: Vec<f64> = problem
        .bays
        .iter()
        .map(|bay| (bay.width * bay.height) as f64)
        .collect();
    let avg_bay_area = bay_areas.iter().sum::<f64>() / bay_areas.len() as f64;
    let bay_load_scale: Vec<f64> = bay_areas.iter().map(|&area| avg_bay_area / area).collect();
    let pref_penalty: Vec<Vec<i64>> = problem
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
    let initial = build_greedy_schedule(
        problem,
        &occupancy,
        &pref_penalty,
        &bay_load_scale,
        min_time,
        search_horizon,
    )?;
    let (initial_objective, _, initial_z2, _) =
        evaluate_schedule(problem, &pref_penalty, &bay_load_scale, &initial);
    let greedy_max_exit = initial
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
    let horizon = greedy_max_exit
        .max(max_due)
        .checked_add(params.horizon_margin)
        .ok_or_else(|| "preoptimize horizon overflowed i64".to_string())?;
    let time_count: usize = (horizon - min_time)
        .try_into()
        .map_err(|_| "preoptimize time range is too large".to_string())?;

    let mut model = RowProblem::default().optimise(Sense::Minimise);
    let mut assignment_factors = vec![Vec::new(); problem.blocks.len()];
    let mut start_factors: Vec<Vec<Vec<(Col, f64)>>> = (0..problem.bays.len())
        .map(|_| (0..time_count).map(|_| Vec::new()).collect())
        .collect();
    let mut end_factors: Vec<Vec<Vec<(Col, f64)>>> = (0..problem.bays.len())
        .map(|_| (0..time_count).map(|_| Vec::new()).collect())
        .collect();
    let mut variables_by_block = vec![Vec::new(); problem.blocks.len()];
    let mut variable_count = 0usize;
    for (block_id, block) in problem.blocks.iter().enumerate() {
        let last_entry = horizon - block.processing_time;
        for bay_id in 0..problem.bays.len() {
            let Some(occupied_area) = occupancy[block_id][bay_id] else {
                continue;
            };
            for entry_time in block.release_time..=last_entry {
                let tardiness = (entry_time + block.processing_time - block.due_date).max(0);
                let objective = problem.weights.w1 * tardiness as f64
                    + problem.weights.w3 * pref_penalty[block_id][bay_id] as f64;
                let col = model.add_integer_column(objective, 0.0..=1.0, []);
                assignment_factors[block_id].push((col, 1.0));
                start_factors[bay_id][(entry_time - min_time) as usize].push((col, occupied_area));
                let exit_time = entry_time + block.processing_time;
                if exit_time < horizon {
                    end_factors[bay_id][(exit_time - min_time) as usize].push((col, occupied_area));
                }
                variables_by_block[block_id].push(StartVariable {
                    col,
                    col_index: variable_count,
                    bay_id,
                    entry_time,
                });
                variable_count += 1;
            }
        }
    }

    let mut usage_variables = Vec::with_capacity(problem.bays.len());
    for &bay_area in &bay_areas {
        let mut row = Vec::with_capacity(time_count);
        for _ in 0..time_count {
            let col_index = variable_count;
            let col = model.add_col(0.0, 0.0..=bay_area, []);
            variable_count += 1;
            row.push((col, col_index));
        }
        usage_variables.push(row);
    }
    let z2_index = variable_count;
    let z2_col = model.add_col(problem.weights.w2, 0.0.., []);
    variable_count += 1;
    for factors in assignment_factors {
        model.add_row(1.0..=1.0, factors);
    }
    for bay_id in 0..problem.bays.len() {
        for time_idx in 0..time_count {
            let mut factors = Vec::with_capacity(
                2 + start_factors[bay_id][time_idx].len() + end_factors[bay_id][time_idx].len(),
            );
            factors.push((usage_variables[bay_id][time_idx].0, 1.0));
            if time_idx > 0 {
                factors.push((usage_variables[bay_id][time_idx - 1].0, -1.0));
            }
            factors.extend(
                start_factors[bay_id][time_idx]
                    .drain(..)
                    .map(|(col, area)| (col, -area)),
            );
            factors.extend(
                end_factors[bay_id][time_idx]
                    .drain(..)
                    .map(|(col, area)| (col, area)),
            );
            model.add_row(0.0..=0.0, factors);
        }
    }
    for a in 0..problem.bays.len() {
        for b in (a + 1)..problem.bays.len() {
            let mut positive = vec![(z2_col, -1.0)];
            let mut negative = vec![(z2_col, -1.0)];
            for (block_id, variables) in variables_by_block.iter().enumerate() {
                for variable in variables {
                    let coefficient =
                        bay_load_scale[variable.bay_id] * problem.blocks[block_id].workload as f64;
                    if variable.bay_id == a {
                        positive.push((variable.col, coefficient));
                        negative.push((variable.col, -coefficient));
                    } else if variable.bay_id == b {
                        positive.push((variable.col, -coefficient));
                        negative.push((variable.col, coefficient));
                    }
                }
            }
            model.add_row(..=0.0, positive);
            model.add_row(..=0.0, negative);
        }
    }
    let constraint_count = problem.blocks.len()
        + problem.bays.len() * time_count
        + problem.bays.len() * (problem.bays.len() - 1);
    let mut initial_columns = vec![0.0; variable_count];
    for (block_id, selected) in initial.iter().enumerate() {
        let variable = variables_by_block[block_id]
            .iter()
            .find(|variable| {
                variable.bay_id == selected.bay_id && variable.entry_time == selected.entry_time
            })
            .ok_or_else(|| format!("greedy start for block {block_id} is outside the MILP"))?;
        initial_columns[variable.col_index] = 1.0;
        let occupied_area = occupancy[block_id][selected.bay_id].unwrap();
        for time in
            selected.entry_time..selected.entry_time + problem.blocks[block_id].processing_time
        {
            initial_columns[usage_variables[selected.bay_id][(time - min_time) as usize].1] +=
                occupied_area;
        }
    }
    initial_columns[z2_index] = initial_z2;

    model.make_quiet();
    model.set_threads(NonZeroU32::new(1).unwrap());
    model.set_option("time_limit", params.time_limit);
    model.set_solution(Some(&initial_columns), None, None, None);
    let model_build_seconds = build_start.elapsed().as_secs_f64();
    let solve_start = Instant::now();
    let solved = model
        .try_solve()
        .map_err(|status| format!("HiGHS failed to solve model: {status:?}"))?;
    let solve_seconds = solve_start.elapsed().as_secs_f64();
    let model_status = solved.status();
    let status = match model_status {
        HighsModelStatus::Optimal => PreoptimizeStatus::Optimal,
        HighsModelStatus::ReachedTimeLimit
            if solved.primal_solution_status() == HighsSolutionStatus::Feasible =>
        {
            PreoptimizeStatus::TimeLimit
        }
        _ => {
            return Err(format!(
                "HiGHS did not produce a feasible solution: model={model_status:?}, primal={:?}",
                solved.primal_solution_status()
            ));
        }
    };

    let solution = solved.get_solution();
    let mut blocks = Vec::with_capacity(problem.blocks.len());
    for (block_id, variables) in variables_by_block.iter().enumerate() {
        let selected = variables
            .iter()
            .max_by(|a, b| solution[a.col].total_cmp(&solution[b.col]))
            .ok_or_else(|| format!("block {block_id} has no start variable"))?;
        if solution[selected.col] < 0.5 {
            return Err(format!(
                "block {block_id} has no selected start variable in the HiGHS solution"
            ));
        }
        blocks.push(PreoptimizedBlock {
            bay_id: selected.bay_id,
            entry_time: selected.entry_time,
        });
    }

    let (objective, z1, z2, z3) =
        evaluate_schedule(problem, &pref_penalty, &bay_load_scale, &blocks);
    let mip_gap = solved.mip_gap();

    Ok(PreoptimizeResult {
        status,
        objective,
        initial_objective,
        z1,
        z2,
        z3,
        blocks,
        horizon,
        variable_count,
        constraint_count,
        mip_gap: mip_gap.is_finite().then_some(mip_gap),
        model_build_seconds,
        solve_seconds,
    })
}
