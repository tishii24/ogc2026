use std::{num::NonZeroU32, time::Instant};

use highs::{Col, HighsModelStatus, HighsSolutionStatus, RowProblem, Sense};

use crate::Problem;
use crate::preoptimize::{
    PreoptimizeParams, PreoptimizeResult, PreoptimizeStatus, PreoptimizedBlock, evaluate_schedule,
    prepare_preoptimize,
};

#[derive(Clone, Copy)]
struct StartVariable {
    col: Col,
    col_index: usize,
    bay_id: usize,
    entry_time: i64,
}

pub(crate) fn preoptimize_highs(
    problem: &Problem,
    params: PreoptimizeParams,
) -> Result<PreoptimizeResult, String> {
    let build_start = Instant::now();
    let data = prepare_preoptimize(problem, params)?;
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

    let occupancy = &data.occupancy;
    let bay_areas = &data.bay_areas;
    let bay_load_scale = &data.bay_load_scale;
    let pref_penalty = &data.pref_penalty;
    let min_time = data.min_time;
    let horizon = data.horizon;
    let initial = &data.initial;
    let initial_objective = data.initial_objective;
    let initial_z2 = data.initial_z2;
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
    for &bay_area in bay_areas {
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

    let (objective, z1, z2, z3) = evaluate_schedule(problem, pref_penalty, bay_load_scale, &blocks);
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
