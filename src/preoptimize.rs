/*
use std::num::NonZero;

use highs::{HighsModelStatus, RowProblem, Sense};

use crate::{Problem, precompute::Precompute};

const FLOOR_EPS: f64 = 1e-6;

pub struct BayAssignmentInput<'a> {
    pub problem: &'a Problem,
    pub pre: &'a Precompute,
    pub target_block_ids: &'a [usize],
    pub fixed_loads: &'a [f64],
}

pub struct BayAssignmentResult {
    pub objective: f64,
    pub z2: f64,
    pub z3: f64,
    pub assignment: Vec<usize>,
    pub normalized_load: Vec<f64>,
}

pub trait BayAssignmentOptimizer {
    fn optimize_bay_assignment(
        &mut self,
        input: BayAssignmentInput<'_>,
    ) -> Result<BayAssignmentResult, String>;
}

pub struct HighsBayAssignmentOptimizer;

impl BayAssignmentOptimizer for HighsBayAssignmentOptimizer {
    fn optimize_bay_assignment(
        &mut self,
        input: BayAssignmentInput<'_>,
    ) -> Result<BayAssignmentResult, String> {
        optimize_bay_assignment_highs(input)
    }
}

fn optimize_bay_assignment_highs(
    input: BayAssignmentInput<'_>,
) -> Result<BayAssignmentResult, String> {
    let problem = input.problem;
    let pre = input.pre;
    let target_block_ids = input.target_block_ids;
    let fixed_loads = input.fixed_loads;
    let n = target_block_ids.len();
    let m = problem.bays.len();

    if m == 0 {
        return Err("bays must be non-empty".to_string());
    }
    if fixed_loads.len() != m {
        return Err(format!(
            "fixed_loads length mismatch: got {}, expected {}",
            fixed_loads.len(),
            m
        ));
    }
    for &block_id in target_block_ids {
        let pref_len = problem.blocks[block_id].bay_preferences.len();
        if pref_len != m {
            return Err(format!(
                "block {block_id} has {pref_len} preferences, but there are {m} bays"
            ));
        }
    }

    let mut pb = RowProblem::default();
    let mut x = vec![Vec::with_capacity(m); n];

    for (i, &block_id) in target_block_ids.iter().enumerate() {
        for j in 0..m {
            let obj_coef = problem.weights.w3 * pre.pref_penalty[block_id][j] as f64;
            let col = pb.add_integer_column(obj_coef, 0.0..=1.0);
            x[i].push(col);
        }
    }

    let z2_col = pb.add_integer_column(problem.weights.w2, 0.0..);

    for row in &x {
        let coeffs: Vec<_> = row.iter().map(|&col| (col, 1.0)).collect();
        pb.add_row(1.0..=1.0, &coeffs);
    }

    for j in 0..m {
        for k in (j + 1)..m {
            let mut row_pos = Vec::with_capacity(2 * n + 1);
            let mut row_neg = Vec::with_capacity(2 * n + 1);

            for (i, &block_id) in target_block_ids.iter().enumerate() {
                let workload = problem.blocks[block_id].workload as f64;
                let wj = pre.bay_load_scale[j] * workload;
                let wk = pre.bay_load_scale[k] * workload;

                row_pos.push((x[i][j], wj));
                row_pos.push((x[i][k], -wk));
                row_neg.push((x[i][j], -wj));
                row_neg.push((x[i][k], wk));
            }

            row_pos.push((z2_col, -1.0));
            row_neg.push((z2_col, -1.0));

            let fixed_diff =
                pre.bay_load_scale[j] * fixed_loads[j] - pre.bay_load_scale[k] * fixed_loads[k];
            pb.add_row(..=1.0 - FLOOR_EPS - fixed_diff, &row_pos);
            pb.add_row(..=1.0 - FLOOR_EPS + fixed_diff, &row_neg);
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_threads(NonZero::new(1).unwrap());

    let solved = model.solve();
    let status = solved.status();
    if status != HighsModelStatus::Optimal {
        return Err(format!(
            "HiGHS did not find an optimal solution: {status:?}"
        ));
    }

    let solution = solved.get_solution();
    let mut assignment = vec![0usize; n];

    for i in 0..n {
        let mut best_j = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for j in 0..m {
            let value = solution[x[i][j]];
            if value > best_value {
                best_value = value;
                best_j = j;
            }
        }
        assignment[i] = best_j;
    }

    let mut loads = fixed_loads.to_vec();
    for (&block_id, &bay_id) in target_block_ids.iter().zip(&assignment) {
        loads[bay_id] += problem.blocks[block_id].workload as f64;
    }

    let normalized_load: Vec<f64> = loads
        .iter()
        .enumerate()
        .map(|(bay_id, &load)| pre.bay_load_scale[bay_id] * load)
        .collect();

    let mut z2_raw: f64 = 0.0;
    for j in 0..m {
        for k in (j + 1)..m {
            z2_raw = z2_raw.max((normalized_load[j] - normalized_load[k]).abs());
        }
    }
    let z2 = z2_raw.floor();

    let z3: f64 = target_block_ids
        .iter()
        .zip(&assignment)
        .map(|(&block_id, &bay_id)| pre.pref_penalty[block_id][bay_id] as f64)
        .sum();
    let objective = problem.weights.w2 * z2 + problem.weights.w3 * z3;

    Ok(BayAssignmentResult {
        objective,
        z2,
        z3,
        assignment,
        normalized_load,
    })
}
*/
