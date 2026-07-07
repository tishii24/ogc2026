use highs::{HighsModelStatus, RowProblem, Sense};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{self, Read};
use std::num::NonZero;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Problem {
    bays: Vec<Bay>,
    blocks: Vec<Block>,
    weights: Weights,
}

#[derive(Debug, Deserialize)]
struct Bay {
    width: f64,
    height: f64,
}

#[derive(Debug, Deserialize)]
struct Block {
    workload: f64,
    bay_preferences: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Weights {
    w2: f64,
    w3: f64,
}

#[derive(Debug, Serialize)]
struct Output {
    status: String,
    objective: f64,
    z2: f64,
    z3: f64,
    assignment: Vec<usize>,
    normalized_load: Vec<f64>,
}

fn read_input() -> Result<String, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() >= 2 {
        Ok(fs::read_to_string(&args[1])?)
    } else {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        Ok(input)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = read_input()?;
    let prob: Problem = serde_json::from_str(&input)?;

    let start = Instant::now();

    let n = prob.blocks.len();
    let m = prob.bays.len();

    if n == 0 || m == 0 {
        return Err("blocks and bays must be non-empty".into());
    }

    for (i, block) in prob.blocks.iter().enumerate() {
        if block.bay_preferences.len() != m {
            return Err(format!(
                "block {} has {} preferences, but there are {} bays",
                i,
                block.bay_preferences.len(),
                m
            )
            .into());
        }
    }

    // u_j = average bay area / area_j
    let areas: Vec<f64> = prob.bays.iter().map(|b| b.width * b.height).collect();

    let avg_area = areas.iter().sum::<f64>() / m as f64;

    let u: Vec<f64> = areas.iter().map(|&a| avg_area / a).collect();

    // c_ij = max_j S_ij - S_ij
    let pref_penalty: Vec<Vec<f64>> = prob
        .blocks
        .iter()
        .map(|block| {
            let max_pref = block
                .bay_preferences
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            block
                .bay_preferences
                .iter()
                .map(|&s| max_pref - s)
                .collect()
        })
        .collect();

    // RowProblem:
    // 先に変数を追加し、その後に制約行を追加する形式。
    let mut pb = RowProblem::default();

    // x[i][j] ∈ {0,1}
    //
    // objective coefficient:
    //   w3 * c_ij
    let mut x = vec![Vec::with_capacity(m); n];

    for i in 0..n {
        for j in 0..m {
            let obj_coef = prob.weights.w3 * pref_penalty[i][j];

            let col = pb.add_integer_column(obj_coef, 0.0..=1.0);

            x[i].push(col);
        }
    }

    // z2 >= 0
    //
    // objective coefficient:
    //   w2
    let z2_col = pb.add_column(prob.weights.w2, 0.0..);

    // 各 block はちょうど 1 bay に割り当てる
    //
    // Σ_j x[i,j] = 1
    for i in 0..n {
        let row: Vec<_> = (0..m).map(|j| (x[i][j], 1.0)).collect();

        pb.add_row(1.0..=1.0, &row);
    }

    // z2 >= |A_j - A_k|
    //
    // A_j = u_j * Σ_i workload_i * x_ij
    //
    // A_j - A_k - z2 <= 0
    // A_k - A_j - z2 <= 0
    for j in 0..m {
        for k in (j + 1)..m {
            let mut row_pos = Vec::with_capacity(2 * n + 1);
            let mut row_neg = Vec::with_capacity(2 * n + 1);

            for i in 0..n {
                let wj = u[j] * prob.blocks[i].workload;
                let wk = u[k] * prob.blocks[i].workload;

                // A_j - A_k - z2 <= 0
                row_pos.push((x[i][j], wj));
                row_pos.push((x[i][k], -wk));

                // A_k - A_j - z2 <= 0
                row_neg.push((x[i][j], -wj));
                row_neg.push((x[i][k], wk));
            }

            row_pos.push((z2_col, -1.0));
            row_neg.push((z2_col, -1.0));

            pb.add_row(..=0.0, &row_pos);
            pb.add_row(..=0.0, &row_neg);
        }
    }

    // 必要なら 4 core に制限。
    //
    // highs crate の Model API では set_threads が提供されています。
    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_threads(NonZero::new(1).unwrap());

    let solved = model.solve();

    let status = solved.status();

    if status != HighsModelStatus::Optimal {
        return Err(format!("HiGHS did not find an optimal solution: {status:?}").into());
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

    let normalized_load: Vec<f64> = (0..m)
        .map(|j| {
            u[j] * (0..n)
                .filter(|&i| assignment[i] == j)
                .map(|i| prob.blocks[i].workload)
                .sum::<f64>()
        })
        .collect();

    let z2 = solution[z2_col];

    let z3: f64 = (0..n).map(|i| pref_penalty[i][assignment[i]]).sum();

    let objective = prob.weights.w2 * z2 + prob.weights.w3 * z3;

    let output = Output {
        status: format!("{status:?}"),
        objective,
        z2,
        z3,
        assignment,
        normalized_load,
    };

    println!("{}", serde_json::to_string(&output)?);

    println!("Elapsed: {}ms", start.elapsed().as_millis());

    println!("diff: {}", prob.weights.w2 * (z2 - z2.floor()));

    Ok(())
}
