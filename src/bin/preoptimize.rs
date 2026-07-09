/*
use std::env;
use std::fs;
use std::io::{self, Read};
use std::process;
use std::time::Instant;

use ogc2026::{
    Problem,
    precompute::Precompute,
    preoptimize::{BayAssignmentInput, BayAssignmentOptimizer, HighsBayAssignmentOptimizer},
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Output {
    status: String,
    objective: f64,
    z2: f64,
    z3: f64,
    assignment: Vec<usize>,
    normalized_load: Vec<f64>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let input = read_input()?;
    let problem: Problem = serde_json::from_str(&input)
        .map_err(|err| format!("failed to parse problem json: {err}"))?;
    let start = Instant::now();

    let pre = Precompute::build(&problem);
    let target_block_ids: Vec<usize> = (0..problem.blocks.len()).collect();
    let fixed_loads = vec![0.0; problem.bays.len()];
    let mut optimizer = HighsBayAssignmentOptimizer;
    let result = optimizer.optimize_bay_assignment(BayAssignmentInput {
        problem: &problem,
        pre: &pre,
        target_block_ids: &target_block_ids,
        fixed_loads: &fixed_loads,
    })?;

    let output = Output {
        status: "Optimal".to_string(),
        objective: result.objective,
        z2: result.z2,
        z3: result.z3,
        assignment: result.assignment,
        normalized_load: result.normalized_load,
    };

    let output = serde_json::to_string(&output)
        .map_err(|err| format!("failed to serialize output json: {err}"))?;
    println!("{output}");
    println!("Elapsed: {}ms", start.elapsed().as_millis());
    Ok(())
}

fn read_input() -> Result<String, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] != "-" {
        fs::read_to_string(&args[1]).map_err(|err| format!("failed to read {}: {err}", args[1]))
    } else {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|err| format!("failed to read stdin: {err}"))?;
        Ok(input)
    }
}
*/
fn main() {
    todo!()
}
