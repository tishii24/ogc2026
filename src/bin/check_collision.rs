use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process;

use ogc2026::{
    Problem,
    collision::{BlockPlacement, CollisionPrecompute, CollisionResult},
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Query {
    id: usize,
    moving: QueryPlacement,
    fixed: QueryPlacement,
}

#[derive(Deserialize)]
struct QueryPlacement {
    block_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
}

#[derive(Serialize)]
struct ResultLine {
    id: usize,
    hit: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let problem_path = env::args()
        .nth(1)
        .ok_or_else(|| "usage: check_collision <problem.json>".to_string())?;
    let input = fs::read_to_string(&problem_path)
        .map_err(|err| format!("failed to read {problem_path}: {err}"))?;
    let problem: Problem = serde_json::from_str(&input)
        .map_err(|err| format!("failed to parse problem json: {err}"))?;
    let pre = CollisionPrecompute::build(&problem);

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());

    for line in stdin.lock().lines() {
        let line = line.map_err(|err| format!("failed to read query: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }

        let query: Query = serde_json::from_str(&line)
            .map_err(|err| format!("failed to parse query: {err}; line={line}"))?;
        let moving = BlockPlacement {
            block_id: query.moving.block_id,
            orient_idx: query.moving.orient_idx,
            x: query.moving.x,
            y: query.moving.y,
        };
        let fixed = BlockPlacement {
            block_id: query.fixed.block_id,
            orient_idx: query.fixed.orient_idx,
            x: query.fixed.x,
            y: query.fixed.y,
        };
        let result = ResultLine {
            id: query.id,
            hit: pre.crane(moving, fixed) == CollisionResult::Hit,
        };

        serde_json::to_writer(&mut stdout, &result)
            .map_err(|err| format!("failed to serialize result: {err}"))?;
        writeln!(stdout).map_err(|err| format!("failed to write result: {err}"))?;
        stdout
            .flush()
            .map_err(|err| format!("failed to flush result: {err}"))?;
    }

    Ok(())
}
