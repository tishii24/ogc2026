mod precompute;
mod solver;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process;

use ahc_library::utils::time;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Deserialize)]
struct Problem {
    bays: Vec<Bay>,
    blocks: Vec<Block>,
}

#[derive(Debug, Deserialize)]
struct Bay {
    width: i64,
    height: i64,
}

#[derive(Debug, Deserialize)]
struct Block {
    release_time: i64,
    due_date: i64,
    processing_time: i64,
    bay_preferences: Vec<i64>,
    shape: Vec<Orientation>,
}

#[derive(Debug, Deserialize)]
struct Orientation {
    layers: Vec<Vec<[f64; 2]>>,
}

#[derive(Debug, Serialize)]
struct Solution {
    operations: BTreeMap<i64, Vec<Operation>>,
}

#[derive(Debug, Serialize)]
struct Operation {
    #[serde(rename = "type")]
    op_type: &'static str,
    block_id: usize,
    bay_id: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    x: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    orient_idx: Option<usize>,
}

#[derive(Debug)]
struct Args {
    input_path: String,
    timelimit: f64,
}

#[derive(Debug, Clone, Copy)]
struct Placement {
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
}

fn main() {
    time::start_clock(1.);
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1).collect())?;
    let input = fs::read_to_string(&args.input_path)
        .map_err(|err| format!("failed to read {}: {err}", args.input_path))?;
    let problem: Problem = serde_json::from_str(&input)
        .map_err(|err| format!("failed to parse problem json: {err}"))?;

    let solution = solver::solve(&problem, args.timelimit)?;
    let output = serde_json::to_string(&solution)
        .map_err(|err| format!("failed to serialize solution json: {err}"))?;
    println!("{output}");
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    if args.is_empty() {
        return Err("usage: ogc2026 <input.json> [timelimit] or ogc2026 --input <input.json> [--timelimit <sec>]".to_string());
    }

    let mut input_path: Option<String> = None;
    let mut timelimit = 60.0;
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                if i >= args.len() {
                    return Err("--input requires a path".to_string());
                }
                input_path = Some(args[i].clone());
            }
            "--timelimit" | "-t" => {
                i += 1;
                if i >= args.len() {
                    return Err("--timelimit requires a number".to_string());
                }
                timelimit = args[i]
                    .parse::<f64>()
                    .map_err(|err| format!("invalid timelimit '{}': {err}", args[i]))?;
            }
            "--help" | "-h" => {
                return Err("usage: ogc2026 <input.json> [timelimit] or ogc2026 --input <input.json> [--timelimit <sec>]".to_string());
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}"));
            }
            _ => positional.push(args[i].clone()),
        }
        i += 1;
    }

    if input_path.is_none() && !positional.is_empty() {
        input_path = Some(positional[0].clone());
    }
    if positional.len() >= 2 {
        timelimit = positional[1]
            .parse::<f64>()
            .map_err(|err| format!("invalid timelimit '{}': {err}", positional[1]))?;
    }
    if positional.len() > 2 {
        return Err(format!(
            "too many positional arguments: {:?}",
            &positional[2..]
        ));
    }

    let input_path = input_path.ok_or_else(|| "missing input path".to_string())?;
    Ok(Args {
        input_path,
        timelimit,
    })
}
