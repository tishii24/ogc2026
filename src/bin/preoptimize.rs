use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process;
use std::time::Instant;

use ogc2026::{
    Problem,
    params::SolverParams,
    preoptimize::{PreoptimizePrecompute, preoptimize},
};

const DEFAULT_TIMELIMIT_SECONDS: f64 = 60.0;

struct Args {
    input_path: String,
    params_path: String,
    time_limit: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1).collect())?;
    let params = SolverParams::load(Path::new(&args.params_path))?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(params.runtime.worker_count)
        .build_global()
        .map_err(|err| format!("failed to initialize rayon thread pool: {err}"))?;
    let input = read_input(&args.input_path)?;
    let problem: Problem = serde_json::from_str(&input)
        .map_err(|err| format!("failed to parse problem json: {err}"))?;
    let start = Instant::now();
    let pre = PreoptimizePrecompute::build(&problem)?;
    let result = preoptimize(
        &problem,
        &pre,
        &params.preoptimize,
        &params.global_neighbor,
        args.time_limit,
        params.runtime.worker_count,
        params.runtime.preoptimize_seed,
    )?;

    eprintln!(
        "preoptimize: objective={:.3}, elapsed={:.3}s",
        result.score,
        start.elapsed().as_secs_f64()
    );
    let output = serde_json::to_string(&result)
        .map_err(|err| format!("failed to serialize output json: {err}"))?;
    println!("{output}");
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    const USAGE: &str = "usage: preoptimize <input.json|-> [timelimit] --params <params.yaml>";
    if args.is_empty() {
        return Err(USAGE.to_string());
    }

    let mut params_path = None;
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--params" | "-p" => {
                i += 1;
                if i >= args.len() {
                    return Err("--params requires a path".to_string());
                }
                params_path = Some(args[i].clone());
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            "-" => positional.push(args[i].clone()),
            other if other.starts_with('-') => return Err(format!("unknown option: {other}")),
            _ => positional.push(args[i].clone()),
        }
        i += 1;
    }
    if positional.is_empty() || positional.len() > 2 {
        return Err(USAGE.to_string());
    }
    let time_limit = positional
        .get(1)
        .map_or(Ok(DEFAULT_TIMELIMIT_SECONDS), |value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("invalid timelimit '{value}': {err}"))
        })?;

    Ok(Args {
        input_path: positional[0].clone(),
        params_path: params_path.ok_or_else(|| "missing --params".to_string())?,
        time_limit,
    })
}

fn read_input(path: &str) -> Result<String, String> {
    if path != "-" {
        return fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"));
    }

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|err| format!("failed to read stdin: {err}"))?;
    Ok(input)
}
