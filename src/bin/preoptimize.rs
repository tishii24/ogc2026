use std::env;
use std::fs;
use std::io::{self, Read};
use std::process;
use std::time::Instant;

use ogc2026::{
    Problem,
    preoptimize::{PreoptimizeParams, PreoptimizePrecompute, preoptimize},
};

const DEFAULT_TIMELIMIT_SECONDS: f64 = 60.0;
const DEFAULT_ALPHA: f64 = 0.0;
const DEFAULT_BETA: f64 = 0.0;
const DISTANCE_WEIGHT: f64 = 0.0;
const RNG_SEED: u64 = 2;
const SWAP_PROBABILITY: f64 = 0.15;
const BAD_BLOCK_SAMPLE_COUNT: usize = 8;
const BAD_BLOCK_SELECT_PROBABILITY: f64 = 0.75;
const MAX_RELOCATE_ATTEMPTS: usize = 8;
const MAX_TIME_SHIFT: i64 = 10;
const END_TEMPERATURE_RATIO: f64 = 1e-4;

struct Args {
    input_path: String,
    time_limit: f64,
    alpha: f64,
    beta: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1).collect())?;
    let input = read_input(&args.input_path)?;
    let problem: Problem = serde_json::from_str(&input)
        .map_err(|err| format!("failed to parse problem json: {err}"))?;
    let start = Instant::now();
    let pre = PreoptimizePrecompute::build(&problem)?;
    let params = PreoptimizeParams {
        alpha: args.alpha,
        beta: args.beta,
        time_limit: args.time_limit,
        distance_weight: DISTANCE_WEIGHT,
        rng_seed: RNG_SEED,
        swap_probability: SWAP_PROBABILITY,
        bad_block_sample_count: BAD_BLOCK_SAMPLE_COUNT,
        bad_block_select_probability: BAD_BLOCK_SELECT_PROBABILITY,
        max_relocate_attempts: MAX_RELOCATE_ATTEMPTS,
        max_time_shift: MAX_TIME_SHIFT,
        end_temperature_ratio: END_TEMPERATURE_RATIO,
    };
    let result = preoptimize(&problem, &pre, None, params)?;

    eprintln!(
        "preoptimize: objective={:.3}, elapsed={:.3}s",
        result.objective,
        start.elapsed().as_secs_f64()
    );
    let output = serde_json::to_string(&result)
        .map_err(|err| format!("failed to serialize output json: {err}"))?;
    println!("{output}");
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    const USAGE: &str = "usage: preoptimize <input.json|-> [timelimit] [alpha] [beta]";
    if args.is_empty() || args.len() > 4 {
        return Err(USAGE.to_string());
    }

    let parse = |index: usize, name: &str, default: f64| -> Result<f64, String> {
        args.get(index).map_or(Ok(default), |value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("invalid {name} '{value}': {err}"))
        })
    };

    Ok(Args {
        input_path: args[0].clone(),
        time_limit: parse(1, "timelimit", DEFAULT_TIMELIMIT_SECONDS)?,
        alpha: parse(2, "alpha", DEFAULT_ALPHA)?,
        beta: parse(3, "beta", DEFAULT_BETA)?,
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
