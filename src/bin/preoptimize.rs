use std::env;
use std::fs;
use std::io::{self, Read};
use std::process;
use std::time::Instant;

use ogc2026::preoptimize_highs::preoptimize_highs;
use ogc2026::{Problem, preoptimize::PreoptimizeParams, preoptimize::preoptimize_annealing};

const DEFAULT_TIMELIMIT_SECONDS: f64 = 60.0;
const DEFAULT_ALPHA: f64 = 0.0;
const DEFAULT_BETA: f64 = 0.0;
const DEFAULT_HORIZON_MARGIN: i64 = 10;

enum Method {
    Milp,
    Annealing,
}

struct Args {
    input_path: String,
    method: Method,
    time_limit: f64,
    alpha: f64,
    beta: f64,
    horizon_margin: i64,
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
    let params = PreoptimizeParams {
        alpha: args.alpha,
        beta: args.beta,
        time_limit: args.time_limit,
        horizon_margin: args.horizon_margin,
    };
    let result = match args.method {
        Method::Milp => preoptimize_highs(&problem, params),
        Method::Annealing => preoptimize_annealing(&problem, params),
    }?;

    eprintln!(
        "preoptimize: status={:?}, horizon={}, variables={}, constraints={}, elapsed={:.3}s",
        result.status,
        result.horizon,
        result.variable_count,
        result.constraint_count,
        start.elapsed().as_secs_f64()
    );
    let output = serde_json::to_string(&result)
        .map_err(|err| format!("failed to serialize output json: {err}"))?;
    println!("{output}");
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    const USAGE: &str = "usage: preoptimize <input.json|-> <milp|annealing> [timelimit] [alpha] [beta] [horizon-margin]";
    if args.len() < 2 || args.len() > 6 {
        return Err(USAGE.to_string());
    }

    let parse = |index: usize, name: &str, default: f64| -> Result<f64, String> {
        args.get(index).map_or(Ok(default), |value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("invalid {name} '{value}': {err}"))
        })
    };

    let method = match args[1].as_str() {
        "milp" => Method::Milp,
        "annealing" => Method::Annealing,
        value => {
            return Err(format!(
                "invalid method '{value}': expected milp or annealing"
            ));
        }
    };

    Ok(Args {
        input_path: args[0].clone(),
        method,
        time_limit: parse(2, "timelimit", DEFAULT_TIMELIMIT_SECONDS)?,
        alpha: parse(3, "alpha", DEFAULT_ALPHA)?,
        beta: parse(4, "beta", DEFAULT_BETA)?,
        horizon_margin: args.get(5).map_or(Ok(DEFAULT_HORIZON_MARGIN), |value| {
            value
                .parse::<i64>()
                .map_err(|err| format!("invalid horizon-margin '{value}': {err}"))
        })?,
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
