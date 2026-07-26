use std::env;
use std::eprintln;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process;

use ogc2026::{Problem, solver, utils::base::time::Timer, utils::params::SolverParams};

#[derive(Debug)]
struct Args {
    input_path: String,
    params_path: String,
    timelimit: f64,
}

fn main() {
    let timer = Timer::start(1.);
    if let Err(err) = run(timer) {
        eprintln!("error: {err}");
        process::exit(1);
    }
    eprintln!("[{:.4}] end.", timer.elapsed_seconds());
    process::exit(0);
}

fn run(timer: Timer) -> Result<(), String> {
    let args = parse_args(env::args().skip(1).collect())?;
    let params = SolverParams::load(Path::new(&args.params_path))?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(params.runtime.worker_count)
        .build_global()
        .map_err(|err| format!("failed to initialize rayon thread pool: {err}"))?;

    let input = if args.input_path == "-" {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|err| format!("failed to read stdin: {err}"))?;
        input
    } else {
        fs::read_to_string(&args.input_path)
            .map_err(|err| format!("failed to read {}: {err}", args.input_path))?
    };
    let problem: Problem = serde_json::from_str(&input)
        .map_err(|err| format!("failed to parse problem json: {err}"))?;

    solver::solve(&problem, args.timelimit, timer, &params)?;
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    const USAGE: &str = "usage: ogc2026 <input.json> [timelimit] --params <params.yaml> or ogc2026 --input <input.json> --params <params.yaml> [--timelimit <sec>] [--visualize <dir>]";
    if args.is_empty() {
        return Err(USAGE.to_string());
    }

    let mut input_path: Option<String> = None;
    let mut params_path: Option<String> = None;
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
            "--params" | "-p" => {
                i += 1;
                if i >= args.len() {
                    return Err("--params requires a path".to_string());
                }
                params_path = Some(args[i].clone());
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
            "--visualize" => {
                i += 1;
                if i >= args.len() {
                    return Err("--visualize requires a directory".to_string());
                }
            }
            "--help" | "-h" => {
                return Err(USAGE.to_string());
            }
            "-" => positional.push(args[i].clone()),
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
    let params_path = params_path.ok_or_else(|| "missing --params".to_string())?;
    Ok(Args {
        input_path,
        params_path,
        timelimit,
    })
}
