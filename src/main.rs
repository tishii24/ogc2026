use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use clap::Parser;
use ogc2026::{Problem, params::SolverParams, solver, utils::time::Timer};

#[derive(Debug, Parser)]
#[command(name = "ogc2026")]
struct Args {
    #[arg(short, long, value_name = "PATH")]
    input: String,
    #[arg(short, long, value_name = "PATH")]
    params: PathBuf,
    #[arg(short, long, default_value_t = 60.0, value_name = "SECONDS")]
    timelimit: f64,
    #[cfg(feature = "anneal-visualizer")]
    #[arg(long, value_name = "DIR")]
    visualize: Option<PathBuf>,
}

fn main() {
    let timer = Timer::start();
    if let Err(err) = run(timer) {
        eprintln!("error: {err}");
        process::exit(1);
    }
    ogc2026::log!("[{:.4}] [main] end.", timer.elapsed_seconds());
    process::exit(0);
}

fn run(timer: Timer) -> Result<(), String> {
    let args = Args::parse();
    #[cfg(feature = "anneal-visualizer")]
    if let Some(dir) = &args.visualize {
        ogc2026::utils::anneal_visualizer::init(dir)?;
    }
    let params = SolverParams::load(&args.params)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(params.runtime.worker_count)
        .build_global()
        .map_err(|err| format!("failed to initialize rayon thread pool: {err}"))?;

    let input = if args.input == "-" {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|err| format!("failed to read stdin: {err}"))?;
        input
    } else {
        fs::read_to_string(&args.input)
            .map_err(|err| format!("failed to read {}: {err}", args.input))?
    };
    let problem: Problem = serde_json::from_str(&input)
        .map_err(|err| format!("failed to parse problem json: {err}"))?;

    solver::solve(&problem, args.timelimit, timer, &params)?;
    Ok(())
}
