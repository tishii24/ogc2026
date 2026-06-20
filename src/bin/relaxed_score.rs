use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use ogc2026::{Problem, relaxed_solver, util::time};

#[derive(Debug)]
struct Args {
    timelimit: f64,
    paths: Vec<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1).collect())?;
    let cases = collect_cases(&args.paths)?;
    if cases.is_empty() {
        return Err("no case files found".to_string());
    }

    println!("case\tscore\tobj1\tobj2\tobj3\telapsed\tstatus");
    for case in cases {
        run_case(&case, args.timelimit);
    }
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    if args.is_empty() {
        return Err("usage: relaxed_score [--timelimit SEC] CASE_OR_DIR...".to_string());
    }

    let mut timelimit = 0.0;
    let mut paths = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
                return Err("usage: relaxed_score [--timelimit SEC] CASE_OR_DIR...".to_string());
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}"));
            }
            path => paths.push(PathBuf::from(path)),
        }
        i += 1;
    }

    if paths.is_empty() {
        return Err("missing CASE_OR_DIR".to_string());
    }
    Ok(Args { timelimit, paths })
}

fn collect_cases(paths: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut cases = Vec::new();
    for path in paths {
        if path.is_file() {
            cases.push(path.clone());
        } else if path.is_dir() {
            for entry in fs::read_dir(path)
                .map_err(|err| format!("failed to read directory {}: {err}", path.display()))?
            {
                let entry = entry.map_err(|err| {
                    format!("failed to read directory entry {}: {err}", path.display())
                })?;
                let child = entry.path();
                if child.extension().is_some_and(|ext| ext == "json") {
                    cases.push(child);
                }
            }
        } else {
            return Err(format!("case path not found: {}", path.display()));
        }
    }
    cases.sort_by(|a, b| natural_case_key(a).cmp(&natural_case_key(b)));
    cases.dedup();
    Ok(cases)
}

fn run_case(case: &Path, timelimit: f64) {
    time::start_clock(1.0);
    let result = (|| -> Result<_, String> {
        let input = fs::read_to_string(case)
            .map_err(|err| format!("failed to read {}: {err}", case.display()))?;
        let problem: Problem = serde_json::from_str(&input)
            .map_err(|err| format!("failed to parse {}: {err}", case.display()))?;
        relaxed_solver::solve_relaxed_score(&problem, timelimit)
    })();
    let elapsed = time::elapsed_seconds();

    match result {
        Ok(result) => println!(
            "{}\t{:.3}\t{}\t{:.3}\t{}\t{:.3}\tok",
            case.display(),
            result.score,
            result.obj1,
            result.obj2,
            result.obj3,
            elapsed
        ),
        Err(err) => println!(
            "{}\tNaN\t0\t0\t0\t{:.3}\terror: {}",
            case.display(),
            elapsed,
            err.replace('\t', " ")
        ),
    }
}

fn natural_case_key(path: &Path) -> (String, usize, String) {
    let parent = path
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_default();
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return (parent, usize::MAX, path.to_string_lossy().to_string());
    };
    let number = stem
        .rsplit_once('_')
        .and_then(|(_, suffix)| suffix.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    (parent, number, stem.to_string())
}
