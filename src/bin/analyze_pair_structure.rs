use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use ogc2026::Problem;
use ogc2026::solver::pair_structure::{
    PairEvaluation, PairPlacedBlock, PairRequest, PairStructurePrecompute,
};
use serde::Deserialize;
use serde::de::IgnoredAny;

const EPS: f64 = 1e-9;

struct Args {
    current_version: String,
    current_timelimit: f64,
    score_csv: String,
    output_prefix: String,
    good_ratio: f64,
}

#[derive(Clone)]
struct ScoreRow {
    timestamp: String,
    version: String,
    case: String,
    timelimit: f64,
    feasible: bool,
    objective: Option<f64>,
    obj1: Option<f64>,
    line_index: usize,
}

#[derive(Deserialize)]
struct ProblemMeta {
    bays: Vec<IgnoredAny>,
    blocks: Vec<BlockMeta>,
}

#[derive(Deserialize)]
struct BlockMeta {
    shape: Vec<IgnoredAny>,
}

#[derive(Deserialize)]
struct SolutionFile {
    operations: BTreeMap<i64, Vec<SolutionOperation>>,
}

#[derive(Deserialize)]
struct SolutionOperation {
    #[serde(rename = "type")]
    op_type: String,
    block_id: usize,
    bay_id: usize,
    x: Option<i64>,
    y: Option<i64>,
    orient_idx: Option<usize>,
}

#[derive(Default)]
struct PartialBlock {
    entry: Option<(i64, PairPlacedBlock)>,
    exit: Option<(i64, usize)>,
}

#[derive(Clone, Copy)]
struct ScheduledBlock {
    placed: PairPlacedBlock,
    entry_time: i64,
    exit_time: i64,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    n_good: usize,
    w_large: f64,
    w_overlap: f64,
}

struct SummaryRecord {
    case: String,
    best_sources: String,
    current_objective: f64,
    best_objective: f64,
    current_tardiness: f64,
    best_tardiness: f64,
    current_metrics: Metrics,
    best_n_good: f64,
    best_w_large: f64,
    best_w_overlap: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1).collect())?;
    let score_text = fs::read_to_string(&args.score_csv)
        .map_err(|err| format!("failed to read {}: {err}", args.score_csv))?;
    let latest = parse_latest_rows(&score_text);
    let mut by_case: HashMap<String, Vec<ScoreRow>> = HashMap::new();
    for row in latest.into_values() {
        by_case.entry(row.case.clone()).or_default().push(row);
    }

    let mut cases: Vec<_> = by_case.keys().cloned().collect();
    cases.sort_by(|a, b| compare_cases(a, b));

    let mut records = Vec::new();
    let mut detail_lines = String::from(
        "case,source_kind,version,timelimit,block_i,block_j,bay_id,orient_i,orient_j,dx,dy,actual_hull_area,area_sum,actual_compactness,best_compactness,best_orient_i,best_orient_j,best_dx,best_dy,good,large_weight,overlap_weight\n",
    );
    let mut skipped = 0usize;

    for case in cases {
        match process_case(
            &case,
            &by_case[&case],
            &args.current_version,
            args.current_timelimit,
            args.good_ratio,
        ) {
            Ok((record, details)) => {
                records.push(record);
                detail_lines.push_str(&details);
            }
            Err(err) => {
                skipped += 1;
                eprintln!("case {case}: {err}; skipped");
            }
        }
    }

    let summary_lines = build_summary_csv(&records, &args);
    let summary_path = PathBuf::from(format!("{}-summary.csv", args.output_prefix));
    let details_path = PathBuf::from(format!("{}-details.csv", args.output_prefix));
    fs::write(&summary_path, summary_lines)
        .map_err(|err| format!("failed to write {}: {err}", summary_path.display()))?;
    fs::write(&details_path, detail_lines)
        .map_err(|err| format!("failed to write {}: {err}", details_path.display()))?;

    print_statistics(&records, skipped);
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    const USAGE: &str = "usage: analyze-pair-structure [--current-version 076] [--current-timelimit 60] [--score-csv log/score.csv] [--output-prefix log/pair-structure] [--good-ratio 1.05]";
    let mut result = Args {
        current_version: "076".to_string(),
        current_timelimit: 60.0,
        score_csv: "log/score.csv".to_string(),
        output_prefix: "log/pair-structure".to_string(),
        good_ratio: 1.05,
    };
    let mut i = 0;
    while i < args.len() {
        let option = args[i].as_str();
        if option == "--help" || option == "-h" {
            return Err(USAGE.to_string());
        }
        if !matches!(
            option,
            "--current-version"
                | "--current-timelimit"
                | "--score-csv"
                | "--output-prefix"
                | "--good-ratio"
        ) {
            return Err(format!("unknown argument: {option}\n{USAGE}"));
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("{option} requires a value"))?;
        match option {
            "--current-version" => result.current_version = value.clone(),
            "--current-timelimit" => {
                result.current_timelimit = parse_finite(value)
                    .ok_or_else(|| format!("invalid current timelimit: {value}"))?;
            }
            "--score-csv" => result.score_csv = value.clone(),
            "--output-prefix" => result.output_prefix = value.clone(),
            "--good-ratio" => {
                result.good_ratio =
                    parse_finite(value).ok_or_else(|| format!("invalid good ratio: {value}"))?;
            }
            _ => unreachable!(),
        }
        i += 2;
    }
    Ok(result)
}

fn parse_finite(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn parse_latest_rows(score_text: &str) -> HashMap<(String, String, u64), ScoreRow> {
    let mut latest = HashMap::new();
    for (line_index, line) in score_text.lines().enumerate().skip(1) {
        let fields: Vec<_> = line.splitn(13, ',').collect();
        if fields.len() < 12 {
            continue;
        }
        let version = fields[1];
        let case = fields[2];
        let Some(timelimit) = parse_finite(fields[3]) else {
            continue;
        };
        if version.is_empty() || case.is_empty() {
            continue;
        }
        let feasible = matches!(
            fields[5].to_ascii_lowercase().as_str(),
            "true" | "1" | "yes"
        );
        let row = ScoreRow {
            timestamp: fields[0].to_string(),
            version: version.to_string(),
            case: case.to_string(),
            timelimit,
            feasible,
            objective: parse_finite(fields[7]),
            obj1: parse_finite(fields[8]),
            line_index,
        };
        let key = (
            row.version.clone(),
            row.case.clone(),
            row.timelimit.to_bits(),
        );
        let replace = latest.get(&key).is_none_or(|current: &ScoreRow| {
            (row.timestamp.as_str(), row.line_index)
                > (current.timestamp.as_str(), current.line_index)
        });
        if replace {
            latest.insert(key, row);
        }
    }
    latest
}

fn process_case(
    case: &str,
    rows: &[ScoreRow],
    current_version: &str,
    current_timelimit: f64,
    good_ratio: f64,
) -> Result<(SummaryRecord, String), String> {
    let current = rows
        .iter()
        .find(|row| {
            row.version == current_version && row.timelimit.to_bits() == current_timelimit.to_bits()
        })
        .filter(|row| row.feasible && row.objective.is_some())
        .ok_or_else(|| {
            format!(
                "usable current row {current_version}/{} is missing",
                format_timelimit(current_timelimit)
            )
        })?;
    let current_objective = current.objective.unwrap();
    let current_tardiness = current
        .obj1
        .ok_or_else(|| "current row has invalid obj1".to_string())?;

    let feasible_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.feasible && row.objective.is_some())
        .collect();
    let best_objective = feasible_rows
        .iter()
        .filter_map(|row| row.objective)
        .min_by(f64::total_cmp)
        .ok_or_else(|| "no feasible row with a valid objective".to_string())?;
    let mut best_rows: Vec<_> = feasible_rows
        .into_iter()
        .filter(|row| row.objective == Some(best_objective))
        .collect();
    best_rows.sort_by(|a, b| {
        a.version
            .cmp(&b.version)
            .then_with(|| a.timelimit.total_cmp(&b.timelimit))
    });

    let mut best_solution_paths = Vec::with_capacity(best_rows.len());
    for row in &best_rows {
        let path = solution_path(row)?;
        if !path.is_file() {
            return Err(format!(
                "best solution is missing for {}/{}: {} (next-best is not used)",
                row.version,
                format_timelimit(row.timelimit),
                path.display()
            ));
        }
        best_solution_paths.push(path);
    }
    let current_solution_path = solution_path(current)?;
    if !current_solution_path.is_file() {
        return Err(format!(
            "current solution is missing: {}",
            current_solution_path.display()
        ));
    }

    let problem_text =
        fs::read_to_string(case).map_err(|err| format!("failed to read problem {case}: {err}"))?;
    let problem: Problem = serde_json::from_str(&problem_text)
        .map_err(|err| format!("failed to parse problem {case}: {err}"))?;
    let meta: ProblemMeta = serde_json::from_str(&problem_text)
        .map_err(|err| format!("failed to parse problem metadata {case}: {err}"))?;

    let current_schedule = load_schedule(&current_solution_path, &meta)?;
    let mut best_schedules = Vec::with_capacity(best_rows.len());
    for path in &best_solution_paths {
        best_schedules.push(load_schedule(path, &meta)?);
    }

    let mut requests = BTreeSet::new();
    collect_pair_requests(&current_schedule, &mut requests);
    for schedule in &best_schedules {
        collect_pair_requests(schedule, &mut requests);
    }
    let requests: Vec<_> = requests.into_iter().collect();
    let precompute = PairStructurePrecompute::build(&problem, &requests)
        .map_err(|err| format!("pair precompute failed: {err}"))?;

    let (current_metrics, mut details) = evaluate_schedule(
        case,
        "current",
        current,
        &current_schedule,
        &precompute,
        good_ratio,
    )?;
    let mut best_metrics = Vec::with_capacity(best_rows.len());
    let mut best_tardiness = Vec::with_capacity(best_rows.len());
    let mut tied_objectives = Vec::with_capacity(best_rows.len());
    for (row, schedule) in best_rows.iter().zip(best_schedules.iter()) {
        let (metrics, lines) =
            evaluate_schedule(case, "best", row, schedule, &precompute, good_ratio)?;
        best_metrics.push(metrics);
        best_tardiness.push(row.obj1.ok_or_else(|| {
            format!(
                "best row {}/{} has invalid obj1",
                row.version,
                format_timelimit(row.timelimit)
            )
        })?);
        tied_objectives.push(row.objective.unwrap());
        details.push_str(&lines);
    }

    let sources = best_rows
        .iter()
        .map(|row| format!("{}/{}", row.version, format_timelimit(row.timelimit)))
        .collect::<Vec<_>>()
        .join("|");
    let best_n_good = median(
        best_metrics
            .iter()
            .map(|metrics| metrics.n_good as f64)
            .collect(),
    );
    let best_w_large = median(best_metrics.iter().map(|metrics| metrics.w_large).collect());
    let best_w_overlap = median(
        best_metrics
            .iter()
            .map(|metrics| metrics.w_overlap)
            .collect(),
    );
    let record = SummaryRecord {
        case: case.to_string(),
        best_sources: sources,
        current_objective,
        best_objective: median(tied_objectives),
        current_tardiness,
        best_tardiness: median(best_tardiness),
        current_metrics,
        best_n_good,
        best_w_large,
        best_w_overlap,
    };

    Ok((record, details))
}

fn solution_path(row: &ScoreRow) -> Result<PathBuf, String> {
    let stem = Path::new(&row.case)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("case path has no valid stem: {}", row.case))?;
    Ok(PathBuf::from("log")
        .join(&row.version)
        .join(format_timelimit(row.timelimit))
        .join(stem)
        .join("solution.json"))
}

fn load_schedule(path: &Path, meta: &ProblemMeta) -> Result<Vec<ScheduledBlock>, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read solution {}: {err}", path.display()))?;
    let solution: SolutionFile = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse solution {}: {err}", path.display()))?;
    let mut partial: Vec<PartialBlock> = (0..meta.blocks.len())
        .map(|_| PartialBlock::default())
        .collect();

    for (time, operations) in solution.operations {
        for operation in operations {
            if operation.block_id >= partial.len() {
                return Err(format!(
                    "solution {} references missing block {}",
                    path.display(),
                    operation.block_id
                ));
            }
            match operation.op_type.as_str() {
                "ENTRY" => {
                    let x = operation
                        .x
                        .ok_or_else(|| format!("solution {} ENTRY has no x", path.display()))?;
                    let y = operation
                        .y
                        .ok_or_else(|| format!("solution {} ENTRY has no y", path.display()))?;
                    let orient_idx = operation.orient_idx.ok_or_else(|| {
                        format!("solution {} ENTRY has no orient_idx", path.display())
                    })?;
                    if operation.bay_id >= meta.bays.len() {
                        return Err(format!(
                            "solution {} block {} has invalid bay {}",
                            path.display(),
                            operation.block_id,
                            operation.bay_id
                        ));
                    }
                    if orient_idx >= meta.blocks[operation.block_id].shape.len() {
                        return Err(format!(
                            "solution {} block {} has invalid orientation {}",
                            path.display(),
                            operation.block_id,
                            orient_idx
                        ));
                    }
                    let block = &mut partial[operation.block_id];
                    if block.entry.is_some() {
                        return Err(format!(
                            "solution {} has duplicate ENTRY for block {}",
                            path.display(),
                            operation.block_id
                        ));
                    }
                    block.entry = Some((
                        time,
                        PairPlacedBlock {
                            block_id: operation.block_id,
                            bay_id: operation.bay_id,
                            orient_idx,
                            x,
                            y,
                        },
                    ));
                }
                "EXIT" => {
                    if operation.x.is_some()
                        || operation.y.is_some()
                        || operation.orient_idx.is_some()
                    {
                        return Err(format!(
                            "solution {} EXIT has placement fields for block {}",
                            path.display(),
                            operation.block_id
                        ));
                    }
                    let block = &mut partial[operation.block_id];
                    if block.exit.is_some() {
                        return Err(format!(
                            "solution {} has duplicate EXIT for block {}",
                            path.display(),
                            operation.block_id
                        ));
                    }
                    block.exit = Some((time, operation.bay_id));
                }
                other => {
                    return Err(format!(
                        "solution {} has unknown operation type {other}",
                        path.display()
                    ));
                }
            }
        }
    }

    partial
        .into_iter()
        .enumerate()
        .map(|(block_id, block)| {
            let (entry_time, placed) = block.entry.ok_or_else(|| {
                format!(
                    "solution {} has no ENTRY for block {block_id}",
                    path.display()
                )
            })?;
            let (exit_time, exit_bay) = block.exit.ok_or_else(|| {
                format!(
                    "solution {} has no EXIT for block {block_id}",
                    path.display()
                )
            })?;
            if placed.bay_id != exit_bay {
                return Err(format!(
                    "solution {} block {block_id} has different ENTRY/EXIT bays",
                    path.display()
                ));
            }
            if entry_time >= exit_time {
                return Err(format!(
                    "solution {} block {block_id} has non-positive occupancy interval",
                    path.display()
                ));
            }
            Ok(ScheduledBlock {
                placed,
                entry_time,
                exit_time,
            })
        })
        .collect()
}

fn collect_pair_requests(schedule: &[ScheduledBlock], requests: &mut BTreeSet<PairRequest>) {
    for first_id in 0..schedule.len() {
        let first = schedule[first_id];
        for second_id in first_id + 1..schedule.len() {
            let second = schedule[second_id];
            if first.placed.bay_id == second.placed.bay_id
                && first.entry_time < second.exit_time
                && second.entry_time < first.exit_time
            {
                requests.insert(PairRequest::new(first_id, second_id, first.placed.bay_id));
            }
        }
    }
}

fn evaluate_schedule(
    case: &str,
    source_kind: &str,
    row: &ScoreRow,
    schedule: &[ScheduledBlock],
    precompute: &PairStructurePrecompute,
    good_ratio: f64,
) -> Result<(Metrics, String), String> {
    let mut metrics = Metrics::default();
    let mut details = String::new();
    for first_id in 0..schedule.len() {
        let first = schedule[first_id];
        for second_id in first_id + 1..schedule.len() {
            let second = schedule[second_id];
            if first.placed.bay_id != second.placed.bay_id
                || first.entry_time >= second.exit_time
                || second.entry_time >= first.exit_time
            {
                continue;
            }
            let evaluation = precompute
                .evaluate(first.placed, second.placed)
                .ok_or_else(|| {
                    format!(
                        "pair evaluation unavailable for {source_kind} {}/{} blocks {first_id},{second_id}",
                        row.version,
                        format_timelimit(row.timelimit)
                    )
                })?;
            let good = evaluation.compactness <= good_ratio * evaluation.best.compactness + EPS;
            let (large_weight, overlap_weight) = if good {
                metrics.n_good += 1;
                let large = evaluation.first_area.min(evaluation.second_area) / evaluation.bay_area;
                let overlap = (evaluation.first_area + evaluation.second_area
                    - evaluation.actual_hull_area)
                    .max(0.0)
                    / evaluation.bay_area;
                metrics.w_large += large;
                metrics.w_overlap += overlap;
                (large, overlap)
            } else {
                (0.0, 0.0)
            };
            append_detail(
                &mut details,
                case,
                source_kind,
                row,
                first,
                second,
                evaluation,
                good,
                large_weight,
                overlap_weight,
            );
        }
    }
    Ok((metrics, details))
}

#[allow(clippy::too_many_arguments)]
fn append_detail(
    output: &mut String,
    case: &str,
    source_kind: &str,
    row: &ScoreRow,
    first: ScheduledBlock,
    second: ScheduledBlock,
    evaluation: PairEvaluation,
    good: bool,
    large_weight: f64,
    overlap_weight: f64,
) {
    let dx = second.placed.x - first.placed.x;
    let dy = second.placed.y - first.placed.y;
    let area_sum = evaluation.first_area + evaluation.second_area;
    let _ = writeln!(
        output,
        "{case},{source_kind},{},{},{},{},{},{},{},{dx},{dy},{},{},{},{},{},{},{},{},{good},{},{}",
        row.version,
        format_timelimit(row.timelimit),
        evaluation.request.first_block_id,
        evaluation.request.second_block_id,
        evaluation.request.bay_id,
        first.placed.orient_idx,
        second.placed.orient_idx,
        format_float(evaluation.actual_hull_area),
        format_float(area_sum),
        format_float(evaluation.compactness),
        format_float(evaluation.best.compactness),
        evaluation.best.first_orient_idx,
        evaluation.best.second_orient_idx,
        evaluation.best.dx,
        evaluation.best.dy,
        format_float(large_weight),
        format_float(overlap_weight),
    );
}

fn build_summary_csv(records: &[SummaryRecord], args: &Args) -> String {
    let mut output = String::from(
        "case,current_version,current_timelimit,best_sources,current_objective,best_objective,score_ratio,current_tardiness,best_tardiness,tardiness_improvement,current_n_good,best_n_good,delta_n_good,current_w_large,best_w_large,delta_w_large,current_w_overlap,best_w_overlap,delta_w_overlap\n",
    );
    for record in records {
        let score_ratio = record.current_objective / record.best_objective;
        let tardiness_improvement = record.current_tardiness - record.best_tardiness;
        let delta_n_good = record.best_n_good - record.current_metrics.n_good as f64;
        let delta_w_large = record.best_w_large - record.current_metrics.w_large;
        let delta_w_overlap = record.best_w_overlap - record.current_metrics.w_overlap;
        let _ = writeln!(
            output,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            record.case,
            args.current_version,
            format_timelimit(args.current_timelimit),
            record.best_sources,
            format_float(record.current_objective),
            format_float(record.best_objective),
            format_float(score_ratio),
            format_float(record.current_tardiness),
            format_float(record.best_tardiness),
            format_float(tardiness_improvement),
            record.current_metrics.n_good,
            format_float(record.best_n_good),
            format_float(delta_n_good),
            format_float(record.current_metrics.w_large),
            format_float(record.best_w_large),
            format_float(delta_w_large),
            format_float(record.current_metrics.w_overlap),
            format_float(record.best_w_overlap),
            format_float(delta_w_overlap),
        );
    }
    output
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn print_statistics(records: &[SummaryRecord], skipped: usize) {
    println!(
        "processed cases: {}; skipped cases: {}",
        records.len(),
        skipped
    );
    let all: Vec<_> = records.iter().collect();
    let t_zero: Vec<_> = records
        .iter()
        .filter(|record| record.current_tardiness <= 0.0 && record.best_tardiness <= 0.0)
        .collect();
    let t_positive: Vec<_> = records
        .iter()
        .filter(|record| record.current_tardiness > 0.0 || record.best_tardiness > 0.0)
        .collect();

    for (label, group) in [("all", all), ("T=0", t_zero), ("T>0", t_positive.clone())] {
        let n_good_better = group
            .iter()
            .filter(|record| record.best_n_good > record.current_metrics.n_good as f64)
            .count();
        let w_large_better = group
            .iter()
            .filter(|record| record.best_w_large > record.current_metrics.w_large)
            .count();
        let w_overlap_better = group
            .iter()
            .filter(|record| record.best_w_overlap > record.current_metrics.w_overlap)
            .count();
        println!(
            "{label}: best>current N_good={n_good_better}/{}, W_large={w_large_better}/{}, W_overlap={w_overlap_better}/{}",
            group.len(),
            group.len(),
            group.len()
        );
        println!(
            "{label}: Spearman score_improvement vs delta_N_good={}, delta_W_large={}, delta_W_overlap={}",
            correlation(
                &group,
                |record| record.best_n_good - record.current_metrics.n_good as f64,
                false
            ),
            correlation(
                &group,
                |record| record.best_w_large - record.current_metrics.w_large,
                false
            ),
            correlation(
                &group,
                |record| record.best_w_overlap - record.current_metrics.w_overlap,
                false
            ),
        );
    }
    println!(
        "T>0: Spearman tardiness_improvement vs delta_N_good={}, delta_W_large={}, delta_W_overlap={}",
        correlation(
            &t_positive,
            |record| record.best_n_good - record.current_metrics.n_good as f64,
            true
        ),
        correlation(
            &t_positive,
            |record| record.best_w_large - record.current_metrics.w_large,
            true
        ),
        correlation(
            &t_positive,
            |record| record.best_w_overlap - record.current_metrics.w_overlap,
            true
        ),
    );
}

fn correlation(
    records: &[&SummaryRecord],
    metric_delta: impl Fn(&SummaryRecord) -> f64,
    tardiness: bool,
) -> String {
    let pairs: Vec<_> = records
        .iter()
        .filter_map(|record| {
            let improvement = if tardiness {
                record.current_tardiness - record.best_tardiness
            } else {
                record.current_objective / record.best_objective - 1.0
            };
            let delta = metric_delta(record);
            (!improvement.is_nan() && !delta.is_nan()).then_some((improvement, delta))
        })
        .collect();
    let (x, y): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    spearman(&x, &y)
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "None".to_string())
}

fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let rank_x = average_ranks(x);
    let rank_y = average_ranks(y);
    let mean_x = rank_x.iter().sum::<f64>() / rank_x.len() as f64;
    let mean_y = rank_y.iter().sum::<f64>() / rank_y.len() as f64;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (&x, &y) in rank_x.iter().zip(&rank_y) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if variance_x == 0.0 || variance_y == 0.0 {
        None
    } else {
        Some(covariance / (variance_x * variance_y).sqrt())
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut first = 0;
    while first < indices.len() {
        let mut end = first + 1;
        while end < indices.len() && values[indices[first]] == values[indices[end]] {
            end += 1;
        }
        let rank = (first + 1 + end) as f64 / 2.0;
        for &index in &indices[first..end] {
            ranks[index] = rank;
        }
        first = end;
    }
    ranks
}

fn compare_cases(a: &str, b: &str) -> Ordering {
    match (case_number(a), case_number(b)) {
        (Some(a_number), Some(b_number)) => a_number.cmp(&b_number).then_with(|| a.cmp(b)),
        _ => a.cmp(b),
    }
}

fn case_number(case: &str) -> Option<u64> {
    Path::new(case)
        .file_stem()?
        .to_str()?
        .strip_prefix("prob_")?
        .parse()
        .ok()
}

fn format_timelimit(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn format_float(value: f64) -> String {
    format!("{value:.17}")
}
