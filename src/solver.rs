use crate::{Problem, Solution, params::SolverParams, utils::time::Timer};

pub(crate) mod annealing;
mod collision;
mod insert;
mod neighbors;
mod objective;
mod optimize;
mod output;
pub mod pair_structure;
mod placement_scan;
mod precompute;
mod preoptimize;
mod reconstruct;

use optimize::GlobalAnnealing;
use output::{CandidateEmitter, schedule_to_solution};
use precompute::Precompute;
use preoptimize::{PreoptimizePrecompute, preoptimize};
use reconstruct::build_optimize_state;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct PreoptimizedBlock {
    pub(crate) bay_id: usize,
    pub(crate) entry_time: i64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PreoptimizeState {
    pub score: f64,
    pub(crate) blocks: Vec<PreoptimizedBlock>,
}

fn phase_time_limit(timelimit: f64, ratio: f64, max_seconds: f64) -> f64 {
    (timelimit * ratio).min(max_seconds).max(1e-4)
}

pub fn solve(
    problem: &Problem,
    timelimit: f64,
    timer: Timer,
    params: &SolverParams,
) -> Result<Solution, String> {
    let deadline = timelimit - params.runtime.solve_time_buffer_seconds;

    log!("[{:.4}] building precompute...", timer.elapsed_seconds());
    let pre = Precompute::build(problem, &params.precompute);
    log!("[{:.4}] precompute built", timer.elapsed_seconds());

    log!(
        "[{:.4}] building preoptimize precompute...",
        timer.elapsed_seconds()
    );
    let preoptimize_pre = PreoptimizePrecompute::build(problem)?;
    log!(
        "[{:.4}] preoptimize precompute built",
        timer.elapsed_seconds()
    );
    let preoptimize_time_limit = phase_time_limit(
        timelimit,
        params.phases.initial_preoptimize.time_ratio,
        params.phases.initial_preoptimize.max_seconds,
    )
    .min((deadline - timer.elapsed_seconds()).max(1e-4));
    let initial_abstract = preoptimize(
        problem,
        &preoptimize_pre,
        &params.preoptimize,
        &params.annealing.preoptimize,
        params.annealing.sample_window,
        params.annealing.worsening_delta_quantile,
        &params.global_neighbor.reconstruct,
        preoptimize_time_limit,
        params.runtime.worker_count,
        params.runtime.preoptimize_seed,
    )?;
    log!(
        "[{:.4}] initial abstract score: {:.3}",
        timer.elapsed_seconds(),
        initial_abstract.score
    );
    let build_time_limit = phase_time_limit(
        timelimit,
        params.phases.initial_build.time_ratio,
        params.phases.initial_build.max_seconds,
    )
    .min((deadline - timer.elapsed_seconds()).max(1e-4));
    let initial = build_optimize_state(
        problem,
        &pre,
        &initial_abstract,
        build_time_limit,
        timer,
        params.runtime.worker_count,
        params.runtime.solver_seed,
        &params.global_neighbor.reconstruct,
        &params.insert,
        params.preoptimize.precedence_margin,
    )
    .ok_or_else(|| "failed to build initial optimize state".to_string())?;
    log!(
        "[{:.4}] initial optimize score: {:.3}",
        timer.elapsed_seconds(),
        initial.objective
    );

    let global_start = timer.elapsed_seconds();
    let global_annealing_time = (deadline - global_start).max(0.0);
    let interval_seconds = params
        .runtime
        .solution_emit_min_interval_seconds
        .max(global_annealing_time / params.runtime.solution_emit_max_count as f64);
    log!(
        "[{:.4}] [candidate-emit] interval={:.3}s, global_time={:.3}s, max_count={}",
        timer.elapsed_seconds(),
        interval_seconds,
        global_annealing_time,
        params.runtime.solution_emit_max_count,
    );
    let candidate_emitter = CandidateEmitter::new(interval_seconds);
    candidate_emitter.emit(&initial, timer, true);

    let global = GlobalAnnealing::new(
        problem,
        &pre,
        &initial_abstract,
        params.preoptimize.precedence_margin,
        timer,
        &candidate_emitter,
    );
    let constrained_time_limit = phase_time_limit(
        (deadline - global_start).max(0.0),
        params.phases.global_constrained.time_ratio,
        params.phases.global_constrained.max_seconds,
    );
    let constrained_deadline = global_start + constrained_time_limit;
    let initial = global.run(
        initial,
        constrained_deadline,
        &params.annealing.global_constrained,
        params.annealing.sample_window,
        params.annealing.worsening_delta_quantile,
        &params.global_neighbor,
        &params.insert,
        true,
        params.runtime.solver_seed,
        params.runtime.worker_count,
    );
    log!(
        "[{:.4}] constrained global annealing score: {:.3}",
        timer.elapsed_seconds(),
        initial.objective
    );
    candidate_emitter.emit(&initial, timer, true);

    let best = global.run(
        initial,
        deadline,
        &params.annealing.global,
        params.annealing.sample_window,
        params.annealing.worsening_delta_quantile,
        &params.global_neighbor,
        &params.insert,
        false,
        params.runtime.solver_seed.wrapping_add(1 << 32),
        params.runtime.worker_count,
    );
    candidate_emitter.emit(&best, timer, true);
    Ok(schedule_to_solution(&best.schedule))
}
