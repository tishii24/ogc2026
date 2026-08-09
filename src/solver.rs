use crate::{Problem, Solution, params::SolverParams, utils::time::Timer};

pub(crate) mod annealing;
mod collision;
mod insert;
mod neighbors;
mod objective;
mod optimize;
mod output;
mod placement_scan;
mod precompute;
mod preoptimize;
mod reconstruct;

use optimize::optimize;
use output::{CandidateEmitter, schedule_to_solution};
use precompute::Precompute;
use preoptimize::{PreoptimizePrecompute, preoptimize};

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
    let pre = Precompute::build(problem);
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
    let preoptimized = preoptimize(
        problem,
        &preoptimize_pre,
        &params.preoptimize,
        &params.annealing.preoptimize,
        &params.neighbor.reconstruct,
        preoptimize_time_limit,
        params.runtime.worker_count,
        params.runtime.preoptimize_seed,
    )?;
    log!(
        "[{:.4}] initial abstract score: {:.3}",
        timer.elapsed_seconds(),
        preoptimized.score
    );

    let candidate_emitter = CandidateEmitter::new(
        params.runtime.solution_emit_min_interval_seconds,
        params.runtime.solution_emit_max_count,
    );
    let best = optimize(
        problem,
        &pre,
        &preoptimized,
        deadline,
        timer,
        &params.phases.optimize,
        &params.annealing.optimize,
        &params.neighbor,
        &params.insert,
        &candidate_emitter,
        params.runtime.worker_count,
        params.runtime.solver_seed,
    );
    candidate_emitter.emit(&best, timer, true);
    Ok(schedule_to_solution(&best.schedule))
}
