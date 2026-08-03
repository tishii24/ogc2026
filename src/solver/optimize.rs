use crate::{
    EPS, Problem, ScheduledBlock,
    params::{AnnealingParamsConfig, InsertParams, NeighborParams},
    utils::{random::RandPcg64Mcg, time::Timer},
};

use super::{
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    neighbors::{
        NeighborKind, sample_neighbor, try_move_neighbor, try_rotate_neighbor, try_shift_neighbor,
        try_swap_neighbor,
    },
    objective::{ScheduleScore, score_schedule},
    output::CandidateEmitter,
    precompute::Precompute,
    reconstruct::try_large_reconstruct,
};

#[derive(Clone, Debug)]
pub(super) struct OptimizeState {
    pub(super) objective: f64,
    pub(super) total_tardiness: i64,
    pub(super) schedule: Vec<ScheduledBlock>,
}

impl AnnealingState for OptimizeState {
    fn annealing_score(&self) -> f64 {
        self.objective
    }

    fn has_tardiness(&self) -> bool {
        self.total_tardiness > 0
    }

    fn tabu_key(&self) -> Option<u64> {
        Some(hash_schedule(&self.schedule))
    }
}

#[derive(Clone, Copy)]
pub(super) struct OptimizeMode {
    pub(super) name: &'static str,
    pub(super) w2: f64,
}

pub(super) fn make_optimize_state(
    problem: &Problem,
    pre: &Precompute,
    schedule: Vec<ScheduledBlock>,
    w2: f64,
) -> OptimizeState {
    let ScheduleScore {
        objective,
        total_tardiness,
    } = score_schedule(problem, pre, &schedule, w2);
    OptimizeState {
        objective,
        total_tardiness,
        schedule,
    }
}

pub(super) struct OptimizeAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    timer: Timer,
}

impl<'a> OptimizeAnnealing<'a> {
    pub(super) fn new(problem: &'a Problem, pre: &'a Precompute, timer: Timer) -> Self {
        Self {
            problem,
            pre,
            timer,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        &self,
        initial: OptimizeState,
        deadline: f64,
        annealing: &AnnealingParamsConfig,
        neighbor_params: &NeighborParams,
        insert_params: &InsertParams,
        mode: OptimizeMode,
        candidate_emitter: Option<&CandidateEmitter>,
        seed: u64,
        max_worker_count: usize,
    ) -> OptimizeState {
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        let annealing_params = annealing.make(
            self.problem,
            initial.annealing_score(),
            initial.schedule.len(),
        );
        let delegate = OptimizeAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            name: mode.name,
            w2: mode.w2,
            initial,
            params: neighbor_params,
            insert_params,
            timer: self.timer,
            candidate_emitter,
        };
        Annealer::new(deadline, worker_count, seed, annealing_params, delegate).run(self.timer)
    }
}

struct OptimizeAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    name: &'static str,
    w2: f64,
    initial: OptimizeState,
    params: &'a NeighborParams,
    insert_params: &'a InsertParams,
    timer: Timer,
    candidate_emitter: Option<&'a CandidateEmitter>,
}

impl AnnealingDelegate for OptimizeAnnealingDelegate<'_> {
    type State = OptimizeState;
    type Output = OptimizeState;

    fn initial_state(&self) -> Self::State {
        self.initial.clone()
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn neighbor_kinds(&self) -> &'static [&'static str] {
        NeighborKind::NAMES
    }

    fn on_shared_best(&self, state: &Self::State, _timer: Timer) {
        if let Some(candidate_emitter) = self.candidate_emitter {
            candidate_emitter.emit(state, self.timer, false);
        }
    }

    fn should_stop(&self, state: &Self::State) -> bool {
        state.objective <= EPS
    }

    #[cfg(feature = "anneal-visualizer")]
    fn visualizer_schedule<'a>(&self, state: &'a Self::State) -> Option<&'a [ScheduledBlock]> {
        self.candidate_emitter.is_some().then_some(&state.schedule)
    }

    #[cfg(feature = "anneal-visualizer")]
    fn visualizer_problem(&self) -> Option<&Problem> {
        self.candidate_emitter.map(|_| self.problem)
    }

    fn propose(
        &self,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let params = self.params;
        let probabilities = params.probabilities.weights();
        let neighbor = sample_neighbor(rng, &probabilities);
        #[cfg(feature = "anneal-visualizer")]
        let mut selected_block_ids = None;
        let schedule = match neighbor {
            NeighborKind::LargeReconstruct => try_large_reconstruct(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                accept_threshold,
                &params.reconstruct,
                self.insert_params,
                self.w2,
            )
            .map(|result| {
                #[cfg(feature = "anneal-visualizer")]
                {
                    selected_block_ids = Some(result.selected_block_ids);
                }
                result.schedule
            }),
            NeighborKind::Shift => try_shift_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                &params.shift,
            ),
            NeighborKind::Move => try_move_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                &params.move_block,
                self.insert_params,
                self.w2,
            ),
            NeighborKind::Rotate => try_rotate_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                &params.rotate,
            ),
            NeighborKind::Swap => {
                try_swap_neighbor(self.problem, self.pre, &current.schedule, rng, &params.swap)
            }
        };
        AnnealingAttempt {
            neighbor_kind: neighbor.index(),
            candidate: schedule
                .map(|schedule| make_optimize_state(self.problem, self.pre, schedule, self.w2)),
            #[cfg(feature = "anneal-visualizer")]
            selected_block_ids,
        }
    }

    fn finish(&self, state: Self::State) -> Self::Output {
        state
    }
}

fn hash_schedule(schedule: &[ScheduledBlock]) -> u64 {
    let mut hash = mix_hash(1469598103934665603, schedule.len() as u64);
    for &block in schedule {
        hash ^= hash_scheduled_block(block);
    }
    hash
}

fn hash_scheduled_block(block: ScheduledBlock) -> u64 {
    let mut hash = 1469598103934665603;
    hash = mix_hash(hash, block.block_id as u64);
    hash = mix_hash(hash, block.bay_id as u64);
    hash = mix_hash(hash, block.orient_idx as u64);
    hash = mix_hash(hash, block.x as u64);
    hash = mix_hash(hash, block.y as u64);
    hash = mix_hash(hash, block.entry_time as u64);
    mix_hash(hash, block.exit_time as u64)
}

fn mix_hash(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash.wrapping_mul(1099511628211)
}
