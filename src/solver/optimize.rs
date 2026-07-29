use crate::{
    Problem, ScheduledBlock,
    params::{AnnealingParamsConfig, InsertParams, NeighborParams},
    utils::{random::RandPcg64Mcg, time::Timer},
};

use super::{
    PreoptimizeState,
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    beam_reconstruct::try_beam_large_reconstruct,
    neighbors::{
        NeighborKind, sample_neighbor, try_move_neighbor, try_rotate_neighbor, try_shift_neighbor,
        try_swap_neighbor,
    },
    objective::{ScheduleScore, score_schedule},
    output::CandidateEmitter,
    precompute::Precompute,
    reconstruct::{HeuristicPrecedence, build_heuristic_precedence, try_large_reconstruct},
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

pub(super) struct GlobalAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    precedence: HeuristicPrecedence,
    timer: Timer,
    candidate_emitter: &'a CandidateEmitter,
}

impl<'a> GlobalAnnealing<'a> {
    pub(super) fn new(
        problem: &'a Problem,
        pre: &'a Precompute,
        abstract_state: &PreoptimizeState,
        precedence_margin: i64,
        timer: Timer,
        candidate_emitter: &'a CandidateEmitter,
    ) -> Self {
        Self {
            problem,
            pre,
            precedence: build_heuristic_precedence(problem, abstract_state, precedence_margin),
            timer,
            candidate_emitter,
        }
    }

    pub(super) fn run(
        &self,
        initial: OptimizeState,
        deadline: f64,
        annealing: &AnnealingParamsConfig,
        neighbor_params: &NeighborParams,
        insert_params: &InsertParams,
        constrained: bool,
        seed: u64,
        max_worker_count: usize,
    ) -> OptimizeState {
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        let annealing_params = annealing.make(self.problem, initial.annealing_score());
        let delegate = GlobalAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            precedence: constrained.then_some(&self.precedence),
            name: if constrained { "global-c" } else { "global" },
            initial,
            params: neighbor_params,
            insert_params,
            timer: self.timer,
            candidate_emitter: self.candidate_emitter,
        };
        Annealer::new(deadline, worker_count, seed, annealing_params, delegate).run(self.timer)
    }
}

struct GlobalAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    precedence: Option<&'a HeuristicPrecedence>,
    name: &'static str,
    initial: OptimizeState,
    params: &'a NeighborParams,
    insert_params: &'a InsertParams,
    timer: Timer,
    candidate_emitter: &'a CandidateEmitter,
}

impl AnnealingDelegate for GlobalAnnealingDelegate<'_> {
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
        self.candidate_emitter.emit(state, self.timer, false);
    }

    fn propose(
        &self,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let precedence = self.precedence;
        let params = self.params;
        let probabilities = params.probabilities.weights();
        let neighbor = sample_neighbor(rng, &probabilities);
        let schedule = match neighbor {
            NeighborKind::LargeReconstruct => try_large_reconstruct(
                self.problem,
                self.pre,
                precedence,
                &current.schedule,
                rng,
                accept_threshold,
                &params.reconstruct,
                self.insert_params,
            ),
            NeighborKind::BeamLargeReconstruct => try_beam_large_reconstruct(
                self.problem,
                self.pre,
                precedence,
                &current.schedule,
                rng,
                accept_threshold,
                &params.reconstruct,
            ),
            NeighborKind::Shift => try_shift_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                precedence,
                &params.shift,
            ),
            NeighborKind::Move => try_move_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                precedence,
                &params.move_block,
                self.insert_params,
            ),
            NeighborKind::Rotate => try_rotate_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                precedence,
                &params.rotate,
            ),
            NeighborKind::Swap => try_swap_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                precedence,
                &params.swap,
            ),
        };
        AnnealingAttempt {
            neighbor_kind: neighbor.index(),
            candidate: schedule.map(|schedule| {
                let ScheduleScore {
                    objective,
                    total_tardiness,
                } = score_schedule(self.problem, self.pre, &schedule);
                OptimizeState {
                    objective,
                    total_tardiness,
                    schedule,
                }
            }),
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
