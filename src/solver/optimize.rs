use super::*;
use crate::{
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    params::{AnnealingParamsConfig, GlobalOptimizeParams, InsertParams},
    solver_util::sample_neighbor,
};

#[cfg(feature = "trace-annealing")]
use crate::tracing::{AnnealingTraceDiff, AnnealingTraceState};

impl AnnealingState for OptimizeState {
    fn annealing_score(&self) -> f64 {
        self.score
    }

    fn has_tardiness(&self) -> bool {
        self.z1 > 0
    }

    fn tabu_key(&self) -> Option<u64> {
        Some(hash_schedule(&self.blocks))
    }
}

pub(crate) struct GlobalAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: PrecedenceConstraints,
    timer: Timer,
}

impl<'a> GlobalAnnealing<'a> {
    pub(crate) fn new(
        problem: &'a Problem,
        pre: &'a Precompute,
        abstract_state: &PreoptimizeState,
        precedence_margin: i64,
        timer: Timer,
    ) -> Self {
        Self {
            problem,
            pre,
            constraints: build_precedence_constraints(problem, abstract_state, precedence_margin),
            timer,
        }
    }

    pub(crate) fn run(
        &self,
        initial: OptimizeState,
        deadline: f64,
        annealing: &AnnealingParamsConfig,
        params: &GlobalOptimizeParams,
        neighbor_params: &NeighborParams,
        insert_params: &InsertParams,
        constrained: bool,
        seed: u64,
        max_worker_count: usize,
    ) -> OptimizeState {
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        let delegate = GlobalAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            constraints: constrained.then_some(&self.constraints),
            name: if constrained { "global-c" } else { "global" },
            initial,
            params: neighbor_params,
            insert_params,
        };
        Annealer::new(
            deadline,
            worker_count,
            seed,
            annealing
                .with_override(&params.annealing)
                .make(self.problem),
            delegate,
        )
        .run(self.timer)
    }
}

struct GlobalAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: Option<&'a PrecedenceConstraints>,
    name: &'static str,
    initial: OptimizeState,
    params: &'a NeighborParams,
    insert_params: &'a InsertParams,
}

impl AnnealingDelegate for GlobalAnnealingDelegate<'_> {
    type State = OptimizeState;
    type Output = OptimizeState;

    fn initial_states(&self) -> Vec<Self::State> {
        vec![self.initial.clone()]
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn neighbor_kinds(&self) -> &'static [&'static str] {
        NEIGHBOR_KINDS
    }

    #[cfg(feature = "trace-annealing")]
    fn trace_state(&self, state: &Self::State) -> Option<AnnealingTraceState> {
        Some(AnnealingTraceState {
            score: state.score,
            z1: state.z1,
            state_hash: hash_schedule(&state.blocks),
            bay_hash: hash_bay_assignment(&state.blocks),
        })
    }

    #[cfg(feature = "trace-annealing")]
    fn trace_diff(&self, before: &Self::State, after: &Self::State) -> AnnealingTraceDiff {
        let before_by_id = scheduled_by_id(self.problem, &before.blocks);
        let after_by_id = scheduled_by_id(self.problem, &after.blocks);
        let mut diff = AnnealingTraceDiff::default();
        for block_id in 0..self.problem.blocks.len() {
            let (Some(before), Some(after)) = (before_by_id[block_id], after_by_id[block_id])
            else {
                continue;
            };
            let bay_changed = before.bay_id != after.bay_id;
            let orientation_changed = before.orient_idx != after.orient_idx;
            let position_changed = before.x != after.x || before.y != after.y;
            let time_changed =
                before.entry_time != after.entry_time || before.exit_time != after.exit_time;
            diff.bay_changes += usize::from(bay_changed);
            diff.orientation_changes += usize::from(orientation_changed);
            diff.position_changes += usize::from(position_changed);
            diff.time_changes += usize::from(time_changed);
            diff.changed_blocks +=
                usize::from(bay_changed || orientation_changed || position_changed || time_changed);
        }
        diff
    }

    fn is_finished(&self, _domain: usize, _state: &Self::State) -> bool {
        false
    }

    fn propose(
        &self,
        _domain: usize,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let constraints = self.constraints;
        let params = self.params;
        let probabilities = params.probabilities();
        let neighbor = sample_neighbor(rng, &probabilities);
        let blocks = match neighbor {
            NeighborKind::LargeReconstruct => try_large_reconstruct(
                self.problem,
                self.pre,
                constraints,
                &current.blocks,
                rng,
                accept_threshold,
                params,
                self.insert_params,
            ),
            NeighborKind::Shift => try_shift_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                params,
            ),
            NeighborKind::Move => try_move_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                None,
                params,
                self.insert_params,
            ),
            NeighborKind::Rotate => try_rotate_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                params,
            ),
            NeighborKind::Swap => try_swap_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                constraints.is_some(),
                params,
            ),
        };
        AnnealingAttempt {
            neighbor_kind: neighbor.index(),
            candidate: blocks.map(|blocks| {
                let (score, z1) = score_schedule(self.problem, self.pre, &blocks);
                OptimizeState { score, z1, blocks }
            }),
        }
    }

    fn finish(&self, mut states: Vec<Self::State>) -> Self::Output {
        states.pop().unwrap()
    }
}

#[cfg(feature = "trace-annealing")]
fn hash_bay_assignment(schedule: &[ScheduledBlock]) -> u64 {
    let mut assignments: Vec<_> = schedule
        .iter()
        .map(|block| (block.block_id, block.bay_id))
        .collect();
    assignments.sort_unstable();
    let mut hash = mix_hash(1469598103934665603, assignments.len() as u64);
    for (block_id, bay_id) in assignments {
        hash = mix_hash(hash, block_id as u64);
        hash = mix_hash(hash, bay_id as u64);
    }
    hash
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
