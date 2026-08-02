use std::{fs, path::Path};

use serde::Deserialize;

use crate::{
    Problem,
    solver::annealing::{AnnealingParams, ReheatParams},
};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverParams {
    pub runtime: RuntimeParams,
    pub phases: PhaseParams,
    pub precompute: PrecomputeParams,
    pub insert: InsertParams,
    pub annealing: AnnealingConfigs,
    pub preoptimize: PreoptimizeSolverParams,
    pub global_neighbor: NeighborParams,
}

impl SolverParams {
    pub fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read_to_string(path)
            .map_err(|err| format!("failed to read params {}: {err}", path.display()))?;
        serde_yaml::from_str(&input)
            .map_err(|err| format!("failed to parse params {}: {err}", path.display()))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeParams {
    pub worker_count: usize,
    pub solver_seed: u64,
    pub preoptimize_seed: u64,
    pub solve_time_buffer_seconds: f64,
    pub solution_emit_min_interval_seconds: f64,
    pub solution_emit_max_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseParams {
    pub initial_preoptimize: LimitedPhaseParams,
    pub initial_build: LimitedPhaseParams,
    pub global_constrained: LimitedPhaseParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitedPhaseParams {
    pub time_ratio: f64,
    pub max_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecomputeParams {
    pub orientation_neighbor_limit: usize,
    pub swap_neighbor_area_top_k: usize,
    pub swap_neighbor_align_delta: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsertParams {
    pub y_buffer: usize,
    pub anchor_randomness: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeSolverParams {
    pub alpha: f64,
    pub beta: f64,
    pub bay_padding: f64,
    pub precedence_margin: i64,
    pub congestion_weight: f64,
    pub initial_build: LimitedPhaseParams,
    pub neighbor_probabilities: PreoptimizeNeighborProbabilities,
    pub neighbor: PreoptimizeNeighborParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeNeighborProbabilities {
    pub relocate: f64,
    pub swap: f64,
    pub large_reconstruct: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeNeighborParams {
    pub min_removed_blocks: usize,
    pub max_removed_blocks: usize,
    pub remove_count_sample_power: f64,
    pub bad_block_sample_count: usize,
    pub bad_block_select_probability: f64,
    pub max_relocate_attempts: usize,
    pub max_time_shift: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightScaleConfig {
    pub w1_scale: f64,
    pub w3_scale: f64,
}

impl WeightScaleConfig {
    fn make(&self, problem: &Problem) -> f64 {
        (self.w1_scale * problem.weights.w1).min(self.w3_scale * problem.weights.w3)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingConfigs {
    pub sample_window: usize,
    pub preoptimize: AnnealingParamsConfig,
    pub global_constrained: AnnealingParamsConfig,
    pub global: AnnealingParamsConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReheatConfig {
    pub stagnation_time_ratio: f64,
    pub worsening_acceptance: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingParamsConfig {
    pub worsening_acceptance: (f64, f64),
    pub exchange_interval: usize,
    pub exchange_threshold: WeightScaleConfig,
    #[serde(default)]
    pub reheat: Option<ReheatConfig>,
    pub tabu_capacity: usize,
}

impl AnnealingParamsConfig {
    pub(crate) fn make(&self, problem: &Problem, sample_window: usize) -> AnnealingParams {
        AnnealingParams {
            sample_window,
            worsening_acceptance: self.worsening_acceptance,
            exchange_interval: self.exchange_interval,
            exchange_threshold: self.exchange_threshold.make(problem),
            reheat: self.reheat.as_ref().map(|reheat| ReheatParams {
                stagnation_time_ratio: reheat.stagnation_time_ratio,
                worsening_acceptance: reheat.worsening_acceptance,
            }),
            tabu_capacity: self.tabu_capacity,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborParams {
    pub probabilities: NeighborProbabilities,
    pub reconstruct: ReconstructNeighborParams,
    pub shift: ShiftNeighborParams,
    #[serde(rename = "move")]
    pub move_block: MoveNeighborParams,
    pub rotate: RotateNeighborParams,
    pub swap: SwapNeighborParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructNeighborParams {
    pub min_removed_blocks: usize,
    pub max_removed_blocks: usize,
    pub remove_pool_factor: usize,
    pub remove_count_sample_power: f64,
    pub remove_seed_per_block: (usize, usize),
    pub remove_entry_base_interval_weights: Vec<f64>,
    pub remove_entry_seed_candidate_count: usize,
    pub remove_seed_method_weights: RemoveSeedMethodWeights,
    pub remove_fluidity_slack_weight_range: (f64, f64),
    pub remove_fluidity_pref_spread_weight_range: (f64, f64),
    pub remove_x_distance_weight_range: (f64, f64),
    pub remove_y_distance_weight_range: (f64, f64),
    pub remove_t_distance_weight_range: (f64, f64),
    pub workload_weight_range: (f64, f64),
    pub volume_weight_range: (f64, f64),
    pub pref_spread_weight_range: (f64, f64),
    pub limit_time_urgency_weight_range: (f64, f64),
    pub slack_tightness_weight_range: (f64, f64),
    pub current_penalty_weight_range: (f64, f64),
    pub order_random_weight_range: (f64, f64),
    pub insert_candidate_top_k: usize,
    pub insert_candidate_select_p: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftNeighborParams {
    pub dy_range: (i64, i64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveNeighborParams {
    pub sample_blocks: usize,
    pub small_pool_size: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotateNeighborParams {
    pub dy_range: (i64, i64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwapNeighborParams {
    pub neighbor_top_k: usize,
    pub dy_range: (i64, i64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoveSeedMethodWeights {
    pub badness: f64,
    pub fluidity: f64,
    pub random: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborProbabilities {
    pub large_reconstruct: f64,
    pub shift: f64,
    #[serde(rename = "move")]
    pub move_block: f64,
    pub rotate: f64,
    pub swap: f64,
}

impl NeighborProbabilities {
    pub(crate) fn weights(&self) -> [f64; 5] {
        [
            self.large_reconstruct,
            self.shift,
            self.move_block,
            self.rotate,
            self.swap,
        ]
    }
}
