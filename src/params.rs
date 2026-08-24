use std::{fs, path::Path};

use serde::Deserialize;

use crate::{Problem, solver::annealing::AnnealingParams, utils::random::Random};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverParams {
    pub runtime: RuntimeParams,
    pub phases: PhaseParams,
    pub insert: InsertParams,
    pub annealing: AnnealingConfigs,
    pub preoptimize: PreoptimizeSolverParams,
    pub neighbor: NeighborParams,
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
    pub preoptimize: LimitedPhaseParams,
    pub optimize: OptimizePhaseParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitedPhaseParams {
    pub time_ratio: f64,
    pub max_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizePhaseParams {
    pub base_horizon_size: usize,
    pub time_allocation_power: f64,
    pub horizon_w2_power: Option<f64>,
    pub expand_time_ratio: f64,
    pub max_expand_seconds: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub enum InsertAnchor {
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsertParams {
    pub y_buffer: usize,
    pub anchor_randomness: f64,
    pub candidate_top_k: usize,
    pub candidate_select_p: f64,
    pub worker_primary_anchors: Vec<InsertAnchor>,
}

impl InsertParams {
    pub(crate) fn worker_primary_anchor(&self, worker_id: usize) -> InsertAnchor {
        self.worker_primary_anchors[worker_id % self.worker_primary_anchors.len()]
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeSolverParams {
    pub alpha: f64,
    pub beta: f64,
    pub initial_build: LimitedPhaseParams,
    pub neighbor_probabilities: PreoptimizeNeighborProbabilities,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeNeighborProbabilities {
    pub relocate: f64,
    pub swap: f64,
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
    pub preoptimize: AnnealingParamsConfig,
    pub optimize: AnnealingParamsConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingParamsConfig {
    pub exchange_interval: usize,
    pub temperature_initial_score_per_block_scale: (f64, f64),
    pub exchange_threshold: WeightScaleConfig,
}

impl AnnealingParamsConfig {
    pub(crate) fn make(
        &self,
        problem: &Problem,
        initial_score: f64,
        block_count: usize,
    ) -> AnnealingParams {
        let initial_score_per_block = initial_score / block_count.max(1) as f64;
        AnnealingParams {
            exchange_interval: self.exchange_interval,
            temperature: (
                self.temperature_initial_score_per_block_scale.0 * initial_score_per_block,
                self.temperature_initial_score_per_block_scale.1 * initial_score_per_block,
            ),
            exchange_threshold: self.exchange_threshold.make(problem),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborParams {
    pub probabilities: NeighborProbabilities,
    pub reconstruct: ReconstructNeighborParams,
    pub shift: ShiftNeighborParams,
    pub rotate: RotateNeighborParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructNeighborParams {
    pub remove_count: UsizeRangeDistribution,
    pub remove_pool_factor: usize,
    pub remove_blocks_per_seed: UsizeRangeDistribution,
    pub remove_entry_base_interval_weights: Vec<f64>,
    pub remove_entry_seed_candidate_count: usize,
    pub remove_seed_method_weights: RemoveSeedMethodWeights,
    pub remove_fluidity_slack_weight_range: (f64, f64),
    pub remove_fluidity_pref_spread_weight_range: (f64, f64),
    pub remove_x_distance_weight_range: (f64, f64),
    pub remove_y_distance_weight_range: (f64, f64),
    pub remove_t_distance_weight_range: (f64, f64),
    pub remove_distance_power_range: (f64, f64),
    pub volume_weight_range: (f64, f64),
    pub pref_spread_weight_range: (f64, f64),
    pub limit_time_urgency_weight_range: (f64, f64),
    pub slack_tightness_weight_range: (f64, f64),
    pub current_penalty_weight_range: (f64, f64),
    pub order_random_weight_range: (f64, f64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsizeRangeDistribution {
    pub range: (usize, usize),
    pub power: f64,
}

impl UsizeRangeDistribution {
    pub fn sample(&self, rng: &mut impl Random) -> usize {
        rng.gen_range_lower(self.range.0, self.range.1 + 1, self.power)
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftNeighborParams {
    pub dy_range: (i64, i64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotateNeighborParams {
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
}

impl NeighborProbabilities {
    pub(crate) fn weights(&self) -> [f64; 4] {
        [
            self.large_reconstruct,
            self.shift,
            self.move_block,
            self.rotate,
        ]
    }
}
