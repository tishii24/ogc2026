use std::{fs, path::Path};

use serde::Deserialize;

use crate::{Problem, annealing::AnnealingParams, solver_util::NeighborKind};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverParamsFile {
    pub schema_version: u32,
    pub runtime: RuntimeParams,
    pub phases: PhaseParams,
    pub precompute: PrecomputeParams,
    pub preoptimize: PreoptimizeSolverParams,
    pub bay_optimize: BayOptimizeParams,
    pub global_optimize: GlobalOptimizeParams,
    pub optimize_neighbor: OptimizeNeighborParams,
}

#[derive(Clone, Debug)]
pub struct SolverParams {
    pub runtime: RuntimeParams,
    pub phases: PhaseParams,
    pub precompute: PrecomputeParams,
    pub preoptimize: PreoptimizeSolverParams,
    pub bay_optimize: BayOptimizeParams,
    pub global_optimize: GlobalOptimizeParams,
    pub bay_neighbor: NeighborParams,
    pub global_neighbor: NeighborParams,
}

impl SolverParams {
    pub fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read_to_string(path)
            .map_err(|err| format!("failed to read params {}: {err}", path.display()))?;
        let file: SolverParamsFile = serde_yaml::from_str(&input)
            .map_err(|err| format!("failed to parse params {}: {err}", path.display()))?;
        file.resolve()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.runtime.worker_count == 0 {
            return Err("runtime.worker_count must be positive".to_string());
        }
        validate_non_negative(
            "runtime.local_search_time_buffer_seconds",
            self.runtime.local_search_time_buffer_seconds,
        )?;
        self.phases.validate()?;
        self.precompute.validate()?;
        self.preoptimize.validate()?;
        self.bay_optimize.validate()?;
        self.global_optimize.validate()?;
        self.bay_neighbor.validate("optimize_neighbor.bay")?;
        self.global_neighbor.validate("optimize_neighbor.global")?;
        Ok(())
    }
}

impl SolverParamsFile {
    fn resolve(self) -> Result<SolverParams, String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported params schema_version: {}",
                self.schema_version
            ));
        }
        let bay_neighbor = self
            .optimize_neighbor
            .default
            .with_override(&self.optimize_neighbor.bay);
        let global_neighbor = self
            .optimize_neighbor
            .default
            .with_override(&self.optimize_neighbor.global);
        let params = SolverParams {
            runtime: self.runtime,
            phases: self.phases,
            precompute: self.precompute,
            preoptimize: self.preoptimize,
            bay_optimize: self.bay_optimize,
            global_optimize: self.global_optimize,
            bay_neighbor,
            global_neighbor,
        };
        params.validate()?;
        Ok(params)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeParams {
    pub worker_count: usize,
    pub solver_seed: u64,
    pub preoptimize_seed: u64,
    pub build_seed_offset: u64,
    pub local_search_time_buffer_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseParams {
    pub initial_preoptimize: LimitedPhaseParams,
    pub initial_build: LimitedPhaseParams,
    pub bay_optimize_time_ratio: f64,
}

impl PhaseParams {
    fn validate(&self) -> Result<(), String> {
        self.initial_preoptimize
            .validate("phases.initial_preoptimize")?;
        self.initial_build.validate("phases.initial_build")?;
        validate_ratio(
            "phases.bay_optimize_time_ratio",
            self.bay_optimize_time_ratio,
        )
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitedPhaseParams {
    pub time_ratio: f64,
    pub max_seconds: f64,
}

impl LimitedPhaseParams {
    fn validate(&self, name: &str) -> Result<(), String> {
        validate_ratio(&format!("{name}.time_ratio"), self.time_ratio)?;
        validate_non_negative(&format!("{name}.max_seconds"), self.max_seconds)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecomputeParams {
    pub orientation_neighbor_limit: usize,
    pub other_block_neighbor_area_top_k: usize,
    pub other_block_neighbor_align_delta: i64,
}

impl PrecomputeParams {
    fn validate(&self) -> Result<(), String> {
        if self.other_block_neighbor_align_delta < 0 {
            return Err(
                "precompute.other_block_neighbor_align_delta must be non-negative".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeSolverParams {
    pub alpha: f64,
    pub beta: f64,
    pub congestion_weight: f64,
    pub neighbor_probabilities: PreoptimizeNeighborProbabilities,
    pub neighbor: PreoptimizeNeighborParams,
    pub annealing: PreoptimizeAnnealingParams,
}

impl PreoptimizeSolverParams {
    fn validate(&self) -> Result<(), String> {
        validate_non_negative("preoptimize.alpha", self.alpha)?;
        validate_non_negative("preoptimize.beta", self.beta)?;
        validate_non_negative("preoptimize.congestion_weight", self.congestion_weight)?;
        self.neighbor_probabilities.validate()?;
        self.neighbor.validate()?;
        self.annealing.validate()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeNeighborProbabilities {
    pub relocate: f64,
    pub swap: f64,
    pub large_reconstruct: f64,
}

impl PreoptimizeNeighborProbabilities {
    fn validate(&self) -> Result<(), String> {
        let values = [self.relocate, self.swap, self.large_reconstruct];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "preoptimize neighbor probabilities must be finite and non-negative".into(),
            );
        }
        if values.iter().sum::<f64>() <= 0.0 {
            return Err("preoptimize neighbor probabilities must have a positive sum".into());
        }
        Ok(())
    }
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

impl PreoptimizeNeighborParams {
    fn validate(&self) -> Result<(), String> {
        if self.min_removed_blocks == 0 || self.min_removed_blocks > self.max_removed_blocks {
            return Err("invalid preoptimize removed block range".into());
        }
        validate_positive(
            "preoptimize.neighbor.remove_count_sample_power",
            self.remove_count_sample_power,
        )?;
        validate_ratio(
            "preoptimize.neighbor.bad_block_select_probability",
            self.bad_block_select_probability,
        )?;
        if self.bad_block_sample_count == 0 || self.max_relocate_attempts == 0 {
            return Err("preoptimize neighbor attempt counts must be positive".into());
        }
        if self.max_time_shift < 0 {
            return Err("preoptimize.neighbor.max_time_shift must be non-negative".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeAnnealingParams {
    pub exchange_interval: usize,
    pub exchange_threshold: f64,
    pub initial_score_per_block_scale: f64,
    pub w1_floor_scale: f64,
    pub w3_floor_scale: f64,
    pub minimum_start_temperature: f64,
    pub end_temperature_ratio: f64,
    pub worker_temperature_scale: f64,
    pub tabu_capacity: usize,
}

impl PreoptimizeAnnealingParams {
    pub fn make(&self, problem: &Problem, initial_score: f64) -> AnnealingParams {
        let start_temperature = (initial_score / problem.blocks.len() as f64
            * self.initial_score_per_block_scale)
            .max(problem.weights.w1 * self.w1_floor_scale)
            .max(problem.weights.w3 * self.w3_floor_scale)
            .max(self.minimum_start_temperature);
        AnnealingParams {
            exchange_interval: self.exchange_interval,
            start_temperature,
            end_temperature: start_temperature * self.end_temperature_ratio,
            worker_temperature_scale: self.worker_temperature_scale,
            tabu_capacity: self.tabu_capacity,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.exchange_interval == 0 {
            return Err("preoptimize.annealing.exchange_interval must be positive".into());
        }
        validate_non_negative(
            "preoptimize.annealing.exchange_threshold",
            self.exchange_threshold,
        )?;
        validate_non_negative(
            "preoptimize.annealing.initial_score_per_block_scale",
            self.initial_score_per_block_scale,
        )?;
        validate_non_negative("preoptimize.annealing.w1_floor_scale", self.w1_floor_scale)?;
        validate_non_negative("preoptimize.annealing.w3_floor_scale", self.w3_floor_scale)?;
        validate_positive(
            "preoptimize.annealing.minimum_start_temperature",
            self.minimum_start_temperature,
        )?;
        validate_positive(
            "preoptimize.annealing.end_temperature_ratio",
            self.end_temperature_ratio,
        )?;
        validate_non_negative(
            "preoptimize.annealing.worker_temperature_scale",
            self.worker_temperature_scale,
        )
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BayOptimizeParams {
    pub exchange_threshold_w1_scale: f64,
    pub annealing: BayAnnealingParams,
}

impl BayOptimizeParams {
    fn validate(&self) -> Result<(), String> {
        validate_non_negative(
            "bay_optimize.exchange_threshold_w1_scale",
            self.exchange_threshold_w1_scale,
        )?;
        self.annealing.validate()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BayAnnealingParams {
    pub exchange_interval: usize,
    pub start_temperature_w1_scale: f64,
    pub end_temperature_w1_scale: f64,
    pub minimum_temperature: f64,
    pub worker_temperature_scale: f64,
    pub tabu_capacity: usize,
}

impl BayAnnealingParams {
    pub fn make(&self, problem: &Problem) -> AnnealingParams {
        AnnealingParams {
            exchange_interval: self.exchange_interval,
            start_temperature: (self.start_temperature_w1_scale * problem.weights.w1)
                .max(self.minimum_temperature),
            end_temperature: (self.end_temperature_w1_scale * problem.weights.w1)
                .max(self.minimum_temperature),
            worker_temperature_scale: self.worker_temperature_scale,
            tabu_capacity: self.tabu_capacity,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.exchange_interval == 0 {
            return Err("bay_optimize.annealing.exchange_interval must be positive".into());
        }
        validate_positive(
            "bay_optimize.annealing.start_temperature_w1_scale",
            self.start_temperature_w1_scale,
        )?;
        validate_positive(
            "bay_optimize.annealing.end_temperature_w1_scale",
            self.end_temperature_w1_scale,
        )?;
        if self.start_temperature_w1_scale < self.end_temperature_w1_scale {
            return Err("bay annealing start temperature scale must be >= end scale".into());
        }
        validate_positive(
            "bay_optimize.annealing.minimum_temperature",
            self.minimum_temperature,
        )?;
        validate_non_negative(
            "bay_optimize.annealing.worker_temperature_scale",
            self.worker_temperature_scale,
        )
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalOptimizeParams {
    pub constraint_time_ratio: f64,
    pub exchange_threshold_w1_scale: f64,
    pub annealing: AnnealingParamsConfig,
}

impl GlobalOptimizeParams {
    fn validate(&self) -> Result<(), String> {
        validate_ratio(
            "global_optimize.constraint_time_ratio",
            self.constraint_time_ratio,
        )?;
        validate_non_negative(
            "global_optimize.exchange_threshold_w1_scale",
            self.exchange_threshold_w1_scale,
        )?;
        self.annealing.validate("global_optimize.annealing")
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingParamsConfig {
    pub exchange_interval: usize,
    pub start_temperature: f64,
    pub end_temperature: f64,
    pub worker_temperature_scale: f64,
    pub tabu_capacity: usize,
}

impl AnnealingParamsConfig {
    pub fn make(&self) -> AnnealingParams {
        AnnealingParams {
            exchange_interval: self.exchange_interval,
            start_temperature: self.start_temperature,
            end_temperature: self.end_temperature,
            worker_temperature_scale: self.worker_temperature_scale,
            tabu_capacity: self.tabu_capacity,
        }
    }

    fn validate(&self, name: &str) -> Result<(), String> {
        if self.exchange_interval == 0 {
            return Err(format!("{name}.exchange_interval must be positive"));
        }
        validate_positive(&format!("{name}.start_temperature"), self.start_temperature)?;
        validate_positive(&format!("{name}.end_temperature"), self.end_temperature)?;
        if self.start_temperature < self.end_temperature {
            return Err(format!(
                "{name}.start_temperature must be >= end_temperature"
            ));
        }
        validate_non_negative(
            &format!("{name}.worker_temperature_scale"),
            self.worker_temperature_scale,
        )
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizeNeighborParams {
    pub default: NeighborParams,
    #[serde(default)]
    pub bay: NeighborParamsOverride,
    #[serde(default)]
    pub global: NeighborParamsOverride,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborParams {
    pub probabilities: NeighborProbabilities,
    pub min_removed_blocks: usize,
    pub max_removed_blocks: usize,
    pub remove_pool_factor: usize,
    pub remove_count_sample_power: f64,
    pub remove_seed_per_block: usize,
    pub remove_random_seed_ratio: f64,
    pub remove_x_distance_weight_max: f64,
    pub remove_y_distance_weight_max: f64,
    pub reconstruct_workload_weight_range: (f64, f64),
    pub reconstruct_volume_weight_range: (f64, f64),
    pub reconstruct_pref_spread_weight_range: (f64, f64),
    pub reconstruct_limit_time_urgency_weight_range: (f64, f64),
    pub reconstruct_order_random_weight_range: (f64, f64),
    pub insert_y_buffer: i64,
    pub shift_max_x: i64,
    pub shift_max_y: i64,
    pub rotate_max_shift_delta: i64,
    pub swap_neighbor_top_k: usize,
    pub swap_max_shift_delta: i64,
    pub move_sample_blocks: usize,
    pub move_small_pool_size: usize,
}

impl NeighborParams {
    pub fn probabilities(&self) -> [(NeighborKind, f64); 5] {
        self.probabilities.weighted()
    }

    fn with_override(&self, value: &NeighborParamsOverride) -> Self {
        Self {
            probabilities: self.probabilities.with_override(&value.probabilities),
            min_removed_blocks: value.min_removed_blocks.unwrap_or(self.min_removed_blocks),
            max_removed_blocks: value.max_removed_blocks.unwrap_or(self.max_removed_blocks),
            remove_pool_factor: value.remove_pool_factor.unwrap_or(self.remove_pool_factor),
            remove_count_sample_power: value
                .remove_count_sample_power
                .unwrap_or(self.remove_count_sample_power),
            remove_seed_per_block: value
                .remove_seed_per_block
                .unwrap_or(self.remove_seed_per_block),
            remove_random_seed_ratio: value
                .remove_random_seed_ratio
                .unwrap_or(self.remove_random_seed_ratio),
            remove_x_distance_weight_max: value
                .remove_x_distance_weight_max
                .unwrap_or(self.remove_x_distance_weight_max),
            remove_y_distance_weight_max: value
                .remove_y_distance_weight_max
                .unwrap_or(self.remove_y_distance_weight_max),
            reconstruct_workload_weight_range: value
                .reconstruct_workload_weight_range
                .unwrap_or(self.reconstruct_workload_weight_range),
            reconstruct_volume_weight_range: value
                .reconstruct_volume_weight_range
                .unwrap_or(self.reconstruct_volume_weight_range),
            reconstruct_pref_spread_weight_range: value
                .reconstruct_pref_spread_weight_range
                .unwrap_or(self.reconstruct_pref_spread_weight_range),
            reconstruct_limit_time_urgency_weight_range: value
                .reconstruct_limit_time_urgency_weight_range
                .unwrap_or(self.reconstruct_limit_time_urgency_weight_range),
            reconstruct_order_random_weight_range: value
                .reconstruct_order_random_weight_range
                .unwrap_or(self.reconstruct_order_random_weight_range),
            insert_y_buffer: value.insert_y_buffer.unwrap_or(self.insert_y_buffer),
            shift_max_x: value.shift_max_x.unwrap_or(self.shift_max_x),
            shift_max_y: value.shift_max_y.unwrap_or(self.shift_max_y),
            rotate_max_shift_delta: value
                .rotate_max_shift_delta
                .unwrap_or(self.rotate_max_shift_delta),
            swap_neighbor_top_k: value
                .swap_neighbor_top_k
                .unwrap_or(self.swap_neighbor_top_k),
            swap_max_shift_delta: value
                .swap_max_shift_delta
                .unwrap_or(self.swap_max_shift_delta),
            move_sample_blocks: value.move_sample_blocks.unwrap_or(self.move_sample_blocks),
            move_small_pool_size: value
                .move_small_pool_size
                .unwrap_or(self.move_small_pool_size),
        }
    }

    fn validate(&self, name: &str) -> Result<(), String> {
        self.probabilities.validate(name)?;
        if self.min_removed_blocks == 0 || self.min_removed_blocks > self.max_removed_blocks {
            return Err(format!("{name}: invalid removed block range"));
        }
        if self.remove_pool_factor == 0
            || self.remove_seed_per_block == 0
            || self.swap_neighbor_top_k == 0
            || self.move_sample_blocks == 0
            || self.move_small_pool_size == 0
        {
            return Err(format!("{name}: count parameters must be positive"));
        }
        validate_positive(
            &format!("{name}.remove_count_sample_power"),
            self.remove_count_sample_power,
        )?;
        validate_ratio(
            &format!("{name}.remove_random_seed_ratio"),
            self.remove_random_seed_ratio,
        )?;
        validate_non_negative(
            &format!("{name}.remove_x_distance_weight_max"),
            self.remove_x_distance_weight_max,
        )?;
        validate_non_negative(
            &format!("{name}.remove_y_distance_weight_max"),
            self.remove_y_distance_weight_max,
        )?;
        validate_range(
            &format!("{name}.reconstruct_workload_weight_range"),
            self.reconstruct_workload_weight_range,
        )?;
        validate_range(
            &format!("{name}.reconstruct_volume_weight_range"),
            self.reconstruct_volume_weight_range,
        )?;
        validate_range(
            &format!("{name}.reconstruct_pref_spread_weight_range"),
            self.reconstruct_pref_spread_weight_range,
        )?;
        validate_range(
            &format!("{name}.reconstruct_limit_time_urgency_weight_range"),
            self.reconstruct_limit_time_urgency_weight_range,
        )?;
        validate_range(
            &format!("{name}.reconstruct_order_random_weight_range"),
            self.reconstruct_order_random_weight_range,
        )?;
        if self.insert_y_buffer < 0
            || self.shift_max_x < 0
            || self.shift_max_y < 0
            || self.rotate_max_shift_delta < 0
            || self.swap_max_shift_delta < 0
        {
            return Err(format!("{name}: spatial deltas must be non-negative"));
        }
        Ok(())
    }
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
    fn weighted(&self) -> [(NeighborKind, f64); 5] {
        [
            (NeighborKind::LargeReconstruct, self.large_reconstruct),
            (NeighborKind::Shift, self.shift),
            (NeighborKind::Move, self.move_block),
            (NeighborKind::Rotate, self.rotate),
            (NeighborKind::Swap, self.swap),
        ]
    }

    fn with_override(&self, value: &NeighborProbabilitiesOverride) -> Self {
        Self {
            large_reconstruct: value.large_reconstruct.unwrap_or(self.large_reconstruct),
            shift: value.shift.unwrap_or(self.shift),
            move_block: value.move_block.unwrap_or(self.move_block),
            rotate: value.rotate.unwrap_or(self.rotate),
            swap: value.swap.unwrap_or(self.swap),
        }
    }

    fn validate(&self, name: &str) -> Result<(), String> {
        let values = [
            self.large_reconstruct,
            self.shift,
            self.move_block,
            self.rotate,
            self.swap,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(format!(
                "{name}.probabilities must be finite and non-negative"
            ));
        }
        if values.iter().sum::<f64>() <= 0.0 {
            return Err(format!("{name}.probabilities must have a positive sum"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeighborParamsOverride {
    pub probabilities: NeighborProbabilitiesOverride,
    pub min_removed_blocks: Option<usize>,
    pub max_removed_blocks: Option<usize>,
    pub remove_pool_factor: Option<usize>,
    pub remove_count_sample_power: Option<f64>,
    pub remove_seed_per_block: Option<usize>,
    pub remove_random_seed_ratio: Option<f64>,
    pub remove_x_distance_weight_max: Option<f64>,
    pub remove_y_distance_weight_max: Option<f64>,
    pub reconstruct_workload_weight_range: Option<(f64, f64)>,
    pub reconstruct_volume_weight_range: Option<(f64, f64)>,
    pub reconstruct_pref_spread_weight_range: Option<(f64, f64)>,
    pub reconstruct_limit_time_urgency_weight_range: Option<(f64, f64)>,
    pub reconstruct_order_random_weight_range: Option<(f64, f64)>,
    pub insert_y_buffer: Option<i64>,
    pub shift_max_x: Option<i64>,
    pub shift_max_y: Option<i64>,
    pub rotate_max_shift_delta: Option<i64>,
    pub swap_neighbor_top_k: Option<usize>,
    pub swap_max_shift_delta: Option<i64>,
    pub move_sample_blocks: Option<usize>,
    pub move_small_pool_size: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeighborProbabilitiesOverride {
    pub large_reconstruct: Option<f64>,
    pub shift: Option<f64>,
    #[serde(rename = "move")]
    pub move_block: Option<f64>,
    pub rotate: Option<f64>,
    pub swap: Option<f64>,
}

fn validate_ratio(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!("{name} must be between 0 and 1"))
    }
}

fn validate_positive(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be positive"))
    }
}

fn validate_non_negative(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be non-negative"))
    }
}

fn validate_range(name: &str, value: (f64, f64)) -> Result<(), String> {
    if value.0.is_finite() && value.1.is_finite() && value.0 <= value.1 {
        Ok(())
    } else {
        Err(format!("{name} must be a finite ordered pair"))
    }
}
