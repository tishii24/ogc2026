pub const ENABLE_LOG: bool = true;

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {{
        if $crate::ENABLE_LOG {
            eprintln!($($arg)*);
        }
    }};
}

pub mod annealing;
pub mod collision;
pub mod insert;
pub mod params;
pub mod precompute;
pub mod preoptimize;
pub mod solver;
pub mod solver_util;
pub mod util;
pub mod vis;

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Deserialize)]
pub struct Problem {
    pub bays: Vec<Bay>,
    pub blocks: Vec<Block>,
    #[serde(default)]
    pub weights: Weights,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct Weights {
    #[serde(default = "default_weight")]
    pub w1: f64,
    #[serde(default = "default_weight")]
    pub w2: f64,
    #[serde(default = "default_weight")]
    pub w3: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            w1: 1.0,
            w2: 1.0,
            w3: 1.0,
        }
    }
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct Bay {
    pub width: i64,
    pub height: i64,
}

#[derive(Debug, Deserialize)]
pub struct Block {
    pub release_time: i64,
    pub due_date: i64,
    pub processing_time: i64,
    #[serde(default)]
    pub workload: i64,
    pub bay_preferences: Vec<i64>,
    pub shape: Vec<Orientation>,
}

#[derive(Debug, Deserialize)]
pub struct Orientation {
    #[serde(deserialize_with = "deserialize_non_empty_layers")]
    pub layers: Vec<Vec<[f64; 2]>>,
}

fn deserialize_non_empty_layers<'de, D>(deserializer: D) -> Result<Vec<Vec<[f64; 2]>>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut layers: Vec<Vec<[f64; 2]>> = Vec::<Vec<[f64; 2]>>::deserialize(deserializer)?
        .into_iter()
        .filter(|layer| !layer.is_empty())
        .collect();
    // 空のレイヤーを除去し、最初のレイヤーの頂点を基準に座標を正規化
    if let Some([ref_x, ref_y]) = layers.first().and_then(|layer| layer.first()).copied() {
        for layer in &mut layers {
            for [x, y] in layer {
                *x -= ref_x;
                *y -= ref_y;
            }
        }
    }
    Ok(layers)
}

#[derive(Debug, Serialize)]
pub struct Solution {
    pub operations: BTreeMap<i64, Vec<Operation>>,
}

#[derive(Debug, Serialize)]
pub struct Operation {
    #[serde(rename = "type")]
    pub op_type: &'static str,
    pub block_id: usize,
    pub bay_id: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orient_idx: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub bay_id: usize,
    pub orient_idx: usize,
    pub x: i64,
    pub y: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ScheduledBlock {
    pub block_id: usize,
    pub bay_id: usize,
    pub orient_idx: usize,
    pub x: i64,
    pub y: i64,
    pub entry_time: i64,
    pub exit_time: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boundsi {
    pub min_x: i64,
    pub max_x: i64,
    pub min_y: i64,
    pub max_y: i64,
}

impl Boundsi {
    pub fn contains(&self, x: i64, y: i64) -> bool {
        self.min_x <= x && x <= self.max_x && self.min_y <= y && y <= self.max_y
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Boundsf {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Pointf {
    pub x: f64,
    pub y: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layers_drops_empty_and_uses_first_resolved_vertex_as_reference() {
        let input = r#"
        {
            "bays": [{"width": 10, "height": 10}],
            "blocks": [{
                "release_time": 0,
                "due_date": 1,
                "processing_time": 1,
                "workload": 1,
                "bay_preferences": [100],
                "shape": [{
                    "orientation": 0,
                    "layers": [[], [[2.0, 3.0], [4.0, 3.0], [4.0, 5.0], [2.0, 5.0]]]
                }]
            }]
        }
        "#;
        let problem: Problem = serde_json::from_str(input).unwrap();
        let layers = &problem.blocks[0].shape[0].layers;
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0][0], [0.0, 0.0]);
        assert_eq!(layers[0][2], [2.0, 2.0]);
    }
}
