use std::sync::OnceLock;

#[inline]
pub fn local_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OGC_LOCAL")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {{
        if $crate::local_enabled() {
            eprintln!($($arg)*);
        }
    }};
}

pub mod solver;
pub mod utils;

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Deserialize)]
pub struct Problem {
    pub(crate) bays: Vec<Bay>,
    pub(crate) blocks: Vec<Block>,
    #[serde(default)]
    pub(crate) weights: Weights,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub(crate) struct Weights {
    #[serde(default = "default_weight")]
    pub(crate) w1: f64,
    #[serde(default = "default_weight")]
    pub(crate) w2: f64,
    #[serde(default = "default_weight")]
    pub(crate) w3: f64,
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
pub(crate) struct Bay {
    pub(crate) width: i64,
    pub(crate) height: i64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Block {
    pub(crate) release_time: i64,
    pub(crate) due_date: i64,
    pub(crate) processing_time: i64,
    #[serde(default)]
    pub(crate) workload: i64,
    pub(crate) bay_preferences: Vec<i64>,
    pub(crate) shape: Vec<Orientation>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Orientation {
    #[serde(deserialize_with = "deserialize_non_empty_layers")]
    pub(crate) layers: Vec<Vec<[f64; 2]>>,
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
    pub(crate) operations: BTreeMap<i64, Vec<Operation>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Operation {
    #[serde(rename = "type")]
    pub(crate) op_type: &'static str,
    pub(crate) block_id: usize,
    pub(crate) bay_id: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) x: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) y: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) orient_idx: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct ScheduledBlock {
    pub(crate) block_id: usize,
    pub(crate) bay_id: usize,
    pub(crate) orient_idx: usize,
    pub(crate) x: i64,
    pub(crate) y: i64,
    pub(crate) entry_time: i64,
    pub(crate) exit_time: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Boundsi {
    pub(crate) min_x: i64,
    pub(crate) max_x: i64,
    pub(crate) min_y: i64,
    pub(crate) max_y: i64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Boundsf {
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) max_x: f64,
    pub(crate) max_y: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Pointf {
    pub(crate) x: f64,
    pub(crate) y: f64,
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
