pub mod collision;
pub mod precompute;
pub mod solver;
pub mod util;

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
    let layers = Vec::<Vec<[f64; 2]>>::deserialize(deserializer)?;
    Ok(layers
        .into_iter()
        .filter(|layer| !layer.is_empty())
        .collect())
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

#[derive(Clone, Copy, Debug)]
pub struct ScheduledBlock {
    block_id: usize,
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
    entry_time: i64,
    exit_time: i64,
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
