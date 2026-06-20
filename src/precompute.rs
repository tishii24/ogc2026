use crate::{collision::CollisionPrecompute, *};
use std::cmp::Reverse;

pub struct Precompute {
    pub collision: CollisionPrecompute,
    pub bay_load_scale: Vec<f64>,
    pub pref_penalty: Vec<Vec<i64>>,
    pub bay_order_by_pref: Vec<Vec<usize>>,
}

impl Precompute {
    pub fn build(problem: &Problem) -> Self {
        let collision = CollisionPrecompute::build(problem);

        let bay_area: Vec<f64> = problem
            .bays
            .iter()
            .map(|bay| (bay.width * bay.height) as f64)
            .collect();
        let avg_area = if bay_area.is_empty() {
            0.0
        } else {
            bay_area.iter().sum::<f64>() / bay_area.len() as f64
        };
        let bay_load_scale = bay_area
            .iter()
            .map(|&area| if area > 0.0 { avg_area / area } else { 0.0 })
            .collect();

        let pref_penalty = problem
            .blocks
            .iter()
            .map(|block| {
                let max_pref = block.bay_preferences.iter().copied().max().unwrap_or(0);
                block
                    .bay_preferences
                    .iter()
                    .map(|&pref| max_pref - pref)
                    .collect()
            })
            .collect();

        let bay_order_by_pref = problem
            .blocks
            .iter()
            .map(|block| {
                let mut order: Vec<usize> = (0..problem.bays.len()).collect();
                order.sort_by_key(|&bay_id| {
                    (
                        Reverse(block.bay_preferences.get(bay_id).copied().unwrap_or(0)),
                        bay_id,
                    )
                });
                order
            })
            .collect();

        Self {
            collision,
            bay_load_scale,
            pref_penalty,
            bay_order_by_pref,
        }
    }
}
