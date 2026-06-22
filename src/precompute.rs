use crate::{collision::CollisionPrecompute, *};
use std::cmp::Reverse;

pub struct Precompute {
    pub collision: CollisionPrecompute,
    pub bay_load_scale: Vec<f64>,
    pub pref_penalty: Vec<Vec<i64>>,
    pub bay_order_by_pref: Vec<Vec<usize>>,
    pub orientation_order_by_bbox: Vec<Vec<usize>>,
    pub orientation_bbox_center: Vec<Vec<(f64, f64)>>,
    pub block_area: Vec<f64>,
}

fn orientation_bbox(orientation: &Orientation) -> (f64, (f64, f64)) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for layer in &orientation.layers {
        for &[x, y] in layer {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    (
        (max_x - min_x) * (max_y - min_y),
        ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5),
    )
}

fn polygon_area(layer: &[[f64; 2]]) -> f64 {
    let mut area = 0.0;
    for i in 0..layer.len() {
        let [x0, y0] = layer[i];
        let [x1, y1] = layer[(i + 1) % layer.len()];
        area += x0 * y1 - x1 * y0;
    }
    (area * 0.5).abs()
}

fn block_area(block: &Block) -> f64 {
    block
        .shape
        .iter()
        .flat_map(|orientation| orientation.layers.iter())
        .map(|layer| polygon_area(layer))
        .fold(0.0, f64::max)
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

        let orientation_order_by_bbox = problem
            .blocks
            .iter()
            .map(|block| {
                let mut order: Vec<usize> = (0..block.shape.len()).collect();
                order.sort_by(|&a, &b| {
                    orientation_bbox(&block.shape[a])
                        .0
                        .total_cmp(&orientation_bbox(&block.shape[b]).0)
                        .then(a.cmp(&b))
                });
                order
            })
            .collect();

        let orientation_bbox_center = problem
            .blocks
            .iter()
            .map(|block| {
                block
                    .shape
                    .iter()
                    .map(|orientation| orientation_bbox(orientation).1)
                    .collect()
            })
            .collect();

        let block_area = problem.blocks.iter().map(block_area).collect();

        Self {
            collision,
            bay_load_scale,
            pref_penalty,
            bay_order_by_pref,
            orientation_order_by_bbox,
            orientation_bbox_center,
            block_area,
        }
    }
}
