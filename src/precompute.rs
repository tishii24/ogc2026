use crate::{collision::CollisionPrecompute, *};
use std::cmp::Reverse;

const ORIENTATION_NEIGHBOR_LIMIT: usize = 100;

pub struct Precompute {
    pub collision: CollisionPrecompute,
    pub bay_load_scale: Vec<f64>,
    pub pref_penalty: Vec<Vec<i64>>,
    pub bay_order_by_pref: Vec<Vec<usize>>,
    pub orientation_order_by_bbox: Vec<Vec<usize>>,
    pub orientation_bbox_center: Vec<Vec<(f64, f64)>>,
    pub orientation_bbox_bounds: Vec<Vec<Boundsf>>,
    pub orientation_neighbors: Vec<Vec<Vec<(usize, i64, i64)>>>,
    pub block_area: Vec<f64>,
}

fn orientation_bbox_bounds(orientation: &Orientation) -> Boundsf {
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

    Boundsf {
        min_x,
        min_y,
        max_x,
        max_y,
    }
}

fn orientation_bbox(orientation: &Orientation) -> (f64, (f64, f64)) {
    let bbox = orientation_bbox_bounds(orientation);
    (
        (bbox.max_x - bbox.min_x) * (bbox.max_y - bbox.min_y),
        (
            (bbox.min_x + bbox.max_x) * 0.5,
            (bbox.min_y + bbox.max_y) * 0.5,
        ),
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

fn bbox_area(bbox: Boundsf) -> f64 {
    ((bbox.max_x - bbox.min_x) * (bbox.max_y - bbox.min_y)).max(0.0)
}

fn bbox_iou(from: Boundsf, to: Boundsf, dx: i64, dy: i64) -> f64 {
    let shifted_min_x = to.min_x + dx as f64;
    let shifted_max_x = to.max_x + dx as f64;
    let shifted_min_y = to.min_y + dy as f64;
    let shifted_max_y = to.max_y + dy as f64;
    let overlap_w = from.max_x.min(shifted_max_x) - from.min_x.max(shifted_min_x);
    let overlap_h = from.max_y.min(shifted_max_y) - from.min_y.max(shifted_min_y);
    if overlap_w <= 0.0 || overlap_h <= 0.0 {
        return 0.0;
    }

    let intersection = overlap_w * overlap_h;
    let union = bbox_area(from) + bbox_area(to) - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn orientation_neighbors_for_block(bboxes: &[Boundsf]) -> Vec<Vec<(usize, i64, i64)>> {
    let mut result = vec![Vec::new(); bboxes.len()];
    for from_orient in 0..bboxes.len() {
        let from = bboxes[from_orient];
        let mut candidates = Vec::new();
        for (to_orient, &to) in bboxes.iter().enumerate() {
            if from_orient == to_orient {
                continue;
            }

            let min_dx = (from.min_x - to.max_x).floor() as i64;
            let max_dx = (from.max_x - to.min_x).ceil() as i64;
            let min_dy = (from.min_y - to.max_y).floor() as i64;
            let max_dy = (from.max_y - to.min_y).ceil() as i64;
            for dx in min_dx..=max_dx {
                for dy in min_dy..=max_dy {
                    let iou = bbox_iou(from, to, dx, dy);
                    if iou > 0.0 {
                        candidates.push((to_orient, iou, dx, dy));
                    }
                }
            }
        }

        candidates.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        candidates.dedup_by_key(|(to_orient, _, _, _)| *to_orient);
        candidates.truncate(ORIENTATION_NEIGHBOR_LIMIT);
        result[from_orient] = candidates
            .into_iter()
            .map(|(to_orient, _, dx, dy)| (to_orient, dx, dy))
            .collect();
    }
    result
}

fn build_orientation_neighbors(
    orientation_bbox_bounds: &[Vec<Boundsf>],
) -> Vec<Vec<Vec<(usize, i64, i64)>>> {
    orientation_bbox_bounds
        .iter()
        .map(|bboxes| orientation_neighbors_for_block(bboxes))
        .collect()
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

        let orientation_bbox_bounds: Vec<Vec<Boundsf>> = problem
            .blocks
            .iter()
            .map(|block| block.shape.iter().map(orientation_bbox_bounds).collect())
            .collect();

        let orientation_neighbors = build_orientation_neighbors(&orientation_bbox_bounds);
        let block_area = problem.blocks.iter().map(block_area).collect();

        Self {
            collision,
            bay_load_scale,
            pref_penalty,
            bay_order_by_pref,
            orientation_order_by_bbox,
            orientation_bbox_center,
            orientation_bbox_bounds,
            orientation_neighbors,
            block_area,
        }
    }
}
