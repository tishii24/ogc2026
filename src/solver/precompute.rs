use crate::{Block, Boundsf, Orientation, Problem};

use super::collision::CollisionPrecompute;
use geo::{Area, BooleanOps, Coord, LineString, MultiPolygon, Polygon};
use std::cmp::Reverse;

#[derive(Clone, Copy, Debug)]
pub(crate) struct OrientationNeighbor {
    pub(crate) orient_idx: usize,
    pub(crate) dy: i64,
}

pub(crate) struct Precompute {
    pub(crate) collision: CollisionPrecompute,
    pub(crate) bay_load_scale: Vec<f64>,
    pub(crate) pref_penalty: Vec<Vec<i64>>,
    pub(crate) pref_spread: Vec<i64>,
    pub(crate) bay_order_by_pref: Vec<Vec<usize>>,
    pub(crate) orientation_order_by_bbox: Vec<Vec<usize>>,
    pub(crate) orientation_bbox_center: Vec<Vec<(f64, f64)>>,
    pub(crate) orientation_bbox_bounds: Vec<Vec<Boundsf>>,
    pub(crate) orientation_neighbors: Vec<Vec<Vec<OrientationNeighbor>>>,
    pub(crate) max_footprint_area: Vec<f64>,
}

pub(crate) fn build_bay_load_scale(problem: &Problem) -> Vec<f64> {
    let bay_areas: Vec<_> = problem
        .bays
        .iter()
        .map(|bay| (bay.width * bay.height) as f64)
        .collect();
    let average_area = bay_areas.iter().sum::<f64>() / bay_areas.len() as f64;
    bay_areas
        .into_iter()
        .map(|area| average_area / area)
        .collect()
}

pub(crate) fn build_pref_penalty(problem: &Problem) -> Vec<Vec<i64>> {
    problem
        .blocks
        .iter()
        .map(|block| {
            let max_preference = block.bay_preferences.iter().copied().max().unwrap_or(0);
            block
                .bay_preferences
                .iter()
                .map(|&preference| max_preference - preference)
                .collect()
        })
        .collect()
}

pub(crate) fn build_pref_spread(problem: &Problem) -> Vec<i64> {
    problem
        .blocks
        .iter()
        .map(|block| {
            let min_pref = block.bay_preferences.iter().copied().min().unwrap_or(0);
            let max_pref = block
                .bay_preferences
                .iter()
                .copied()
                .max()
                .unwrap_or(min_pref);
            max_pref - min_pref
        })
        .collect()
}

pub(crate) fn orientation_bounds(orientation: &Orientation) -> Boundsf {
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

fn layer_polygon(layer: &[[f64; 2]]) -> Polygon<f64> {
    let mut coords: Vec<_> = layer.iter().map(|&[x, y]| Coord { x, y }).collect();
    if coords.first() != coords.last() {
        coords.push(coords[0]);
    }
    Polygon::new(LineString::from(coords), vec![])
}

pub(crate) fn orientation_union(orientation: &Orientation) -> Option<MultiPolygon<f64>> {
    let mut polygons = orientation.layers.iter().map(|layer| layer_polygon(layer));
    let first = polygons.next()?;
    let mut union = MultiPolygon(vec![first]);
    for polygon in polygons {
        union = union.union(&polygon);
    }
    Some(union)
}

fn max_footprint_area(block: &Block) -> f64 {
    block
        .shape
        .iter()
        .map(|orientation| {
            orientation_union(orientation)
                .map(|union| union.unsigned_area())
                .unwrap_or(0.0)
        })
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

fn orientation_neighbors_for_block(bboxes: &[Boundsf]) -> Vec<Vec<OrientationNeighbor>> {
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
                .then(b.1.total_cmp(&a.1))
                .then(a.2.cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        candidates.dedup_by_key(|(to_orient, _, _, _)| *to_orient);
        result[from_orient] = candidates
            .into_iter()
            .map(|(orient_idx, _, _, dy)| OrientationNeighbor { orient_idx, dy })
            .collect();
    }
    result
}

fn build_orientation_neighbors(
    orientation_bbox_bounds: &[Vec<Boundsf>],
) -> Vec<Vec<Vec<OrientationNeighbor>>> {
    orientation_bbox_bounds
        .iter()
        .map(|bboxes| orientation_neighbors_for_block(bboxes))
        .collect()
}

impl Precompute {
    pub(crate) fn build(problem: &Problem) -> Self {
        let collision = CollisionPrecompute::build(problem);

        let bay_load_scale = build_bay_load_scale(problem);
        let pref_penalty = build_pref_penalty(problem);
        let pref_spread = build_pref_spread(problem);

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

        let orientation_bbox_bounds: Vec<Vec<Boundsf>> = problem
            .blocks
            .iter()
            .map(|block| block.shape.iter().map(orientation_bounds).collect())
            .collect();
        let orientation_order_by_bbox = orientation_bbox_bounds
            .iter()
            .map(|bounds| {
                let mut order: Vec<usize> = (0..bounds.len()).collect();
                order.sort_by(|&a, &b| {
                    bbox_area(bounds[a])
                        .total_cmp(&bbox_area(bounds[b]))
                        .then(a.cmp(&b))
                });
                order
            })
            .collect();
        let orientation_bbox_center = orientation_bbox_bounds
            .iter()
            .map(|bounds| {
                bounds
                    .iter()
                    .map(|bounds| {
                        (
                            (bounds.min_x + bounds.max_x) * 0.5,
                            (bounds.min_y + bounds.max_y) * 0.5,
                        )
                    })
                    .collect()
            })
            .collect();

        let orientation_neighbors = build_orientation_neighbors(&orientation_bbox_bounds);
        let max_footprint_area: Vec<f64> = problem.blocks.iter().map(max_footprint_area).collect();

        Self {
            collision,
            bay_load_scale,
            pref_penalty,
            pref_spread,
            bay_order_by_pref,
            orientation_order_by_bbox,
            orientation_bbox_center,
            orientation_bbox_bounds,
            orientation_neighbors,
            max_footprint_area,
        }
    }
}
