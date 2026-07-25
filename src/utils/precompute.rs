use crate::{utils::collision::CollisionPrecompute, utils::params::PrecomputeParams, *};
use geo::{Area, BooleanOps, Coord, LineString, MultiPolygon, Polygon};
use std::cmp::Reverse;

#[derive(Clone, Copy, Debug)]
pub(crate) struct OtherBlockNeighbor {
    pub(crate) block_id: usize,
    pub(crate) orient_idx: usize,
    pub(crate) dx: i64,
    pub(crate) dy: i64,
}

pub(crate) struct Precompute {
    pub(crate) collision: CollisionPrecompute,
    pub(crate) bay_load_scale: Vec<f64>,
    pub(crate) pref_penalty: Vec<Vec<i64>>,
    pub(crate) bay_order_by_pref: Vec<Vec<usize>>,
    pub(crate) orientation_order_by_bbox: Vec<Vec<usize>>,
    pub(crate) orientation_bbox_center: Vec<Vec<(f64, f64)>>,
    pub(crate) orientation_bbox_bounds: Vec<Vec<Boundsf>>,
    pub(crate) orientation_neighbors: Vec<Vec<Vec<(usize, i64, i64)>>>,
    pub(crate) other_block_neighbors: Vec<Vec<Vec<OtherBlockNeighbor>>>,
    pub(crate) block_area: Vec<f64>,
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

fn block_area(block: &Block) -> f64 {
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

fn orientation_neighbors_for_block(
    bboxes: &[Boundsf],
    limit: usize,
) -> Vec<Vec<(usize, i64, i64)>> {
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
        candidates.truncate(limit);
        result[from_orient] = candidates
            .into_iter()
            .map(|(to_orient, _, dx, dy)| (to_orient, dx, dy))
            .collect();
    }
    result
}

fn build_orientation_neighbors(
    orientation_bbox_bounds: &[Vec<Boundsf>],
    limit: usize,
) -> Vec<Vec<Vec<(usize, i64, i64)>>> {
    orientation_bbox_bounds
        .iter()
        .map(|bboxes| orientation_neighbors_for_block(bboxes, limit))
        .collect()
}

fn area_neighbor_blocks(block_area: &[f64], from_block: usize, top_k: usize) -> Vec<usize> {
    let from_area = block_area[from_block];
    let mut order: Vec<usize> = (0..block_area.len())
        .filter(|&block_id| block_id != from_block)
        .collect();
    order.sort_by(|&a, &b| {
        (from_area - block_area[a])
            .abs()
            .total_cmp(&(from_area - block_area[b]).abs())
            .then(a.cmp(&b))
    });
    order.truncate(top_k.min(order.len()));
    order
}

fn best_bbox_neighbor_offset(
    from: Boundsf,
    to: Boundsf,
    align_delta: i64,
) -> Option<(f64, i64, i64)> {
    let from_cx = (from.min_x + from.max_x) * 0.5;
    let from_cy = (from.min_y + from.max_y) * 0.5;
    let to_cx = (to.min_x + to.max_x) * 0.5;
    let to_cy = (to.min_y + to.max_y) * 0.5;
    let base_dx = (from_cx - to_cx).round() as i64;
    let base_dy = (from_cy - to_cy).round() as i64;

    let mut best: Option<(f64, i64, i64)> = None;
    for dx in base_dx - align_delta..=base_dx + align_delta {
        for dy in base_dy - align_delta..=base_dy + align_delta {
            let iou = bbox_iou(from, to, dx, dy);
            if iou <= 0.0 {
                continue;
            }
            if best.as_ref().map_or(true, |&(best_iou, best_dx, best_dy)| {
                iou.total_cmp(&best_iou)
                    .then(best_dx.cmp(&dx))
                    .then(best_dy.cmp(&dy))
                    .is_gt()
            }) {
                best = Some((iou, dx, dy));
            }
        }
    }
    best
}

fn build_other_block_neighbors(
    orientation_bbox_bounds: &[Vec<Boundsf>],
    block_area: &[f64],
    top_k: usize,
    align_delta: i64,
) -> Vec<Vec<Vec<OtherBlockNeighbor>>> {
    (0..orientation_bbox_bounds.len())
        .map(|from_block| {
            let to_blocks = area_neighbor_blocks(block_area, from_block, top_k);
            orientation_bbox_bounds[from_block]
                .iter()
                .map(|&from_bbox| {
                    let mut candidates = Vec::new();
                    for &to_block in &to_blocks {
                        for (to_orient, &to_bbox) in
                            orientation_bbox_bounds[to_block].iter().enumerate()
                        {
                            if let Some((iou, dx, dy)) =
                                best_bbox_neighbor_offset(from_bbox, to_bbox, align_delta)
                            {
                                candidates.push((
                                    OtherBlockNeighbor {
                                        block_id: to_block,
                                        orient_idx: to_orient,
                                        dx,
                                        dy,
                                    },
                                    iou,
                                ));
                            }
                        }
                    }
                    candidates.sort_by(|a, b| {
                        b.1.total_cmp(&a.1)
                            .then(a.0.block_id.cmp(&b.0.block_id))
                            .then(a.0.orient_idx.cmp(&b.0.orient_idx))
                            .then(a.0.dx.cmp(&b.0.dx))
                            .then(a.0.dy.cmp(&b.0.dy))
                    });
                    candidates
                        .into_iter()
                        .map(|(neighbor, _)| neighbor)
                        .collect()
                })
                .collect()
        })
        .collect()
}

impl Precompute {
    pub(crate) fn build(problem: &Problem, params: &PrecomputeParams) -> Self {
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

        let orientation_neighbors = build_orientation_neighbors(
            &orientation_bbox_bounds,
            params.orientation_neighbor_limit,
        );
        let block_area: Vec<f64> = problem.blocks.iter().map(block_area).collect();
        let other_block_neighbors = build_other_block_neighbors(
            &orientation_bbox_bounds,
            &block_area,
            params.other_block_neighbor_area_top_k,
            params.other_block_neighbor_align_delta,
        );

        Self {
            collision,
            bay_load_scale,
            pref_penalty,
            bay_order_by_pref,
            orientation_order_by_bbox,
            orientation_bbox_center,
            orientation_bbox_bounds,
            orientation_neighbors,
            other_block_neighbors,
            block_area,
        }
    }
}
