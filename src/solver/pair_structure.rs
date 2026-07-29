use std::collections::{BTreeSet, HashMap};

use geo::{Area, ConvexHull, MultiPolygon, Polygon, Translate};
use rayon::prelude::*;

use crate::Problem;

use super::{
    collision::{BlockOrient, CollisionPrecompute},
    precompute::orientation_union,
};

const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PairRequest {
    pub first_block_id: usize,
    pub second_block_id: usize,
    pub bay_id: usize,
}

impl PairRequest {
    pub fn new(first_block_id: usize, second_block_id: usize, bay_id: usize) -> Self {
        assert_ne!(first_block_id, second_block_id);
        let (first_block_id, second_block_id) = if first_block_id < second_block_id {
            (first_block_id, second_block_id)
        } else {
            (second_block_id, first_block_id)
        };
        Self {
            first_block_id,
            second_block_id,
            bay_id,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PairPlacedBlock {
    pub block_id: usize,
    pub bay_id: usize,
    pub orient_idx: usize,
    pub x: i64,
    pub y: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct PairPlacement {
    pub first_orient_idx: usize,
    pub second_orient_idx: usize,
    pub dx: i64,
    pub dy: i64,
    pub hull_area: f64,
    pub area_sum: f64,
    pub compactness: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PairEvaluation {
    pub request: PairRequest,
    pub actual_hull_area: f64,
    pub first_area: f64,
    pub second_area: f64,
    pub compactness: f64,
    pub bay_area: f64,
    pub best: PairPlacement,
}

struct OrientationProjection {
    area: f64,
    hull: Polygon<f64>,
}

pub struct PairStructurePrecompute {
    projections: Vec<Vec<OrientationProjection>>,
    bay_areas: Vec<f64>,
    best: HashMap<PairRequest, PairPlacement>,
}

impl PairStructurePrecompute {
    pub fn build(problem: &Problem, requests: &[PairRequest]) -> Result<Self, String> {
        let collision = CollisionPrecompute::build(problem);
        Self::build_with_collision(problem, &collision, requests)
    }

    pub(crate) fn build_with_collision(
        problem: &Problem,
        collision: &CollisionPrecompute,
        requests: &[PairRequest],
    ) -> Result<Self, String> {
        let projections: Vec<Vec<_>> = problem
            .blocks
            .iter()
            .enumerate()
            .map(|(block_id, block)| {
                block
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(orient_idx, orientation)| {
                        let union = orientation_union(orientation).ok_or_else(|| {
                            format!("block {block_id} orientation {orient_idx} has no layers")
                        })?;
                        let area = union.unsigned_area();
                        if area <= EPS {
                            return Err(format!(
                                "block {block_id} orientation {orient_idx} has non-positive projected area"
                            ));
                        }
                        Ok(OrientationProjection {
                            area,
                            hull: union.convex_hull(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<_, _>>()?;
        let bay_areas = problem
            .bays
            .iter()
            .map(|bay| (bay.width * bay.height) as f64)
            .collect();

        let requests: Vec<_> = requests
            .iter()
            .map(|request| {
                PairRequest::new(
                    request.first_block_id,
                    request.second_block_id,
                    request.bay_id,
                )
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for request in &requests {
            if request.second_block_id >= problem.blocks.len() {
                return Err(format!(
                    "pair request references missing block: {:?}",
                    request
                ));
            }
            if request.bay_id >= problem.bays.len() {
                return Err(format!(
                    "pair request references missing bay: {:?}",
                    request
                ));
            }
        }

        let best: HashMap<_, _> = requests
            .par_iter()
            .filter_map(|&request| {
                best_pair_placement(problem, collision, &projections, request)
                    .map(|placement| (request, placement))
            })
            .collect();

        Ok(Self {
            projections,
            bay_areas,
            best,
        })
    }

    pub fn best(&self, request: PairRequest) -> Option<PairPlacement> {
        self.best.get(&request).copied()
    }

    pub fn evaluate(
        &self,
        first: PairPlacedBlock,
        second: PairPlacedBlock,
    ) -> Option<PairEvaluation> {
        if first.block_id == second.block_id || first.bay_id != second.bay_id {
            return None;
        }
        let (first, second) = if first.block_id < second.block_id {
            (first, second)
        } else {
            (second, first)
        };
        let request = PairRequest::new(first.block_id, second.block_id, first.bay_id);
        let best = self.best(request)?;
        let first_projection = self
            .projections
            .get(first.block_id)?
            .get(first.orient_idx)?;
        let second_projection = self
            .projections
            .get(second.block_id)?
            .get(second.orient_idx)?;
        let dx = second.x - first.x;
        let dy = second.y - first.y;
        let actual_hull_area = pair_hull_area(first_projection, second_projection, dx, dy);
        let area_sum = first_projection.area + second_projection.area;
        Some(PairEvaluation {
            request,
            actual_hull_area,
            first_area: first_projection.area,
            second_area: second_projection.area,
            compactness: actual_hull_area / area_sum,
            bay_area: *self.bay_areas.get(first.bay_id)?,
            best,
        })
    }
}

fn best_pair_placement(
    problem: &Problem,
    collision: &CollisionPrecompute,
    projections: &[Vec<OrientationProjection>],
    request: PairRequest,
) -> Option<PairPlacement> {
    let mut best: Option<PairPlacement> = None;
    for first_orient_idx in 0..problem.blocks[request.first_block_id].shape.len() {
        let Some(first_fit) =
            collision.fit_range(request.bay_id, request.first_block_id, first_orient_idx)
        else {
            continue;
        };
        let first_projection = &projections[request.first_block_id][first_orient_idx];
        for second_orient_idx in 0..problem.blocks[request.second_block_id].shape.len() {
            let Some(second_fit) =
                collision.fit_range(request.bay_id, request.second_block_id, second_orient_idx)
            else {
                continue;
            };
            let second_projection = &projections[request.second_block_id][second_orient_idx];
            let first_orient = BlockOrient {
                block_id: request.first_block_id,
                orient_idx: first_orient_idx,
            };
            let second_orient = BlockOrient {
                block_id: request.second_block_id,
                orient_idx: second_orient_idx,
            };
            let (first_second, second_first) =
                collision.crane_pairs_both_directions(first_orient, second_orient)?;

            let min_dx = second_fit.min_x - first_fit.max_x;
            let max_dx = second_fit.max_x - first_fit.min_x;
            let min_dy = second_fit.min_y - first_fit.max_y;
            let max_dy = second_fit.max_y - first_fit.min_y;
            for dy in min_dy..=max_dy {
                for dx in min_dx..=max_dx {
                    let first_blocked = first_second.crane.contains(dx, dy);
                    let second_blocked = second_first.crane.contains(-dx, -dy);
                    if first_blocked && second_blocked {
                        continue;
                    }

                    let hull_area = pair_hull_area(first_projection, second_projection, dx, dy);
                    let area_sum = first_projection.area + second_projection.area;
                    let candidate = PairPlacement {
                        first_orient_idx,
                        second_orient_idx,
                        dx,
                        dy,
                        hull_area,
                        area_sum,
                        compactness: hull_area / area_sum,
                    };
                    if best.as_ref().is_none_or(|best| {
                        candidate
                            .compactness
                            .total_cmp(&best.compactness)
                            .then(candidate.hull_area.total_cmp(&best.hull_area))
                            .then(candidate.first_orient_idx.cmp(&best.first_orient_idx))
                            .then(candidate.second_orient_idx.cmp(&best.second_orient_idx))
                            .then(candidate.dy.cmp(&best.dy))
                            .then(candidate.dx.cmp(&best.dx))
                            .is_lt()
                    }) {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    best
}

fn pair_hull_area(
    first: &OrientationProjection,
    second: &OrientationProjection,
    dx: i64,
    dy: i64,
) -> f64 {
    MultiPolygon(vec![
        first.hull.clone(),
        second.hull.translate(dx as f64, dy as f64),
    ])
    .convex_hull()
    .unsigned_area()
}
