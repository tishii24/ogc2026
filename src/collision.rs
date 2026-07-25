use crate::*;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

const AREA_EPS: f64 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockOrient {
    pub(crate) block_id: usize,
    pub(crate) orient_idx: usize,
}

#[derive(Clone, Debug)]
struct ConvexPart {
    points: Vec<Pointf>,
}

struct ConvexScratch {
    neg_b: Vec<Pointf>,
    hull: Vec<Pointf>,
}

#[derive(Clone, Debug)]
struct PolyLayer {
    parts: Vec<ConvexPart>,
    bbox: Boundsf,
}

#[derive(Clone, Debug)]
struct ShapeGeom {
    layers: Vec<PolyLayer>,
    bbox: Boundsf,
}

#[derive(Clone, Copy, Debug)]
struct DeltaRange {
    min_dx: i64,
    max_dx: i64,
    min_dy: i64,
    max_dy: i64,
}

#[derive(Clone, Debug)]
pub(crate) struct CollisionGrid {
    delta: DeltaRange,
    row_offsets: Vec<usize>,
    intervals: Vec<(i64, i64)>,
}

struct CollisionGridBuilder {
    delta: DeltaRange,
    rows: Vec<Vec<(i64, i64)>>,
}

#[derive(Clone, Debug)]
pub(crate) struct OrientPairCollision {
    pub(crate) crane: CollisionGrid,
}

struct OrientPairCache {
    ptr: AtomicPtr<OrientPairCollision>,
}

struct BlockPairCollision {
    fixed_orients: usize,
    orient_pairs: Vec<OrientPairCache>,
}

pub(crate) struct CollisionPrecompute {
    fit_ranges: Vec<Vec<Vec<Option<Boundsi>>>>,
    geoms: Vec<Vec<ShapeGeom>>,
    block_pair_index: Vec<Option<usize>>,
    block_pairs: Vec<BlockPairCollision>,
    n: usize,
}

impl CollisionPrecompute {
    pub(crate) fn build(problem: &Problem) -> Self {
        let n = problem.blocks.len();
        let geoms = build_all_geoms(problem);
        let fit_ranges = build_all_fit_ranges(problem, &geoms);
        let mut block_pair_index = vec![None; n * n];
        let mut block_pairs = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pair_idx = block_pairs.len();
                block_pairs.push(build_empty_block_pair(geoms[i].len(), geoms[j].len()));
                block_pair_index[i * n + j] = Some(pair_idx);
            }
        }

        Self {
            fit_ranges,
            geoms,
            block_pair_index,
            block_pairs,
            n,
        }
    }

    #[inline]
    pub(crate) fn crane_pair(
        &self,
        moving: BlockOrient,
        fixed: BlockOrient,
    ) -> Option<&OrientPairCollision> {
        if moving.block_id == fixed.block_id {
            return None;
        }

        let pair_idx = self.block_pair_index[moving.block_id * self.n + fixed.block_id]
            .expect("collision block pair should be precomputed");
        let pair = &self.block_pairs[pair_idx];
        let key = moving.orient_idx * pair.fixed_orients + fixed.orient_idx;
        Some(self.get_or_build_orient_pair(moving, fixed, pair_idx, key))
    }

    #[inline]
    pub(crate) fn crane_pairs_both_directions(
        &self,
        a: BlockOrient,
        b: BlockOrient,
    ) -> Option<(&OrientPairCollision, &OrientPairCollision)> {
        let ab = self.crane_pair(a, b)?;
        let ba = self.crane_pair(b, a)?;
        Some((ab, ba))
    }

    fn get_or_build_orient_pair(
        &self,
        moving: BlockOrient,
        fixed: BlockOrient,
        pair_idx: usize,
        key: usize,
    ) -> &OrientPairCollision {
        let cell = &self.block_pairs[pair_idx].orient_pairs[key];
        let ptr = cell.ptr.load(Ordering::Acquire);
        if !ptr.is_null() {
            return unsafe { &*ptr };
        }

        let moving_geom = &self.geoms[moving.block_id][moving.orient_idx];
        let fixed_geom = &self.geoms[fixed.block_id][fixed.orient_idx];
        let (forward_crane, reverse_crane) =
            build_crane_grids_both_directions(moving_geom, fixed_geom);

        let reverse_pair_idx = self.block_pair_index[fixed.block_id * self.n + moving.block_id]
            .expect("reverse collision block pair should be precomputed");
        let reverse_pair = &self.block_pairs[reverse_pair_idx];
        let reverse_key = fixed.orient_idx * reverse_pair.fixed_orients + moving.orient_idx;
        let reverse_cell = &reverse_pair.orient_pairs[reverse_key];

        let reverse = OrientPairCollision {
            crane: reverse_crane,
        };
        reverse_cell.publish(reverse);

        let forward = OrientPairCollision {
            crane: forward_crane,
        };
        let ptr = cell.publish(forward);
        unsafe { &*ptr }
    }

    pub(crate) fn fit_range(
        &self,
        bay_id: usize,
        block_id: usize,
        orient_idx: usize,
    ) -> Option<Boundsi> {
        self.fit_ranges
            .get(bay_id)?
            .get(block_id)?
            .get(orient_idx)?
            .to_owned()
    }
}

impl OrientPairCache {
    fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(ptr::null_mut()),
        }
    }

    fn publish(&self, collision: OrientPairCollision) -> *mut OrientPairCollision {
        let raw = Box::into_raw(Box::new(collision));
        match self
            .ptr
            .compare_exchange(ptr::null_mut(), raw, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => raw,
            Err(existing) => {
                unsafe {
                    drop(Box::from_raw(raw));
                }
                existing
            }
        }
    }
}

impl Drop for OrientPairCache {
    fn drop(&mut self) {
        let ptr = *self.ptr.get_mut();
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

impl CollisionGrid {
    #[inline]
    pub(crate) fn dx_intervals(&self, dy: i64) -> &[(i64, i64)] {
        if dy < self.delta.min_dy || self.delta.max_dy < dy {
            return &[];
        }

        let row = (dy - self.delta.min_dy) as usize;
        let begin = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        &self.intervals[begin..end]
    }
}

impl CollisionGridBuilder {
    fn new(range: DeltaRange) -> Self {
        let height = (range.max_dy - range.min_dy + 1) as usize;
        Self {
            delta: DeltaRange {
                min_dx: range.min_dx,
                max_dx: range.max_dx,
                min_dy: range.min_dy,
                max_dy: range.max_dy,
            },
            rows: vec![Vec::new(); height],
        }
    }

    fn add_x_interval(&mut self, dy: i64, min_dx: i64, max_dx: i64) {
        if dy < self.delta.min_dy || self.delta.max_dy < dy {
            return;
        }
        let min_dx = min_dx.max(self.delta.min_dx);
        let max_dx = max_dx.min(self.delta.max_dx);
        if min_dx > max_dx {
            return;
        }
        self.rows[(dy - self.delta.min_dy) as usize].push((min_dx, max_dx));
    }

    fn add_grid(&mut self, grid: &CollisionGrid) {
        for row in 0..(grid.delta.max_dy - grid.delta.min_dy + 1) as usize {
            let dy = grid.delta.min_dy + row as i64;
            let begin = grid.row_offsets[row];
            let end = grid.row_offsets[row + 1];
            for &(lo, hi) in &grid.intervals[begin..end] {
                self.add_x_interval(dy, lo, hi);
            }
        }
    }

    fn add_reversed_grid(&mut self, grid: &CollisionGrid) {
        for row in 0..(grid.delta.max_dy - grid.delta.min_dy + 1) as usize {
            let dy = grid.delta.min_dy + row as i64;
            let begin = grid.row_offsets[row];
            let end = grid.row_offsets[row + 1];
            for &(lo, hi) in &grid.intervals[begin..end] {
                self.add_x_interval(-dy, -hi, -lo);
            }
        }
    }

    fn finish(mut self) -> CollisionGrid {
        let mut row_offsets = Vec::with_capacity(self.rows.len() + 1);
        let mut intervals: Vec<(i64, i64)> = Vec::new();
        row_offsets.push(0);

        for row in &mut self.rows {
            row.sort_unstable();
            let row_start = intervals.len();
            for &(lo, hi) in row.iter() {
                if intervals.len() > row_start {
                    let last = intervals.last_mut().unwrap();
                    if lo <= last.1 + 1 {
                        last.1 = last.1.max(hi);
                        continue;
                    }
                }
                intervals.push((lo, hi));
            }
            row_offsets.push(intervals.len());
        }

        CollisionGrid {
            delta: DeltaRange {
                min_dx: self.delta.min_dx,
                max_dx: self.delta.max_dx,
                min_dy: self.delta.min_dy,
                max_dy: self.delta.max_dy,
            },
            row_offsets,
            intervals,
        }
    }
}

fn build_all_geoms(problem: &Problem) -> Vec<Vec<ShapeGeom>> {
    problem
        .blocks
        .iter()
        .map(|block| block.shape.iter().map(build_shape_geom).collect())
        .collect()
}

fn build_all_fit_ranges(
    problem: &Problem,
    geoms: &[Vec<ShapeGeom>],
) -> Vec<Vec<Vec<Option<Boundsi>>>> {
    problem
        .bays
        .iter()
        .map(|bay| {
            geoms
                .iter()
                .map(|block_geoms| {
                    block_geoms
                        .iter()
                        .map(|geom| build_fit_range(bay, geom.bbox))
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn build_shape_geom(orientation: &Orientation) -> ShapeGeom {
    let mut layers = Vec::new();
    let mut all_bbox: Option<Boundsf> = None;

    for layer in &orientation.layers {
        let points: Vec<Pointf> = layer.iter().map(|&[x, y]| Pointf { x, y }).collect();
        let parts = build_convex_parts(&points);
        assert!(
            !parts.is_empty(),
            "failed to build convex parts for layer with {} vertices, points={:?}",
            points.len(),
            points
        );
        let bbox = bbox_of_points(layer);
        all_bbox = Some(match all_bbox {
            Some(b) => merge_bbox(b, bbox),
            None => bbox,
        });
        layers.push(PolyLayer { parts, bbox });
    }

    let bbox = all_bbox.unwrap_or(Boundsf {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 0.0,
        max_y: 0.0,
    });
    ShapeGeom { layers, bbox }
}

fn build_convex_parts(points: &[Pointf]) -> Vec<ConvexPart> {
    if is_convex_polygon(points) {
        return vec![make_convex_part(points)];
    }

    triangulate_polygon(points)
        .into_iter()
        .map(|tri| make_convex_part(&tri))
        .collect()
}

fn is_convex_polygon(points: &[Pointf]) -> bool {
    if points.len() < 3 {
        return false;
    }
    let area = signed_area(points);
    if area.abs() <= AREA_EPS {
        return false;
    }
    let sign = if area > 0.0 { 1.0 } else { -1.0 };
    for i in 0..points.len() {
        if sign
            * cross(
                points[i],
                points[(i + 1) % points.len()],
                points[(i + 2) % points.len()],
            )
            < -AREA_EPS
        {
            return false;
        }
    }
    true
}

fn triangulate_polygon(points: &[Pointf]) -> Vec<[Pointf; 3]> {
    if points.len() < 3 || signed_area(points).abs() <= AREA_EPS {
        return Vec::new();
    }

    let mut idx: Vec<usize> = (0..points.len()).collect();
    if signed_area(points) < 0.0 {
        idx.reverse();
    }

    let mut triangles = Vec::with_capacity(points.len() - 2);
    while idx.len() > 3 {
        if polygon_area_abs_by_indices(points, &idx) <= AREA_EPS {
            return triangles;
        }

        let m = idx.len();
        let mut ear_pos = None;

        for pos in 0..m {
            let i0 = idx[(pos + m - 1) % m];
            let i1 = idx[pos];
            let i2 = idx[(pos + 1) % m];
            let tri = [points[i0], points[i1], points[i2]];

            if cross(tri[0], tri[1], tri[2]) <= AREA_EPS {
                continue;
            }

            let mut contains_other = false;
            for &iv in &idx {
                if iv == i0 || iv == i1 || iv == i2 {
                    continue;
                }
                if point_in_triangle_strict(points[iv], tri) {
                    contains_other = true;
                    break;
                }
            }

            if !contains_other && diagonal_clear(points, &idx, i0, i2) {
                ear_pos = Some(pos);
                triangles.push(tri);
                break;
            }
        }

        let Some(pos) = ear_pos else {
            if polygon_area_abs_by_indices(points, &idx) <= AREA_EPS {
                return triangles;
            }
            return Vec::new();
        };
        idx.remove(pos);
    }

    let tri = [points[idx[0]], points[idx[1]], points[idx[2]]];
    if cross(tri[0], tri[1], tri[2]) <= AREA_EPS {
        if polygon_area_abs_by_indices(points, &idx) <= AREA_EPS {
            return triangles;
        }
        return Vec::new();
    }
    triangles.push(tri);
    triangles
}

fn polygon_area_abs_by_indices(points: &[Pointf], idx: &[usize]) -> f64 {
    let mut area = 0.0;
    for i in 0..idx.len() {
        let a = points[idx[i]];
        let b = points[idx[(i + 1) % idx.len()]];
        area += a.x * b.y - b.x * a.y;
    }
    (area * 0.5).abs()
}

fn diagonal_clear(points: &[Pointf], idx: &[usize], a_idx: usize, b_idx: usize) -> bool {
    let a = points[a_idx];
    let b = points[b_idx];
    for i in 0..idx.len() {
        let c_idx = idx[i];
        let d_idx = idx[(i + 1) % idx.len()];
        if c_idx == a_idx || c_idx == b_idx || d_idx == a_idx || d_idx == b_idx {
            continue;
        }
        if segments_intersect(a, b, points[c_idx], points[d_idx]) {
            return false;
        }
    }
    true
}

fn segments_intersect(a: Pointf, b: Pointf, c: Pointf, d: Pointf) -> bool {
    let ab_c = cross(a, b, c);
    let ab_d = cross(a, b, d);
    let cd_a = cross(c, d, a);
    let cd_b = cross(c, d, b);

    if ab_c.abs() <= AREA_EPS && point_on_segment(c, a, b) {
        return true;
    }
    if ab_d.abs() <= AREA_EPS && point_on_segment(d, a, b) {
        return true;
    }
    if cd_a.abs() <= AREA_EPS && point_on_segment(a, c, d) {
        return true;
    }
    if cd_b.abs() <= AREA_EPS && point_on_segment(b, c, d) {
        return true;
    }

    ((ab_c > AREA_EPS && ab_d < -AREA_EPS) || (ab_c < -AREA_EPS && ab_d > AREA_EPS))
        && ((cd_a > AREA_EPS && cd_b < -AREA_EPS) || (cd_a < -AREA_EPS && cd_b > AREA_EPS))
}

fn point_on_segment(p: Pointf, a: Pointf, b: Pointf) -> bool {
    a.x.min(b.x) - AREA_EPS <= p.x
        && p.x <= a.x.max(b.x) + AREA_EPS
        && a.y.min(b.y) - AREA_EPS <= p.y
        && p.y <= a.y.max(b.y) + AREA_EPS
}

fn make_convex_part(points: &[Pointf]) -> ConvexPart {
    assert!(points.len() >= 3);
    let points = if signed_area(points) >= 0.0 {
        points.to_vec()
    } else {
        points.iter().rev().copied().collect()
    };

    ConvexPart { points }
}

fn signed_area(points: &[Pointf]) -> f64 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

fn cross(a: Pointf, b: Pointf, c: Pointf) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn point_in_triangle_strict(p: Pointf, t: [Pointf; 3]) -> bool {
    cross(t[0], t[1], p) > AREA_EPS
        && cross(t[1], t[2], p) > AREA_EPS
        && cross(t[2], t[0], p) > AREA_EPS
}

fn bbox_of_points(points: &[[f64; 2]]) -> Boundsf {
    let mut bbox = Boundsf {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };

    for &[x, y] in points {
        bbox.min_x = bbox.min_x.min(x);
        bbox.min_y = bbox.min_y.min(y);
        bbox.max_x = bbox.max_x.max(x);
        bbox.max_y = bbox.max_y.max(y);
    }

    bbox
}

fn merge_bbox(a: Boundsf, b: Boundsf) -> Boundsf {
    Boundsf {
        min_x: a.min_x.min(b.min_x),
        min_y: a.min_y.min(b.min_y),
        max_x: a.max_x.max(b.max_x),
        max_y: a.max_y.max(b.max_y),
    }
}

fn build_fit_range(bay: &Bay, bbox: Boundsf) -> Option<Boundsi> {
    let min_x = (-bbox.min_x).ceil() as i64;
    let max_x = (bay.width as f64 - bbox.max_x).floor() as i64;
    let min_y = (-bbox.min_y).ceil() as i64;
    let max_y = (bay.height as f64 - bbox.max_y).floor() as i64;

    if min_x <= max_x && min_y <= max_y {
        Some(Boundsi {
            min_x,
            max_x,
            min_y,
            max_y,
        })
    } else {
        None
    }
}

fn build_empty_block_pair(moving_orients: usize, fixed_orients: usize) -> BlockPairCollision {
    let orient_pairs = (0..moving_orients * fixed_orients)
        .map(|_| OrientPairCache::new())
        .collect();
    BlockPairCollision {
        fixed_orients,
        orient_pairs,
    }
}

fn build_crane_grids_both_directions(
    a: &ShapeGeom,
    b: &ShapeGeom,
) -> (CollisionGrid, CollisionGrid) {
    let mut ab_builder = CollisionGridBuilder::new(delta_range(a.bbox, b.bbox));
    let mut ba_builder = CollisionGridBuilder::new(delta_range(b.bbox, a.bbox));

    for ka in 0..a.layers.len() {
        for lb in 0..b.layers.len() {
            let grid = build_layer_pair_grid(&a.layers[ka], &b.layers[lb]);
            if lb >= ka {
                ab_builder.add_grid(&grid);
            }
            if ka >= lb {
                ba_builder.add_reversed_grid(&grid);
            }
        }
    }

    (ab_builder.finish(), ba_builder.finish())
}

fn build_layer_pair_grid(a: &PolyLayer, b: &PolyLayer) -> CollisionGrid {
    let mut builder = CollisionGridBuilder::new(delta_range(a.bbox, b.bbox));
    let mut scratch = ConvexScratch {
        neg_b: Vec::new(),
        hull: Vec::new(),
    };

    for pa in &a.parts {
        for pb in &b.parts {
            rasterize_convex_pair(&mut builder, pa, pb, &mut scratch);
        }
    }

    builder.finish()
}

fn rasterize_convex_pair(
    builder: &mut CollisionGridBuilder,
    a: &ConvexPart,
    b: &ConvexPart,
    scratch: &mut ConvexScratch,
) {
    let hull = minkowski_difference_hull(a, b, scratch);
    assert!(hull.len() >= 3, "failed to build Minkowski difference hull");

    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &p in hull {
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let min_dy = (min_y - AREA_EPS).ceil() as i64;
    let max_dy = (max_y + AREA_EPS).floor() as i64;
    let min_dy = min_dy.max(builder.delta.min_dy);
    let max_dy = max_dy.min(builder.delta.max_dy);

    for dy in min_dy..=max_dy {
        if let Some((min_dx, max_dx)) = horizontal_slice_conservative(hull, dy) {
            builder.add_x_interval(dy, min_dx, max_dx);
        }
    }
}

fn minkowski_difference_hull<'a>(
    a: &ConvexPart,
    b: &ConvexPart,
    scratch: &'a mut ConvexScratch,
) -> &'a [Pointf] {
    scratch.neg_b.clear();
    scratch.neg_b.reserve(b.points.len());
    scratch
        .neg_b
        .extend(b.points.iter().map(|&p| Pointf { x: -p.x, y: -p.y }));

    let a_points = a.points.as_slice();
    let b_points = scratch.neg_b.as_slice();
    let start_a = lowest_leftmost_index(a_points);
    let start_b = lowest_leftmost_index(b_points);

    scratch.hull.clear();
    scratch.hull.reserve(a_points.len() + b_points.len());
    let mut cur = Pointf {
        x: a_points[start_a].x + b_points[start_b].x,
        y: a_points[start_a].y + b_points[start_b].y,
    };
    scratch.hull.push(cur);

    let mut ia = 0;
    let mut ib = 0;
    while ia < a_points.len() || ib < b_points.len() {
        let take_a = if ib == b_points.len() {
            true
        } else if ia == a_points.len() {
            false
        } else {
            let ea = rotated_edge(a_points, start_a, ia);
            let eb = rotated_edge(b_points, start_b, ib);
            ea.x * eb.y - ea.y * eb.x >= 0.0
        };

        let edge = if take_a {
            let edge = rotated_edge(a_points, start_a, ia);
            ia += 1;
            edge
        } else {
            let edge = rotated_edge(b_points, start_b, ib);
            ib += 1;
            edge
        };

        cur.x += edge.x;
        cur.y += edge.y;
        if ia < a_points.len() || ib < b_points.len() {
            scratch.hull.push(cur);
        }
    }

    scratch.hull.as_slice()
}

fn lowest_leftmost_index(points: &[Pointf]) -> usize {
    let mut best = 0;
    for i in 1..points.len() {
        if points[i]
            .y
            .total_cmp(&points[best].y)
            .then(points[i].x.total_cmp(&points[best].x))
            .is_lt()
        {
            best = i;
        }
    }
    best
}

fn rotated_edge(points: &[Pointf], start: usize, offset: usize) -> Pointf {
    let len = points.len();
    let i = (start + offset) % len;
    let j = (start + offset + 1) % len;
    Pointf {
        x: points[j].x - points[i].x,
        y: points[j].y - points[i].y,
    }
}

fn horizontal_slice_conservative(poly: &[Pointf], dy: i64) -> Option<(i64, i64)> {
    let y = dy as f64;
    let mut low = f64::NEG_INFINITY;
    let mut high = f64::INFINITY;

    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        let ex = q.x - p.x;
        let ey = q.y - p.y;
        let rhs = ex * (y - p.y);

        if ey > AREA_EPS {
            high = high.min(p.x + rhs / ey);
        } else if ey < -AREA_EPS {
            low = low.max(p.x + rhs / ey);
        } else if ex * (y - p.y) < -AREA_EPS {
            return None;
        }
    }

    let min_dx = (low - AREA_EPS).ceil() as i64;
    let max_dx = (high + AREA_EPS).floor() as i64;
    if min_dx <= max_dx {
        Some((min_dx, max_dx))
    } else {
        None
    }
}

fn delta_range(moving: Boundsf, fixed: Boundsf) -> DeltaRange {
    DeltaRange {
        min_dx: (moving.min_x - fixed.max_x).floor() as i64 - 1,
        max_dx: (moving.max_x - fixed.min_x).ceil() as i64 + 1,
        min_dy: (moving.min_y - fixed.max_y).floor() as i64 - 1,
        max_dy: (moving.max_y - fixed.min_y).ceil() as i64 + 1,
    }
}
