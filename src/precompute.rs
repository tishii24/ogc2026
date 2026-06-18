use crate::*;
#[cfg(test)]
use geo::{Coord, LineString, Polygon};
use rayon::prelude::*;

const AREA_EPS: f64 = 1e-10;
const MAX_CONVEX_VERTS: usize = 10;
const MAX_MINKOWSKI_POINTS: usize = MAX_CONVEX_VERTS * MAX_CONVEX_VERTS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionResult {
    Hit,
    Clear,
    NotPrecomputed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPlacement {
    pub block_id: usize,
    pub orient_idx: usize,
    pub x: i64,
    pub y: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitRange {
    pub min_x: i64,
    pub max_x: i64,
    pub min_y: i64,
    pub max_y: i64,
}

#[derive(Clone, Copy, Debug)]
struct BBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct ConvexPart {
    points: [Point; MAX_CONVEX_VERTS],
    len: usize,
}

#[derive(Clone, Copy, Debug)]
struct ConvexPolygon {
    points: [Point; MAX_MINKOWSKI_POINTS],
    len: usize,
}

#[derive(Clone, Debug)]
struct PolyLayer {
    #[cfg(test)]
    polygon: Polygon<f64>,
    parts: Vec<ConvexPart>,
    bbox: BBox,
}

#[derive(Clone, Debug)]
struct ShapeGeom {
    layers: Vec<PolyLayer>,
    bbox: BBox,
}

#[derive(Clone, Copy, Debug)]
struct DeltaRange {
    min_dx: i64,
    max_dx: i64,
    min_dy: i64,
    max_dy: i64,
}

#[derive(Clone, Debug)]
struct CollisionGrid {
    min_dx: i64,
    max_dx: i64,
    min_dy: i64,
    max_dy: i64,
    column_offsets: Vec<usize>,
    intervals: Vec<(i64, i64)>,
}

struct CollisionGridBuilder {
    min_dx: i64,
    max_dx: i64,
    min_dy: i64,
    max_dy: i64,
    columns: Vec<Vec<(i64, i64)>>,
}

#[derive(Clone, Debug)]
struct OrientPairCollision {
    crane: CollisionGrid,
}

#[derive(Clone, Debug)]
struct BlockPairCollision {
    fixed_orients: usize,
    orient_pair_index: Vec<Option<usize>>,
    orient_pairs: Vec<OrientPairCollision>,
}

pub struct CollisionPrecompute {
    fit_ranges: Vec<Vec<Vec<Option<FitRange>>>>,
    block_pair_index: Vec<Option<usize>>,
    block_pairs: Vec<BlockPairCollision>,
    n: usize,
}

impl CollisionPrecompute {
    pub fn build(problem: &Problem, c: i64) -> Self {
        let n = problem.blocks.len();
        let geoms = build_all_geoms(problem);
        let fit_ranges = build_all_fit_ranges(problem, &geoms);
        let windows: Vec<(i64, i64)> = problem.blocks.iter().map(|b| time_window(b, c)).collect();
        let mut block_pair_index = vec![None; n * n];
        let mut block_pairs = Vec::new();

        let mut tasks = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if windows_overlap(windows[i], windows[j]) {
                    tasks.push((i, j));
                }
            }
        }
        eprintln!("building {} collision block pairs", tasks.len());

        let built_pairs: Vec<_> = tasks
            .par_iter()
            .map(|&(i, j)| {
                let (ij, ji) = build_bidirectional_block_pair(&geoms, i, j);
                (i, j, ij, ji)
            })
            .collect();

        for (i, j, ij_collision, ji_collision) in built_pairs {
            let ij = block_pairs.len();
            block_pairs.push(ij_collision);
            block_pair_index[i * n + j] = Some(ij);

            let ji = block_pairs.len();
            block_pairs.push(ji_collision);
            block_pair_index[j * n + i] = Some(ji);
        }

        Self {
            fit_ranges,
            block_pair_index,
            block_pairs,
            n,
        }
    }

    pub fn crane(&self, moving: BlockPlacement, fixed: BlockPlacement) -> CollisionResult {
        if moving.block_id == fixed.block_id {
            return CollisionResult::Clear;
        }

        let Some(pair_idx) = self.block_pair_index[moving.block_id * self.n + fixed.block_id]
        else {
            return CollisionResult::NotPrecomputed;
        };

        let pair = &self.block_pairs[pair_idx];
        let key = moving.orient_idx * pair.fixed_orients + fixed.orient_idx;
        let Some(orient_pair_idx) = pair.orient_pair_index[key] else {
            return CollisionResult::NotPrecomputed;
        };

        let orient_pair = &pair.orient_pairs[orient_pair_idx];
        let dx = fixed.x - moving.x;
        let dy = fixed.y - moving.y;

        if orient_pair.crane.get(dx, dy) {
            CollisionResult::Hit
        } else {
            CollisionResult::Clear
        }
    }

    pub fn fit_range(&self, bay_id: usize, block_id: usize, orient_idx: usize) -> Option<FitRange> {
        self.fit_ranges
            .get(bay_id)?
            .get(block_id)?
            .get(orient_idx)?
            .to_owned()
    }
}

impl CollisionGrid {
    fn get(&self, dx: i64, dy: i64) -> bool {
        if dx < self.min_dx || self.max_dx < dx || dy < self.min_dy || self.max_dy < dy {
            return false;
        }

        let col = (dx - self.min_dx) as usize;
        let begin = self.column_offsets[col];
        let end = self.column_offsets[col + 1];
        self.intervals[begin..end]
            .binary_search_by(|&(lo, hi)| {
                if dy < lo {
                    std::cmp::Ordering::Greater
                } else if hi < dy {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }
}

impl CollisionGridBuilder {
    fn new(range: DeltaRange) -> Self {
        let width = (range.max_dx - range.min_dx + 1) as usize;
        Self {
            min_dx: range.min_dx,
            max_dx: range.max_dx,
            min_dy: range.min_dy,
            max_dy: range.max_dy,
            columns: vec![Vec::new(); width],
        }
    }

    fn add_interval(&mut self, dx: i64, min_dy: i64, max_dy: i64) {
        if dx < self.min_dx || self.max_dx < dx {
            return;
        }
        let min_dy = min_dy.max(self.min_dy);
        let max_dy = max_dy.min(self.max_dy);
        if min_dy > max_dy {
            return;
        }
        self.columns[(dx - self.min_dx) as usize].push((min_dy, max_dy));
    }

    fn add_grid(&mut self, grid: &CollisionGrid) {
        for col in 0..(grid.max_dx - grid.min_dx + 1) as usize {
            let dx = grid.min_dx + col as i64;
            let begin = grid.column_offsets[col];
            let end = grid.column_offsets[col + 1];
            for &(lo, hi) in &grid.intervals[begin..end] {
                self.add_interval(dx, lo, hi);
            }
        }
    }

    fn add_reversed_grid(&mut self, grid: &CollisionGrid) {
        for col in 0..(grid.max_dx - grid.min_dx + 1) as usize {
            let dx = grid.min_dx + col as i64;
            let begin = grid.column_offsets[col];
            let end = grid.column_offsets[col + 1];
            for &(lo, hi) in &grid.intervals[begin..end] {
                self.add_interval(-dx, -hi, -lo);
            }
        }
    }

    fn finish(mut self) -> CollisionGrid {
        let mut column_offsets = Vec::with_capacity(self.columns.len() + 1);
        let mut intervals: Vec<(i64, i64)> = Vec::new();
        column_offsets.push(0);

        for col in &mut self.columns {
            col.sort_unstable();
            let col_start = intervals.len();
            for &(lo, hi) in col.iter() {
                if intervals.len() > col_start {
                    let last = intervals.last_mut().unwrap();
                    if lo <= last.1 + 1 {
                        last.1 = last.1.max(hi);
                        continue;
                    }
                }
                intervals.push((lo, hi));
            }
            column_offsets.push(intervals.len());
        }

        CollisionGrid {
            min_dx: self.min_dx,
            max_dx: self.max_dx,
            min_dy: self.min_dy,
            max_dy: self.max_dy,
            column_offsets,
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
) -> Vec<Vec<Vec<Option<FitRange>>>> {
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
    let mut all_bbox: Option<BBox> = None;

    for layer in &orientation.layers {
        #[cfg(test)]
        let polygon = {
            let mut coords: Vec<Coord<f64>> = layer.iter().map(|&[x, y]| Coord { x, y }).collect();
            if coords.first() != coords.last() {
                if let Some(first) = coords.first().copied() {
                    coords.push(first);
                }
            }
            Polygon::new(LineString::from(coords), vec![])
        };

        let points: Vec<Point> = layer.iter().map(|&[x, y]| Point { x, y }).collect();
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
        layers.push(PolyLayer {
            #[cfg(test)]
            polygon,
            parts,
            bbox,
        });
    }

    let bbox = all_bbox.unwrap_or(BBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 0.0,
        max_y: 0.0,
    });
    ShapeGeom { layers, bbox }
}

fn build_convex_parts(points: &[Point]) -> Vec<ConvexPart> {
    if is_convex_polygon(points) {
        return vec![make_convex_part(points)];
    }

    triangulate_polygon(points)
        .into_iter()
        .map(|tri| make_convex_part(&tri))
        .collect()
}

fn is_convex_polygon(points: &[Point]) -> bool {
    if points.len() < 3 || points.len() > MAX_CONVEX_VERTS {
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

fn triangulate_polygon(points: &[Point]) -> Vec<[Point; 3]> {
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

fn polygon_area_abs_by_indices(points: &[Point], idx: &[usize]) -> f64 {
    let mut area = 0.0;
    for i in 0..idx.len() {
        let a = points[idx[i]];
        let b = points[idx[(i + 1) % idx.len()]];
        area += a.x * b.y - b.x * a.y;
    }
    (area * 0.5).abs()
}

fn diagonal_clear(points: &[Point], idx: &[usize], a_idx: usize, b_idx: usize) -> bool {
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

fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
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

fn point_on_segment(p: Point, a: Point, b: Point) -> bool {
    a.x.min(b.x) - AREA_EPS <= p.x
        && p.x <= a.x.max(b.x) + AREA_EPS
        && a.y.min(b.y) - AREA_EPS <= p.y
        && p.y <= a.y.max(b.y) + AREA_EPS
}

fn make_convex_part(points: &[Point]) -> ConvexPart {
    assert!(points.len() >= 3 && points.len() <= MAX_CONVEX_VERTS);
    let mut part_points = [Point { x: 0.0, y: 0.0 }; MAX_CONVEX_VERTS];
    let len = points.len();
    if signed_area(points) >= 0.0 {
        part_points[..len].copy_from_slice(points);
    } else {
        for i in 0..len {
            part_points[i] = points[len - 1 - i];
        }
    }

    ConvexPart {
        points: part_points,
        len,
    }
}

fn signed_area(points: &[Point]) -> f64 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn point_in_triangle_strict(p: Point, t: [Point; 3]) -> bool {
    cross(t[0], t[1], p) > AREA_EPS
        && cross(t[1], t[2], p) > AREA_EPS
        && cross(t[2], t[0], p) > AREA_EPS
}

#[cfg(test)]
fn bbox_of_point_slice(points: &[Point]) -> BBox {
    let mut bbox = BBox {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };

    for &p in points {
        bbox.min_x = bbox.min_x.min(p.x);
        bbox.min_y = bbox.min_y.min(p.y);
        bbox.max_x = bbox.max_x.max(p.x);
        bbox.max_y = bbox.max_y.max(p.y);
    }

    bbox
}

fn bbox_of_points(points: &[[f64; 2]]) -> BBox {
    let mut bbox = BBox {
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

fn merge_bbox(a: BBox, b: BBox) -> BBox {
    BBox {
        min_x: a.min_x.min(b.min_x),
        min_y: a.min_y.min(b.min_y),
        max_x: a.max_x.max(b.max_x),
        max_y: a.max_y.max(b.max_y),
    }
}

fn build_fit_range(bay: &Bay, bbox: BBox) -> Option<FitRange> {
    let min_x = (-bbox.min_x).ceil() as i64;
    let max_x = (bay.width as f64 - bbox.max_x).floor() as i64;
    let min_y = (-bbox.min_y).ceil() as i64;
    let max_y = (bay.height as f64 - bbox.max_y).floor() as i64;

    if min_x <= max_x && min_y <= max_y {
        Some(FitRange {
            min_x,
            max_x,
            min_y,
            max_y,
        })
    } else {
        None
    }
}

fn time_window(block: &Block, c: i64) -> (i64, i64) {
    ((block.release_time - c).max(0), block.due_date + c)
}

fn windows_overlap(a: (i64, i64), b: (i64, i64)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

fn build_bidirectional_block_pair(
    geoms: &[Vec<ShapeGeom>],
    a_block: usize,
    b_block: usize,
) -> (BlockPairCollision, BlockPairCollision) {
    let a_orients = geoms[a_block].len();
    let b_orients = geoms[b_block].len();
    let mut ab_orient_pair_index = vec![None; a_orients * b_orients];
    let mut ba_orient_pair_index = vec![None; b_orients * a_orients];
    let mut ab_orient_pairs = Vec::new();
    let mut ba_orient_pairs = Vec::new();

    for a_orient in 0..a_orients {
        for b_orient in 0..b_orients {
            let a = &geoms[a_block][a_orient];
            let b = &geoms[b_block][b_orient];
            let (ab_crane, ba_crane) = build_crane_grids_both_directions(a, b);

            let ab_idx = ab_orient_pairs.len();
            ab_orient_pair_index[a_orient * b_orients + b_orient] = Some(ab_idx);
            ab_orient_pairs.push(OrientPairCollision { crane: ab_crane });

            let ba_idx = ba_orient_pairs.len();
            ba_orient_pair_index[b_orient * a_orients + a_orient] = Some(ba_idx);
            ba_orient_pairs.push(OrientPairCollision { crane: ba_crane });
        }
    }

    (
        BlockPairCollision {
            fixed_orients: b_orients,
            orient_pair_index: ab_orient_pair_index,
            orient_pairs: ab_orient_pairs,
        },
        BlockPairCollision {
            fixed_orients: a_orients,
            orient_pair_index: ba_orient_pair_index,
            orient_pairs: ba_orient_pairs,
        },
    )
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

    for &pa in &a.parts {
        for &pb in &b.parts {
            rasterize_convex_pair(&mut builder, pa, pb);
        }
    }

    builder.finish()
}

fn rasterize_convex_pair(builder: &mut CollisionGridBuilder, a: ConvexPart, b: ConvexPart) {
    let hull = minkowski_difference_hull(a, b);
    assert!(hull.len >= 3, "failed to build Minkowski difference hull");

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    for i in 0..hull.len {
        min_x = min_x.min(hull.points[i].x);
        max_x = max_x.max(hull.points[i].x);
    }
    let min_dx = ((min_x.floor() as i64) + 1).max(builder.min_dx);
    let max_dx = ((max_x.ceil() as i64) - 1).min(builder.max_dx);

    for dx in min_dx..=max_dx {
        if let Some((min_dy, max_dy)) = vertical_slice_strict(&hull, dx) {
            builder.add_interval(dx, min_dy, max_dy);
        }
    }
}

fn minkowski_difference_hull(a: ConvexPart, b: ConvexPart) -> ConvexPolygon {
    let mut neg_b = [Point { x: 0.0, y: 0.0 }; MAX_CONVEX_VERTS];
    for (dst, src) in neg_b.iter_mut().zip(b.points.iter()).take(b.len) {
        *dst = Point {
            x: -src.x,
            y: -src.y,
        };
    }

    let start_a = lowest_leftmost_index(&a.points, a.len);
    let start_b = lowest_leftmost_index(&neg_b, b.len);
    let mut points = [Point { x: 0.0, y: 0.0 }; MAX_MINKOWSKI_POINTS];
    let mut len = 1;
    let mut cur = Point {
        x: a.points[start_a].x + neg_b[start_b].x,
        y: a.points[start_a].y + neg_b[start_b].y,
    };
    points[0] = cur;

    let mut ia = 0;
    let mut ib = 0;
    while ia < a.len || ib < b.len {
        let take_a = if ib == b.len {
            true
        } else if ia == a.len {
            false
        } else {
            let ea = rotated_edge(&a.points, a.len, start_a, ia);
            let eb = rotated_edge(&neg_b, b.len, start_b, ib);
            ea.x * eb.y - ea.y * eb.x >= 0.0
        };

        let edge = if take_a {
            let edge = rotated_edge(&a.points, a.len, start_a, ia);
            ia += 1;
            edge
        } else {
            let edge = rotated_edge(&neg_b, b.len, start_b, ib);
            ib += 1;
            edge
        };

        cur.x += edge.x;
        cur.y += edge.y;
        if ia < a.len || ib < b.len {
            points[len] = cur;
            len += 1;
        }
    }

    ConvexPolygon { points, len }
}

fn lowest_leftmost_index(points: &[Point], len: usize) -> usize {
    let mut best = 0;
    for i in 1..len {
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

fn rotated_edge(points: &[Point], len: usize, start: usize, offset: usize) -> Point {
    let i = (start + offset) % len;
    let j = (start + offset + 1) % len;
    Point {
        x: points[j].x - points[i].x,
        y: points[j].y - points[i].y,
    }
}

fn vertical_slice_strict(poly: &ConvexPolygon, dx: i64) -> Option<(i64, i64)> {
    let x = dx as f64;
    let mut low = f64::NEG_INFINITY;
    let mut high = f64::INFINITY;

    for i in 0..poly.len {
        let p = poly.points[i];
        let q = poly.points[(i + 1) % poly.len];
        let ex = q.x - p.x;
        let ey = q.y - p.y;
        let rhs = ey * (x - p.x) + AREA_EPS;

        if ex > AREA_EPS {
            low = low.max(p.y + rhs / ex);
        } else if ex < -AREA_EPS {
            high = high.min(p.y + rhs / ex);
        } else if -ey * (x - p.x) <= AREA_EPS {
            return None;
        }
    }

    let min_dy = low.floor() as i64 + 1;
    let max_dy = high.ceil() as i64 - 1;
    if min_dy <= max_dy {
        Some((min_dy, max_dy))
    } else {
        None
    }
}

fn delta_range(moving: BBox, fixed: BBox) -> DeltaRange {
    DeltaRange {
        min_dx: (moving.min_x - fixed.max_x).floor() as i64 - 1,
        max_dx: (moving.max_x - fixed.min_x).ceil() as i64 + 1,
        min_dy: (moving.min_y - fixed.max_y).floor() as i64 - 1,
        max_dy: (moving.max_y - fixed.min_y).ceil() as i64 + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Area, BooleanOps};

    fn translate_polygon(poly: &Polygon<f64>, dx: i64, dy: i64) -> Polygon<f64> {
        let dx = dx as f64;
        let dy = dy as f64;
        let exterior: Vec<Coord<f64>> = poly
            .exterior()
            .points()
            .map(|p| Coord {
                x: p.x() + dx,
                y: p.y() + dy,
            })
            .collect();
        Polygon::new(LineString::from(exterior), vec![])
    }

    fn polygons_overlap_area_positive(a: &Polygon<f64>, b: &Polygon<f64>) -> bool {
        a.intersection(b).unsigned_area() > AREA_EPS
    }

    fn rect(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<[f64; 2]> {
        vec![
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ]
    }

    fn test_block(release_time: i64, due_date: i64, layers: Vec<Vec<[f64; 2]>>) -> Block {
        Block {
            release_time,
            due_date,
            processing_time: 1,
            bay_preferences: vec![1],
            shape: vec![Orientation { layers }],
        }
    }

    fn problem(bays: Vec<Bay>, blocks: Vec<Block>) -> Problem {
        Problem { bays, blocks }
    }

    fn placement(block_id: usize, x: i64, y: i64) -> BlockPlacement {
        BlockPlacement {
            block_id,
            orient_idx: 0,
            x,
            y,
        }
    }

    fn single_layer_geom(layer: Vec<[f64; 2]>) -> PolyLayer {
        build_shape_geom(&Orientation {
            layers: vec![layer],
        })
        .layers
        .into_iter()
        .next()
        .unwrap()
    }

    fn build_crane_grid(moving: &ShapeGeom, fixed: &ShapeGeom, range: DeltaRange) -> CollisionGrid {
        let mut builder = CollisionGridBuilder::new(range);

        for k in 0..moving.layers.len() {
            for j in k..fixed.layers.len() {
                let grid = build_layer_pair_grid(&moving.layers[k], &fixed.layers[j]);
                builder.add_grid(&grid);
            }
        }

        builder.finish()
    }

    fn convex_parts_overlap_any(a: &PolyLayer, b: &PolyLayer, dx: i64, dy: i64) -> bool {
        let dx_f = dx as f64;
        let dy_f = dy as f64;

        for &pa in &a.parts {
            for &pb in &b.parts {
                if !bbox_may_overlap(
                    bbox_of_point_slice(&pa.points[..pa.len]),
                    bbox_of_point_slice(&pb.points[..pb.len]),
                    dx,
                    dy,
                ) {
                    continue;
                }
                if convex_parts_overlap_area_positive(pa, pb, dx_f, dy_f) {
                    return true;
                }
            }
        }
        false
    }

    fn convex_parts_overlap_area_positive(a: ConvexPart, b: ConvexPart, dx: f64, dy: f64) -> bool {
        for i in 0..a.len {
            let axis = edge_normal(a.points[i], a.points[(i + 1) % a.len]);
            if axis.x.abs() <= AREA_EPS && axis.y.abs() <= AREA_EPS {
                continue;
            }
            let (amin, amax) = project_point_slice(&a.points[..a.len], axis);
            let (bmin, bmax) = project_convex_part(b, axis, dx, dy);
            if amax <= bmin + AREA_EPS || bmax <= amin + AREA_EPS {
                return false;
            }
        }

        for i in 0..b.len {
            let axis = edge_normal(b.points[i], b.points[(i + 1) % b.len]);
            if axis.x.abs() <= AREA_EPS && axis.y.abs() <= AREA_EPS {
                continue;
            }
            let shift = dx * axis.x + dy * axis.y;
            let (bmin0, bmax0) = project_point_slice(&b.points[..b.len], axis);
            let (bmin, bmax) = (bmin0 + shift, bmax0 + shift);
            let (amin, amax) = project_point_slice(&a.points[..a.len], axis);
            if amax <= bmin + AREA_EPS || bmax <= amin + AREA_EPS {
                return false;
            }
        }

        true
    }

    fn edge_normal(a: Point, b: Point) -> Point {
        let ex = b.x - a.x;
        let ey = b.y - a.y;
        Point { x: -ey, y: ex }
    }

    fn project_convex_part(part: ConvexPart, axis: Point, dx: f64, dy: f64) -> (f64, f64) {
        let (min, max) = project_point_slice(&part.points[..part.len], axis);
        let shift = dx * axis.x + dy * axis.y;
        (min + shift, max + shift)
    }

    fn project_point_slice(points: &[Point], axis: Point) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &p in points {
            let v = project_point(p, axis, 0.0, 0.0);
            min = min.min(v);
            max = max.max(v);
        }
        (min, max)
    }

    fn project_point(p: Point, axis: Point, dx: f64, dy: f64) -> f64 {
        (p.x + dx) * axis.x + (p.y + dy) * axis.y
    }

    fn bbox_may_overlap(a: BBox, b: BBox, dx: i64, dy: i64) -> bool {
        let dx = dx as f64;
        let dy = dy as f64;
        let b_min_x = b.min_x + dx;
        let b_max_x = b.max_x + dx;
        let b_min_y = b.min_y + dy;
        let b_max_y = b.max_y + dy;

        a.min_x < b_max_x - AREA_EPS
            && b_min_x < a.max_x - AREA_EPS
            && a.min_y < b_max_y - AREA_EPS
            && b_min_y < a.max_y - AREA_EPS
    }

    #[test]
    fn triangulation_sat_matches_geo_for_sample_offsets() {
        let a = single_layer_geom(vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [1.0, 1.0],
            [1.0, 4.0],
            [0.0, 4.0],
        ]);
        let b = single_layer_geom(vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.5, 1.5],
            [1.0, 2.5],
            [-0.5, 1.0],
        ]);

        for dx in -4..=5 {
            for dy in -4..=5 {
                let fast = convex_parts_overlap_any(&a, &b, dx, dy);
                let shifted_b = translate_polygon(&b.polygon, dx, dy);
                let exact = polygons_overlap_area_positive(&a.polygon, &shifted_b);
                assert_eq!(fast, exact, "mismatch at dx={dx}, dy={dy}");
            }
        }
    }

    #[test]
    fn triangulation_sat_treats_boundary_touch_as_clear() {
        let a = single_layer_geom(rect(0.0, 0.0, 2.0, 2.0));
        let b = single_layer_geom(rect(0.0, 0.0, 2.0, 2.0));

        assert!(!convex_parts_overlap_any(&a, &b, 2, 0));
        assert!(!convex_parts_overlap_any(&a, &b, 2, 2));
        assert!(convex_parts_overlap_any(&a, &b, 1, 0));
    }

    #[test]
    fn triangulation_accepts_degenerate_remainder_cases() {
        let cases = [
            vec![
                [0.0, 0.0],
                [5.8138, 0.0],
                [11.6276, 5.0638],
                [11.6276, 0.0],
                [12.1797, 0.0],
                [5.4132, -4.7149],
            ],
            vec![
                [0.0, 0.0],
                [-0.9711, 0.0],
                [-2.9132, 0.0],
                [-2.9132, 3.169],
                [-2.9132, 6.338],
                [-0.9711, 3.169],
                [0.0, 3.169],
            ],
            vec![
                [0.0, 0.0],
                [3.7083, 0.0],
                [3.7083, 3.2128],
                [7.4166, 3.2128],
                [11.1249, 3.2128],
                [7.4166, 0.0],
                [7.4166, -0.1289],
                [3.0559, -2.6476],
            ],
        ];

        for layer in cases {
            let geom = single_layer_geom(layer);
            assert!(!geom.parts.is_empty());
        }
    }

    #[test]
    fn rasterized_grid_matches_geo_for_sample_offsets() {
        let moving = build_shape_geom(&Orientation {
            layers: vec![vec![
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 1.0],
                [1.0, 1.0],
                [1.0, 4.0],
                [0.0, 4.0],
            ]],
        });
        let fixed = build_shape_geom(&Orientation {
            layers: vec![vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.5, 1.5],
                [1.0, 2.5],
                [-0.5, 1.0],
            ]],
        });
        let grid = build_crane_grid(&moving, &fixed, delta_range(moving.bbox, fixed.bbox));
        let a = &moving.layers[0];
        let b = &fixed.layers[0];

        for dx in -4..=5 {
            for dy in -4..=5 {
                let shifted_b = translate_polygon(&b.polygon, dx, dy);
                let exact = polygons_overlap_area_positive(&a.polygon, &shifted_b);
                assert_eq!(grid.get(dx, dy), exact, "mismatch at dx={dx}, dy={dy}");
            }
        }
    }

    fn prob1_sample_orientation_pairs() -> Vec<(usize, Vec<Vec<[f64; 2]>>, Vec<Vec<[f64; 2]>>)> {
        vec![
            // sample 0: moving block 93 orient 5, fixed block 73 orient 2
            (
                0,
                vec![
                    vec![
                        [0.0, 0.0],
                        [2.6458, 2.6323],
                        [4.8109, 5.28],
                        [7.5066, 7.8563],
                        [10.0008, 10.1925],
                        [14.2912, 6.8295],
                        [11.745, 4.2176],
                        [9.1329, 1.9117],
                        [6.2574, -0.8649],
                        [4.1787, -3.4575],
                    ],
                    vec![
                        [0.0, 0.0],
                        [2.6458, 2.6323],
                        [4.8109, 5.28],
                        [7.5066, 7.8563],
                        [11.745, 4.2176],
                        [9.1329, 1.9117],
                        [6.2574, -0.8649],
                        [4.1787, -3.4575],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [-0.1208, 4.4143],
                        [0.1192, 8.9078],
                        [-2.7367, 8.5968],
                        [-2.5549, 4.3381],
                        [-2.7477, 0.3806],
                    ],
                    vec![
                        [0.0, 0.0],
                        [-0.1208, 4.4143],
                        [-2.5549, 4.3381],
                        [-2.7477, 0.3806],
                    ],
                ],
            ),
            // sample 1: moving block 86 orient 0, fixed block 1 orient 1
            (
                1,
                vec![
                    vec![
                        [0.0, 0.0],
                        [0.0587, 2.0547],
                        [0.8348, 6.3076],
                        [6.7991, 6.0449],
                        [6.7855, 4.5264],
                        [2.0032, -0.0708],
                    ],
                    vec![
                        [0.0, 0.0],
                        [0.0587, 2.0547],
                        [0.8348, 6.3076],
                        [6.7991, 6.0449],
                        [6.7855, 4.5264],
                        [2.0032, -0.0708],
                    ],
                ],
                vec![vec![
                    [0.0, 0.0],
                    [-3.5357, -2.9894],
                    [-7.1829, -6.8657],
                    [-10.4375, -9.9073],
                    [-13.9889, -13.8325],
                    [-13.2975, -7.9815],
                    [-9.3778, -4.5306],
                    [-5.9856, -1.0237],
                    [-2.7886, 2.2157],
                ]],
            ),
            // sample 2: moving block 50 orient 3, fixed block 28 orient 5
            (
                2,
                vec![
                    vec![
                        [0.0, 0.0],
                        [3.9402, -4.4489],
                        [8.0537, -8.4897],
                        [11.9908, -12.3331],
                        [5.5327, -10.9862],
                        [1.3074, -7.4016],
                        [-2.7125, -2.9988],
                    ],
                    vec![
                        [3.9402, -4.4489],
                        [8.0537, -8.4897],
                        [11.9908, -12.3331],
                        [5.5327, -10.9862],
                        [1.3074, -7.4016],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [5.1922, 5.6864],
                        [9.6326, 2.1882],
                        [3.9616, -2.9865],
                    ],
                    vec![
                        [-3.4518, 3.7324],
                        [1.7561, 8.8989],
                        [5.1922, 5.6864],
                        [9.6326, 2.1882],
                        [3.9616, -2.9865],
                        [0.0, 0.0],
                    ],
                ],
            ),
            // sample 3: moving block 65 orient 3, fixed block 20 orient 6
            (
                3,
                vec![
                    vec![
                        [0.0, 0.0],
                        [-3.9415, 3.8597],
                        [-5.7884, 5.5226],
                        [-11.6666, 0.2447],
                        [-9.9061, -1.4452],
                    ],
                    vec![
                        [-3.9415, 3.8597],
                        [-7.6354, 7.1853],
                        [-13.4271, 1.9346],
                        [-9.9061, -1.4452],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [-0.1529, 2.642],
                        [-0.1986, 5.3743],
                        [0.0005, 8.3586],
                        [2.2661, 8.5556],
                        [2.4148, 5.5336],
                        [2.4125, 2.8385],
                        [2.3494, 0.1132],
                    ],
                    vec![
                        [0.0, 0.0],
                        [-0.1529, 2.642],
                        [-0.1986, 5.3743],
                        [0.0005, 8.3586],
                        [2.2661, 8.5556],
                        [2.4148, 5.5336],
                        [2.4125, 2.8385],
                        [2.3494, 0.1132],
                    ],
                ],
            ),
            // sample 4: moving block 86 orient 2, fixed block 71 orient 4
            (
                4,
                vec![
                    vec![
                        [0.0, 0.0],
                        [-2.0547, 0.0587],
                        [-6.3076, 0.8348],
                        [-6.0449, 6.7991],
                        [-4.5264, 6.7855],
                        [0.0708, 2.0032],
                    ],
                    vec![
                        [0.0, 0.0],
                        [-2.0547, 0.0587],
                        [-6.3076, 0.8348],
                        [-6.0449, 6.7991],
                        [-4.5264, 6.7855],
                        [0.0708, 2.0032],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [-7.6229, -6.1855],
                        [-7.7176, -9.1117],
                        [-0.6491, -8.83],
                        [-0.6396, -5.8127],
                    ],
                    vec![
                        [0.0, 0.0],
                        [-7.6229, -6.1855],
                        [-7.7176, -9.1117],
                        [-0.6491, -8.83],
                        [-0.6396, -5.8127],
                    ],
                ],
            ),
            // sample 5: moving block 93 orient 7, fixed block 9 orient 7
            (
                5,
                vec![
                    vec![
                        [0.0, 0.0],
                        [-2.6323, 2.6458],
                        [-5.28, 4.8109],
                        [-7.8563, 7.5066],
                        [-10.1925, 10.0008],
                        [-6.8295, 14.2912],
                        [-4.2176, 11.745],
                        [-1.9117, 9.1329],
                        [0.8649, 6.2574],
                        [3.4575, 4.1787],
                    ],
                    vec![
                        [0.0, 0.0],
                        [-2.6323, 2.6458],
                        [-5.28, 4.8109],
                        [-7.8563, 7.5066],
                        [-4.2176, 11.745],
                        [-1.9117, 9.1329],
                        [0.8649, 6.2574],
                        [3.4575, 4.1787],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [4.496, -4.8214],
                        [8.0377, -1.4448],
                        [3.4539, 3.158],
                    ],
                    vec![
                        [-2.5515, 2.379],
                        [0.0, 0.0],
                        [2.248, -2.4107],
                        [5.7458, 0.8567],
                        [3.4539, 3.158],
                        [1.0224, 5.5883],
                    ],
                ],
            ),
            // sample 6: moving block 8 orient 4, fixed block 75 orient 0
            (
                6,
                vec![
                    vec![
                        [0.0, 0.0],
                        [5.3022, 0.0079],
                        [5.6723, -4.3543],
                        [5.9821, -9.6951],
                        [6.059, -11.6614],
                        [0.5404, -11.4865],
                        [0.3443, -9.2268],
                        [0.2134, -4.8939],
                    ],
                    vec![
                        [0.0, 0.0],
                        [5.3022, 0.0079],
                        [5.6723, -4.3543],
                        [5.9821, -9.6951],
                        [6.059, -11.6614],
                        [0.5404, -11.4865],
                        [0.3443, -9.2268],
                        [0.2134, -4.8939],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [-5.6189, -0.5908],
                        [-11.1639, -0.59],
                        [-6.4127, 3.9415],
                        [-0.8178, 4.1183],
                    ],
                    vec![
                        [5.3776, -0.2609],
                        [0.0, 0.0],
                        [-5.6189, -0.5908],
                        [-11.1639, -0.59],
                        [-6.4127, 3.9415],
                        [-0.8178, 4.1183],
                    ],
                ],
            ),
            // sample 7: moving block 17 orient 7, fixed block 62 orient 4
            (
                7,
                vec![
                    vec![
                        [0.0, 0.0],
                        [2.2154, 1.9789],
                        [4.0528, 3.6944],
                        [6.2955, 5.5252],
                        [8.0597, 4.0116],
                        [6.7397, 2.6259],
                        [1.4912, 0.0623],
                    ],
                    vec![
                        [0.0, 0.0],
                        [2.2154, 1.9789],
                        [3.1341, 2.8367],
                        [4.4903, 1.5271],
                        [1.4912, 0.0623],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [-0.2722, -3.5461],
                        [-0.5495, -6.6514],
                        [-5.0807, -6.789],
                        [-5.1042, -5.7177],
                        [-0.8327, 0.0247],
                    ],
                    vec![
                        [0.0, 0.0],
                        [-0.2722, -3.5461],
                        [-3.5029, -3.565],
                        [-0.8327, 0.0247],
                    ],
                ],
            ),
            // sample 8: moving block 50 orient 0, fixed block 24 orient 6
            (
                8,
                vec![
                    vec![
                        [0.0, 0.0],
                        [-5.932, 0.3597],
                        [-11.698, 0.3083],
                        [-17.1996, 0.242],
                        [-11.6806, 3.8562],
                        [-6.1582, 4.3092],
                        [-0.2025, 4.0385],
                    ],
                    vec![
                        [-5.932, 0.3597],
                        [-11.698, 0.3083],
                        [-17.1996, 0.242],
                        [-11.6806, 3.8562],
                        [-6.1582, 4.3092],
                    ],
                ],
                vec![
                    vec![
                        [0.0, 0.0],
                        [5.7866, 0.6434],
                        [5.7361, 4.082],
                        [4.5867, 4.0557],
                        [0.0031, 0.8224],
                    ],
                    vec![
                        [0.0, 0.0],
                        [5.7866, 0.6434],
                        [5.7361, 4.082],
                        [4.5867, 4.0557],
                        [0.0031, 0.8224],
                    ],
                ],
            ),
            // sample 9: moving block 87 orient 7, fixed block 24 orient 6
            (
                9,
                vec![vec![
                    [0.0, 0.0],
                    [2.5405, 2.1002],
                    [5.2756, -0.2082],
                    [7.2288, -2.7494],
                    [10.0017, -5.23],
                    [12.8006, -7.6544],
                    [12.0478, -7.5989],
                    [1.2214, -1.1249],
                ]],
                vec![
                    vec![
                        [0.0, 0.0],
                        [5.7866, 0.6434],
                        [5.7361, 4.082],
                        [4.5867, 4.0557],
                        [0.0031, 0.8224],
                    ],
                    vec![
                        [0.0, 0.0],
                        [5.7866, 0.6434],
                        [5.7361, 4.082],
                        [4.5867, 4.0557],
                        [0.0031, 0.8224],
                    ],
                ],
            ),
        ]
    }

    #[test]
    fn rasterized_grid_matches_geo_for_prob1_embedded_samples() {
        for (case_id, moving_layers, fixed_layers) in prob1_sample_orientation_pairs() {
            let moving = build_shape_geom(&Orientation {
                layers: moving_layers,
            });
            let fixed = build_shape_geom(&Orientation {
                layers: fixed_layers,
            });
            let range = delta_range(moving.bbox, fixed.bbox);
            let grid = build_crane_grid(&moving, &fixed, range);

            for dx in range.min_dx..=range.max_dx {
                for dy in range.min_dy..=range.max_dy {
                    let exact = crane_collision_direct_geo(&moving, &fixed, dx, dy);
                    assert_eq!(
                        grid.get(dx, dy),
                        exact,
                        "prob1 sample {case_id} mismatch at dx={dx}, dy={dy}"
                    );
                }
            }
        }
    }

    #[test]
    fn collision_precompute_crane_matches_geo_for_prob1_embedded_samples() {
        for (case_id, moving_layers, fixed_layers) in prob1_sample_orientation_pairs() {
            let moving = build_shape_geom(&Orientation {
                layers: moving_layers.clone(),
            });
            let fixed = build_shape_geom(&Orientation {
                layers: fixed_layers.clone(),
            });
            let problem = problem(
                vec![Bay {
                    width: 1000,
                    height: 1000,
                }],
                vec![
                    test_block(0, 10, moving_layers),
                    test_block(0, 10, fixed_layers),
                ],
            );
            let pre = CollisionPrecompute::build(&problem, 0);

            assert_precompute_matches_geo(&pre, case_id, &moving, &fixed, 0, 1, "forward");
            assert_precompute_matches_geo(&pre, case_id, &fixed, &moving, 1, 0, "reverse");
        }
    }

    fn assert_precompute_matches_geo(
        pre: &CollisionPrecompute,
        case_id: usize,
        moving: &ShapeGeom,
        fixed: &ShapeGeom,
        moving_block_id: usize,
        fixed_block_id: usize,
        label: &str,
    ) {
        let range = delta_range(moving.bbox, fixed.bbox);
        for dx in range.min_dx..=range.max_dx {
            for dy in range.min_dy..=range.max_dy {
                let exact = crane_collision_direct_geo(moving, fixed, dx, dy);
                let expected = if exact {
                    CollisionResult::Hit
                } else {
                    CollisionResult::Clear
                };
                assert_eq!(
                    pre.crane(
                        BlockPlacement {
                            block_id: moving_block_id,
                            orient_idx: 0,
                            x: 0,
                            y: 0,
                        },
                        BlockPlacement {
                            block_id: fixed_block_id,
                            orient_idx: 0,
                            x: dx,
                            y: dy,
                        },
                    ),
                    expected,
                    "prob1 sample {case_id} {label} mismatch at dx={dx}, dy={dy}"
                );
            }
        }
    }

    fn crane_collision_direct_geo(moving: &ShapeGeom, fixed: &ShapeGeom, dx: i64, dy: i64) -> bool {
        for k in 0..moving.layers.len() {
            for j in k..fixed.layers.len() {
                let a = &moving.layers[k];
                let b = &fixed.layers[j];
                if !bbox_may_overlap(a.bbox, b.bbox, dx, dy) {
                    continue;
                }
                let shifted_b = translate_polygon(&b.polygon, dx, dy);
                if polygons_overlap_area_positive(&a.polygon, &shifted_b) {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn fit_range_uses_bbox_and_integer_bounds() {
        let problem = problem(
            vec![Bay {
                width: 10,
                height: 8,
            }],
            vec![test_block(0, 10, vec![rect(-0.2, -0.7, 2.3, 3.1)])],
        );
        let pre = CollisionPrecompute::build(&problem, 0);

        assert_eq!(
            pre.fit_range(0, 0, 0),
            Some(FitRange {
                min_x: 1,
                max_x: 7,
                min_y: 1,
                max_y: 4,
            })
        );
    }

    #[test]
    fn fit_range_is_none_when_shape_cannot_fit() {
        let problem = problem(
            vec![Bay {
                width: 3,
                height: 3,
            }],
            vec![test_block(0, 10, vec![rect(0.0, 0.0, 4.0, 1.0)])],
        );
        let pre = CollisionPrecompute::build(&problem, 0);

        assert_eq!(pre.fit_range(0, 0, 0), None);
    }

    #[test]
    fn crane_distinguishes_overlap_from_edge_touch() {
        let problem = problem(
            vec![Bay {
                width: 10,
                height: 10,
            }],
            vec![
                test_block(0, 10, vec![rect(0.0, 0.0, 2.0, 2.0)]),
                test_block(0, 10, vec![rect(0.0, 0.0, 2.0, 2.0)]),
            ],
        );
        let pre = CollisionPrecompute::build(&problem, 0);

        assert_eq!(
            pre.crane(placement(0, 0, 0), placement(1, 2, 0)),
            CollisionResult::Clear
        );
        assert_eq!(
            pre.crane(placement(0, 0, 0), placement(1, 1, 0)),
            CollisionResult::Hit
        );
    }

    #[test]
    fn crane_uses_directed_layer_rule() {
        let problem = problem(
            vec![Bay {
                width: 10,
                height: 10,
            }],
            vec![
                test_block(0, 10, vec![rect(0.0, 0.0, 2.0, 2.0)]),
                test_block(
                    0,
                    10,
                    vec![rect(5.0, 5.0, 6.0, 6.0), rect(0.0, 0.0, 2.0, 2.0)],
                ),
            ],
        );
        let pre = CollisionPrecompute::build(&problem, 0);

        assert_eq!(
            pre.crane(placement(0, 0, 0), placement(1, 0, 0)),
            CollisionResult::Hit
        );
        assert_eq!(
            pre.crane(placement(1, 0, 0), placement(0, 0, 0)),
            CollisionResult::Clear
        );
    }

    #[test]
    fn crane_uses_fixed_minus_moving_delta() {
        let problem = problem(
            vec![Bay {
                width: 10,
                height: 10,
            }],
            vec![
                test_block(0, 10, vec![rect(4.0, 0.0, 6.0, 2.0)]),
                test_block(0, 10, vec![rect(0.0, 0.0, 2.0, 2.0)]),
            ],
        );
        let pre = CollisionPrecompute::build(&problem, 0);

        assert_eq!(
            pre.crane(placement(0, 0, 0), placement(1, 4, 0)),
            CollisionResult::Hit
        );
        assert_eq!(
            pre.crane(placement(0, 0, 0), placement(1, -4, 0)),
            CollisionResult::Clear
        );
    }

    #[test]
    fn crane_returns_not_precomputed_for_disjoint_time_windows() {
        let problem = problem(
            vec![Bay {
                width: 10,
                height: 10,
            }],
            vec![
                test_block(0, 10, vec![rect(0.0, 0.0, 2.0, 2.0)]),
                test_block(20, 30, vec![rect(0.0, 0.0, 2.0, 2.0)]),
            ],
        );
        let pre = CollisionPrecompute::build(&problem, 0);

        assert_eq!(
            pre.crane(placement(0, 0, 0), placement(1, 0, 0)),
            CollisionResult::NotPrecomputed
        );
        assert_eq!(
            pre.crane(placement(1, 0, 0), placement(0, 0, 0)),
            CollisionResult::NotPrecomputed
        );
    }

    #[test]
    fn touching_time_windows_are_precomputed() {
        let problem = problem(
            vec![Bay {
                width: 10,
                height: 10,
            }],
            vec![
                test_block(0, 10, vec![rect(0.0, 0.0, 2.0, 2.0)]),
                test_block(10, 20, vec![rect(0.0, 0.0, 2.0, 2.0)]),
            ],
        );
        let pre = CollisionPrecompute::build(&problem, 0);

        assert_eq!(
            pre.crane(placement(0, 0, 0), placement(1, 0, 0)),
            CollisionResult::Hit
        );
    }
}
