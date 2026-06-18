//! # CollisionPrecompute 実装方針
//!
//! この文書は、OGC 2026 solver の Rust 側で使う幾何前計算 `CollisionPrecompute` の
//! 実装仕様を固定するための設計メモである。
//!
//! 他の agent / 実装者がこの文書だけを読んでも同じ設計で実装できるように、
//! API、内部構造、判定仕様、`utils.py` との対応を明示する。
//!
//! ## 目的
//!
//! 前計算で高速化したい判定は次の 2 つである。
//!
//! 1. `ENTRY` / `EXIT` 時のクレーン干渉判定
//! 2. block が bay 内に収まるかの境界判定
//!
//! 通常の同時占有衝突判定、いわゆる `stay` table は当面実装しない。
//!
//! 理由は、公式 checker の `check_entry()` / `check_exit()` が `j >= k` の
//! layer pair を全て見ており、その中に `j == k` の same-layer collision も含まれるためである。
//! solver がすべての `ENTRY` を `pre.crane(new, existing)` で検査するなら、
//! `stay` 制約はその時点で満たされる。
//!
//! したがって、まず実装する API は以下だけでよい。
//!
//! ```rust,ignore
//! pub enum CollisionResult {
//!     Hit,
//!     Clear,
//!     NotPrecomputed,
//! }
//!
//! pub struct BlockPlacement {
//!     pub block_id: usize,
//!     pub orient_idx: usize,
//!     pub x: i64,
//!     pub y: i64,
//! }
//!
//! pub struct FitRange {
//!     pub min_x: i64,
//!     pub max_x: i64,
//!     pub min_y: i64,
//!     pub max_y: i64,
//! }
//!
//! pub struct CollisionPrecompute { /* private fields */ }
//!
//! impl CollisionPrecompute {
//!     pub fn new(problem: &Problem, c: i64) -> Self;
//!
//!     pub fn crane(
//!         &self,
//!         moving: BlockPlacement,
//!         fixed: BlockPlacement,
//!     ) -> CollisionResult;
//!
//!     pub fn fits_in_bay(
//!         &self,
//!         bay_id: usize,
//!         placement: BlockPlacement,
//!     ) -> bool;
//!
//!     pub fn fit_range(
//!         &self,
//!         bay_id: usize,
//!         block_id: usize,
//!         orient_idx: usize,
//!     ) -> Option<FitRange>;
//! }
//! ```
//!
//! ## Cargo dependency
//!
//! Polygon intersection は Rust の `geo` crate で行う。
//!
//! `Cargo.toml` に次を追加する想定。
//!
//! ```toml
//! geo = "0.30"
//! ```
//!
//! `geo` の boolean operation で intersection を作り、面積が正なら衝突とする。
//!
//! ```rust,ignore
//! use geo::{Area, BooleanOps, Coord, LineString, Polygon};
//! ```
//!
//! ## checker との対応
//!
//! ### `check_entry()` / `check_exit()`
//!
//! `ogc2026/alg_tester/utils.py` の `check_entry()` は、既存 block `exist` に対して、
//! new block の layer `k` と existing block の layer `j` のうち `j >= k` を全て見る。
//!
//! ```python
//! for k in range(n_new):
//!     for j in range(k, n_exist):
//!         # polygon intersection area > 0 なら obstruction
//! ```
//!
//! `check_exit()` も同じ rule である。
//!
//! したがって Rust 側の `pre.crane(moving, fixed)` は次を意味する。
//!
//! ```text
//! moving を ENTRY/EXIT で垂直移動するとき、fixed が邪魔になるか。
//! moving.layer[k] と fixed.layer[j] のうち j >= k を全て調べ、
//! どれかの intersection area が正なら Hit。
//! ```
//!
//! この判定は directed であり、一般に次は同じではない。
//!
//! ```text
//! crane(A, B)
//! crane(B, A)
//! ```
//!
//! ### `Bay.contains_block()`
//!
//! `utils.py` の `Bay.contains_block()` は、block の polygon containment ではなく、
//! block の axis-aligned bounding rectangle を使う。
//!
//! ```python
//! bb = block.bounding_rect()
//! return (
//!     bb[0] >= 0 and bb[1] >= 0
//!     and bb[2] <= self.width and bb[3] <= self.height
//! )
//! ```
//!
//! そのため Rust 側の `fits_in_bay()` も、各 `(block, orient)` の全 layer bbox から
//! 求めた整数配置範囲 `FitRange` だけで正確に判定できる。
//!
//! ## 座標・差分の定義
//!
//! `BlockPlacement` の `x, y` は solution に出力する整数座標と同じ意味である。
//!
//! `crane(moving, fixed)` 内部の table lookup では、差分を必ず次で定義する。
//!
//! ```rust,ignore
//! let dx = fixed.x - moving.x;
//! let dy = fixed.y - moving.y;
//! ```
//!
//! つまり table は次の状況を表す。
//!
//! ```text
//! moving block を (0, 0) に置く。
//! fixed block を (dx, dy) に置く。
//! ```
//!
//! 実装中にこの向きを反転してはいけない。
//!
//! ## public types
//!
//! ### `CollisionResult`
//!
//! ```rust,ignore
//! #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//! pub enum CollisionResult {
//!     Hit,
//!     Clear,
//!     NotPrecomputed,
//! }
//! ```
//!
//! 意味は以下。
//!
//! - `Hit`: 前計算済みで、crane obstruction がある。
//! - `Clear`: 前計算済みで、crane obstruction がない。
//! - `NotPrecomputed`: 時間窓 filter により、この block pair の table が存在しない。
//!
//! `NotPrecomputed` を `Clear` と同一視してはいけない。
//! solver 側では最初は安全に「不可」と扱う。
//!
//! ```rust,ignore
//! match pre.crane(moving, fixed) {
//!     CollisionResult::Clear => {}
//!     CollisionResult::Hit | CollisionResult::NotPrecomputed => return false,
//! }
//! ```
//!
//! ### `BlockPlacement`
//!
//! ```rust,ignore
//! #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//! pub struct BlockPlacement {
//!     pub block_id: usize,
//!     pub orient_idx: usize,
//!     pub x: i64,
//!     pub y: i64,
//! }
//! ```
//!
//! bay id は入れない。collision 判定は同じ bay 内の block 同士に対して呼ぶ前提であり、
//! bay id は `fits_in_bay()` の引数でのみ必要になる。
//!
//! ### `FitRange`
//!
//! ```rust,ignore
//! #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//! pub struct FitRange {
//!     pub min_x: i64,
//!     pub max_x: i64,
//!     pub min_y: i64,
//!     pub max_y: i64,
//! }
//! ```
//!
//! `min_x <= x <= max_x` かつ `min_y <= y <= max_y` なら、その `(bay, block, orient)` は
//! bay 内に収まる。
//!
//! `fit_range()` が `None` の場合、その bay にはその block/orient をどこにも置けない。
//!
//! ## private internal types
//!
//! 実装では以下のような private type を使う。
//!
//! ```rust,ignore
//! #[derive(Clone, Copy, Debug)]
//! struct BBox {
//!     min_x: f64,
//!     min_y: f64,
//!     max_x: f64,
//!     max_y: f64,
//! }
//!
//! #[derive(Clone, Debug)]
//! struct PolyLayer {
//!     polygon: Polygon<f64>,
//!     bbox: BBox,
//! }
//!
//! #[derive(Clone, Debug)]
//! struct ShapeGeom {
//!     layers: Vec<PolyLayer>,
//!     bbox: BBox,
//! }
//!
//! #[derive(Clone, Copy, Debug)]
//! struct DeltaRange {
//!     min_dx: i64,
//!     max_dx: i64,
//!     min_dy: i64,
//!     max_dy: i64,
//! }
//!
//! #[derive(Clone, Debug)]
//! struct CollisionGrid {
//!     min_dx: i64,
//!     max_dx: i64,
//!     min_dy: i64,
//!     max_dy: i64,
//!     data: Vec<bool>,
//! }
//!
//! #[derive(Clone, Debug)]
//! struct OrientPairCollision {
//!     moving_orient: usize,
//!     fixed_orient: usize,
//!     crane: CollisionGrid,
//! }
//!
//! #[derive(Clone, Debug)]
//! struct BlockPairCollision {
//!     moving_block: usize,
//!     fixed_block: usize,
//!     moving_orients: usize,
//!     fixed_orients: usize,
//!     orient_pair_index: Vec<Option<usize>>,
//!     orient_pairs: Vec<OrientPairCollision>,
//! }
//! ```
//!
//! `CollisionPrecompute` 本体は次の構成を推奨する。
//!
//! ```rust,ignore
//! pub struct CollisionPrecompute {
//!     geoms: Vec<Vec<ShapeGeom>>,
//!     fit_ranges: Vec<Vec<Vec<Option<FitRange>>>>,
//!     block_pair_index: Vec<Option<usize>>,
//!     block_pairs: Vec<BlockPairCollision>,
//!     n: usize,
//!     c: i64,
//! }
//! ```
//!
//! `block_pair_index` は directed pair 用で、添字は必ず次にする。
//!
//! ```rust,ignore
//! let key = moving_block * self.n + fixed_block;
//! ```
//!
//! `orient_pair_index` の添字は必ず次にする。
//!
//! ```rust,ignore
//! let key = moving_orient * fixed_orients + fixed_orient;
//! ```
//!
//! ## time-window filter
//!
//! 全 block pair に対して table を作ると重いので、時間窓が重なる pair だけ作る。
//!
//! block `i` の前計算対象 window は次とする。
//!
//! ```text
//! [release_time_i - C, due_date_i + C]
//! ```
//!
//! 下限は 0 に丸める。
//!
//! ```rust,ignore
//! fn time_window(block: &Block, c: i64) -> (i64, i64) {
//!     ((block.release_time - c).max(0), block.due_date + c)
//! }
//! ```
//!
//! window overlap は広めに閉区間として扱う。
//!
//! ```rust,ignore
//! fn windows_overlap(a: (i64, i64), b: (i64, i64)) -> bool {
//!     a.0 <= b.1 && b.0 <= a.1
//! }
//! ```
//!
//! unordered pair `{i, j}` の window が重なる場合、directed に両方作る。
//!
//! ```text
//! i -> j
//! j -> i
//! ```
//!
//! window が重ならない pair は `block_pair_index[i * n + j] == None` になり、
//! `pre.crane()` は `CollisionResult::NotPrecomputed` を返す。
//!
//! ## geometry build
//!
//! 各 `Orientation` の各 layer を `geo::Polygon<f64>` に変換する。
//!
//! 入力 layer は `Vec<[f64; 2]>` である。`geo::Polygon` の exterior ring として使うため、
//! 先頭点と末尾点が異なる場合は、末尾に先頭点を追加して閉じる。
//!
//! ```rust,ignore
//! fn build_shape_geom(orientation: &Orientation) -> ShapeGeom {
//!     let mut layers = Vec::new();
//!     let mut all_bbox: Option<BBox> = None;
//!
//!     for layer in &orientation.layers {
//!         let mut coords: Vec<Coord<f64>> = layer
//!             .iter()
//!             .map(|&[x, y]| Coord { x, y })
//!             .collect();
//!
//!         if coords.first() != coords.last() {
//!             if let Some(first) = coords.first().copied() {
//!                 coords.push(first);
//!             }
//!         }
//!
//!         let polygon = Polygon::new(LineString::from(coords), vec![]);
//!         let bbox = bbox_of_points(layer);
//!         all_bbox = Some(match all_bbox {
//!             Some(b) => merge_bbox(b, bbox),
//!             None => bbox,
//!         });
//!         layers.push(PolyLayer { polygon, bbox });
//!     }
//!
//!     ShapeGeom { layers, bbox: all_bbox.unwrap() }
//! }
//! ```
//!
//! `ShapeGeom::bbox` はその orientation の全 layer を含む bbox である。
//!
//! ## fit range build
//!
//! bay は `[0, W] x [0, H]` の軸平行長方形である。
//! `utils.py` も bbox containment で判定しているため、`FitRange` は bbox から求める。
//!
//! orientation bbox を `[min_x, min_y, max_x, max_y]` とする。
//! block を整数位置 `(x, y)` に置くと bbox は次になる。
//!
//! ```text
//! [x + min_x, y + min_y, x + max_x, y + max_y]
//! ```
//!
//! bay 内に収まる条件は次。
//!
//! ```text
//! x + min_x >= 0
//! y + min_y >= 0
//! x + max_x <= W
//! y + max_y <= H
//! ```
//!
//! よって整数配置範囲は次。
//!
//! ```text
//! ceil(-min_x) <= x <= floor(W - max_x)
//! ceil(-min_y) <= y <= floor(H - max_y)
//! ```
//!
//! 実装例。
//!
//! ```rust,ignore
//! fn build_fit_range(bay: &Bay, bbox: BBox) -> Option<FitRange> {
//!     let min_x = (-bbox.min_x).ceil() as i64;
//!     let max_x = (bay.width as f64 - bbox.max_x).floor() as i64;
//!     let min_y = (-bbox.min_y).ceil() as i64;
//!     let max_y = (bay.height as f64 - bbox.max_y).floor() as i64;
//!
//!     if min_x <= max_x && min_y <= max_y {
//!         Some(FitRange { min_x, max_x, min_y, max_y })
//!     } else {
//!         None
//!     }
//! }
//! ```
//!
//! `fits_in_bay()` はこの range を見るだけにする。
//!
//! ```rust,ignore
//! pub fn fits_in_bay(&self, bay_id: usize, placement: BlockPlacement) -> bool {
//!     let Some(range) = self.fit_range(
//!         bay_id,
//!         placement.block_id,
//!         placement.orient_idx,
//!     ) else {
//!         return false;
//!     };
//!
//!     range.min_x <= placement.x
//!         && placement.x <= range.max_x
//!         && range.min_y <= placement.y
//!         && placement.y <= range.max_y
//! }
//! ```
//!
//! 全 `(x, y)` の bool table は作らない。`FitRange` で完全に表現できる。
//!
//! ## collision grid build
//!
//! `CollisionGrid` は各 directed `(moving_block, fixed_block, moving_orient, fixed_orient)` ごとに
//! 1 個作る。
//!
//! grid が表す座標は次。
//!
//! ```text
//! moving at (0, 0)
//! fixed  at (dx, dy)
//! ```
//!
//! bbox から、衝突し得る `dx, dy` の範囲だけを持つ。
//!
//! ```rust,ignore
//! fn delta_range(moving: BBox, fixed: BBox) -> DeltaRange {
//!     DeltaRange {
//!         min_dx: (moving.min_x - fixed.max_x).floor() as i64 - 1,
//!         max_dx: (moving.max_x - fixed.min_x).ceil() as i64 + 1,
//!         min_dy: (moving.min_y - fixed.max_y).floor() as i64 - 1,
//!         max_dy: (moving.max_y - fixed.min_y).ceil() as i64 + 1,
//!     }
//! }
//! ```
//!
//! 範囲外は必ず `Clear` とみなせるため、`CollisionGrid::get()` は範囲外なら `false` を返してよい。
//! ただし、block pair 自体が未前計算の場合は `CollisionResult::NotPrecomputed` を返す。
//!
//! `CollisionGrid` の data index は次で固定する。
//!
//! ```rust,ignore
//! impl CollisionGrid {
//!     fn width(&self) -> usize {
//!         (self.max_dx - self.min_dx + 1) as usize
//!     }
//!
//!     fn idx(&self, dx: i64, dy: i64) -> Option<usize> {
//!         if dx < self.min_dx || self.max_dx < dx {
//!             return None;
//!         }
//!         if dy < self.min_dy || self.max_dy < dy {
//!             return None;
//!         }
//!         let x = (dx - self.min_dx) as usize;
//!         let y = (dy - self.min_dy) as usize;
//!         Some(y * self.width() + x)
//!     }
//!
//!     fn get(&self, dx: i64, dy: i64) -> bool {
//!         self.idx(dx, dy).is_some_and(|idx| self.data[idx])
//!     }
//! }
//! ```
//!
//! 最初は `Vec<bool>` でよい。必要になったら `Vec<u64>` bitset に置き換える。
//!
//! ## direct crane collision
//!
//! `geo` で polygon intersection を取り、面積が正なら hit とする。
//!
//! 辺接触は衝突ではないため、`intersects()` だけで判定してはいけない。
//!
//! ```rust,ignore
//! const AREA_EPS: f64 = 1e-10;
//!
//! fn polygons_overlap_area_positive(a: &Polygon<f64>, b: &Polygon<f64>) -> bool {
//!     let inter = a.intersection(b);
//!     inter.unsigned_area() > AREA_EPS
//! }
//! ```
//!
//! layer bbox で先に弾く。
//!
//! ```rust,ignore
//! fn bbox_may_overlap(a: BBox, b: BBox, dx: i64, dy: i64) -> bool {
//!     let dx = dx as f64;
//!     let dy = dy as f64;
//!
//!     let b_min_x = b.min_x + dx;
//!     let b_max_x = b.max_x + dx;
//!     let b_min_y = b.min_y + dy;
//!     let b_max_y = b.max_y + dy;
//!
//!     a.min_x < b_max_x - AREA_EPS
//!         && b_min_x < a.max_x - AREA_EPS
//!         && a.min_y < b_max_y - AREA_EPS
//!         && b_min_y < a.max_y - AREA_EPS
//! }
//! ```
//!
//! fixed polygon は `(dx, dy)` だけ平行移動して比較する。
//!
//! ```rust,ignore
//! fn translate_polygon(poly: &Polygon<f64>, dx: i64, dy: i64) -> Polygon<f64> {
//!     let dx = dx as f64;
//!     let dy = dy as f64;
//!     let exterior: Vec<Coord<f64>> = poly
//!         .exterior()
//!         .points()
//!         .map(|p| Coord { x: p.x() + dx, y: p.y() + dy })
//!         .collect();
//!     Polygon::new(LineString::from(exterior), vec![])
//! }
//! ```
//!
//! directed crane collision は次。
//!
//! ```rust,ignore
//! fn crane_collision_direct(
//!     moving: &ShapeGeom,
//!     fixed: &ShapeGeom,
//!     dx: i64,
//!     dy: i64,
//! ) -> bool {
//!     for k in 0..moving.layers.len() {
//!         for j in k..fixed.layers.len() {
//!             let a = &moving.layers[k];
//!             let b = &fixed.layers[j];
//!
//!             if !bbox_may_overlap(a.bbox, b.bbox, dx, dy) {
//!                 continue;
//!             }
//!
//!             let shifted_b = translate_polygon(&b.polygon, dx, dy);
//!             if polygons_overlap_area_positive(&a.polygon, &shifted_b) {
//!                 return true;
//!             }
//!         }
//!     }
//!     false
//! }
//! ```
//!
//! ## `CollisionPrecompute::new` の処理手順
//!
//! 実装順は次で固定する。
//!
//! 1. `problem.blocks[block_id].shape[orient_idx]` から `geoms[block_id][orient_idx]` を作る。
//! 2. `fit_ranges[bay_id][block_id][orient_idx]` を全 bay / block / orientation について作る。
//! 3. 全 block の time window を作る。
//! 4. `i < j` の unordered block pair を走査し、window が重なる場合だけ directed pair を 2 個作る。
//!    - `i -> j`
//!    - `j -> i`
//! 5. directed pair ごとに全 orientation pair の `CollisionGrid` を作る。
//! 6. `block_pair_index[moving * n + fixed]` に `block_pairs` の index を入れる。
//!
//! 疑似コード。
//!
//! ```rust,ignore
//! pub fn new(problem: &Problem, c: i64) -> Self {
//!     let n = problem.blocks.len();
//!     let geoms = build_all_geoms(problem);
//!     let fit_ranges = build_all_fit_ranges(problem, &geoms);
//!
//!     let windows: Vec<(i64, i64)> = problem.blocks
//!         .iter()
//!         .map(|b| time_window(b, c))
//!         .collect();
//!
//!     let mut block_pair_index = vec![None; n * n];
//!     let mut block_pairs = Vec::new();
//!
//!     for i in 0..n {
//!         for j in (i + 1)..n {
//!             if !windows_overlap(windows[i], windows[j]) {
//!                 continue;
//!             }
//!
//!             let ij = block_pairs.len();
//!             block_pairs.push(build_directed_block_pair(&geoms, i, j));
//!             block_pair_index[i * n + j] = Some(ij);
//!
//!             let ji = block_pairs.len();
//!             block_pairs.push(build_directed_block_pair(&geoms, j, i));
//!             block_pair_index[j * n + i] = Some(ji);
//!         }
//!     }
//!
//!     Self { geoms, fit_ranges, block_pair_index, block_pairs, n, c }
//! }
//! ```
//!
//! `moving_block == fixed_block` の table は作らない。
//!
//! ## `build_directed_block_pair`
//!
//! ```rust,ignore
//! fn build_directed_block_pair(
//!     geoms: &[Vec<ShapeGeom>],
//!     moving_block: usize,
//!     fixed_block: usize,
//! ) -> BlockPairCollision {
//!     let moving_orients = geoms[moving_block].len();
//!     let fixed_orients = geoms[fixed_block].len();
//!
//!     let mut orient_pair_index = vec![None; moving_orients * fixed_orients];
//!     let mut orient_pairs = Vec::new();
//!
//!     for moving_orient in 0..moving_orients {
//!         for fixed_orient in 0..fixed_orients {
//!             let moving = &geoms[moving_block][moving_orient];
//!             let fixed = &geoms[fixed_block][fixed_orient];
//!             let range = delta_range(moving.bbox, fixed.bbox);
//!             let mut crane = CollisionGrid::new(range);
//!
//!             for dx in range.min_dx..=range.max_dx {
//!                 for dy in range.min_dy..=range.max_dy {
//!                     let hit = crane_collision_direct(moving, fixed, dx, dy);
//!                     crane.set(dx, dy, hit);
//!                 }
//!             }
//!
//!             let idx = orient_pairs.len();
//!             orient_pair_index[moving_orient * fixed_orients + fixed_orient] = Some(idx);
//!             orient_pairs.push(OrientPairCollision { moving_orient, fixed_orient, crane });
//!         }
//!     }
//!
//!     BlockPairCollision {
//!         moving_block,
//!         fixed_block,
//!         moving_orients,
//!         fixed_orients,
//!         orient_pair_index,
//!         orient_pairs,
//!     }
//! }
//! ```
//!
//! ## `crane()` query
//!
//! `crane()` は以下の順に処理する。
//!
//! 1. `moving.block_id == fixed.block_id` なら通常は `Clear` でよい。
//!    - 呼び出し側が self pair を渡さない設計なら `debug_assert_ne!` でもよい。
//! 2. `block_pair_index[moving.block_id * n + fixed.block_id]` を見る。
//! 3. `None` なら `NotPrecomputed`。
//! 4. orientation pair table を見る。
//! 5. `dx = fixed.x - moving.x`, `dy = fixed.y - moving.y` で grid を引く。
//! 6. `true` なら `Hit`、`false` なら `Clear`。
//!
//! ```rust,ignore
//! pub fn crane(&self, moving: BlockPlacement, fixed: BlockPlacement) -> CollisionResult {
//!     if moving.block_id == fixed.block_id {
//!         return CollisionResult::Clear;
//!     }
//!
//!     let Some(pair_idx) = self.block_pair_index[moving.block_id * self.n + fixed.block_id] else {
//!         return CollisionResult::NotPrecomputed;
//!     };
//!
//!     let pair = &self.block_pairs[pair_idx];
//!     let key = moving.orient_idx * pair.fixed_orients + fixed.orient_idx;
//!     let Some(orient_pair_idx) = pair.orient_pair_index[key] else {
//!         return CollisionResult::NotPrecomputed;
//!     };
//!
//!     let orient_pair = &pair.orient_pairs[orient_pair_idx];
//!     let dx = fixed.x - moving.x;
//!     let dy = fixed.y - moving.y;
//!
//!     if orient_pair.crane.get(dx, dy) {
//!         CollisionResult::Hit
//!     } else {
//!         CollisionResult::Clear
//!     }
//! }
//! ```
//!
//! ## solver 側の使い方
//!
//! ### ENTRY
//!
//! ```rust,ignore
//! fn can_entry(
//!     pre: &CollisionPrecompute,
//!     bay_id: usize,
//!     moving: BlockPlacement,
//!     existing: &[BlockPlacement],
//! ) -> bool {
//!     if !pre.fits_in_bay(bay_id, moving) {
//!         return false;
//!     }
//!
//!     for &fixed in existing {
//!         match pre.crane(moving, fixed) {
//!             CollisionResult::Clear => {}
//!             CollisionResult::Hit | CollisionResult::NotPrecomputed => return false,
//!         }
//!     }
//!     true
//! }
//! ```
//!
//! ### EXIT
//!
//! ```rust,ignore
//! fn can_exit(
//!     pre: &CollisionPrecompute,
//!     moving: BlockPlacement,
//!     existing: &[BlockPlacement],
//! ) -> bool {
//!     for &fixed in existing {
//!         if fixed.block_id == moving.block_id {
//!             continue;
//!         }
//!         match pre.crane(moving, fixed) {
//!             CollisionResult::Clear => {}
//!             CollisionResult::Hit | CollisionResult::NotPrecomputed => return false,
//!         }
//!     }
//!     true
//! }
//! ```
//!
//! `EXIT` では bay 境界チェックは不要。すでに置かれている block を出すだけだからである。
//!
//! ## `stay` table を作らない理由
//!
//! `check_collisions()` / Stage4 は same-layer collision を見る。
//! しかし `check_entry()` の `j >= k` には `j == k` が含まれる。
//!
//! そのため、solver が時系列に沿って `ENTRY` を処理し、各 `ENTRY` で
//! `pre.crane(new, existing)` を全既存 block に対して確認していれば、
//! same-layer collision は発生しない。
//!
//! block は ENTRY 後に移動・回転しないため、後から same-layer collision が自然発生することもない。
//!
//! したがって当面は `stay` table を実装しない。
//!
//! ただし、将来「ENTRY 順序を考えずに同時存在可能性だけを緩く評価する」探索を入れるなら、
//! `stay` table は有用になり得る。その場合も本 API とは別拡張として追加する。
//!
//! ## 実装上の注意
//!
//! - `NotPrecomputed` は安全側に倒して不可扱いにする。
//! - `crane` は directed。pair を片方向だけに省略してはいけない。
//! - `stay` は実装しない。`can_entry` は `fits_in_bay + crane` のみ。
//! - `fits_in_bay` は全 `(x, y)` table を作らず、`FitRange` だけで判定する。
//! - boundary 判定は `utils.py` に合わせて bbox ベースにする。
//! - polygon 接触だけなら衝突ではない。intersection area が正のときだけ `Hit`。
//! - dense grid の範囲外は `Clear`。ただし block pair table 自体がない場合は `NotPrecomputed`。
//! - 最初は `Vec<bool>` でよい。メモリが問題になってから bitset 化する。
