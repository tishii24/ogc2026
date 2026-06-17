"""
utils.py -- Bay/Block geometry and feasibility checking utilities

===============================================================================
PHYSICAL MODEL
===============================================================================

A *bay* is a rectangular storage area (width * height in integer grid units).
Each *block* occupies a region of the bay for a time interval
[entry_time, exit_time).  Blocks are loaded and unloaded by a vertical
overhead crane that moves exclusively in the vertical (z) direction: the crane
lowers a block straight down into the bay (ENTRY) and lifts it straight up out
of the bay (EXIT).  No horizontal repositioning during crane travel is allowed.

Block shape -- layers:
  Each block shape is described by one or more *layers*, indexed 0, 1, 2, ...
  A layer is a polygon in the 2-D (x, y) footprint plane.  Layer 0 represents
  the lowest physical level of the block; higher layer indices correspond to
  higher physical levels.  Multiple orientations of the same block are stored
  as separate entries in the "shape" list; orient_idx selects which one to use.

Crane-path geometry -- the j >= k rule:
  When the crane lowers a block, every layer k of the new block sweeps through
  all heights above its final resting level before coming to rest.  At descent
  offset d the new-block layer k occupies the same absolute height as
  existing-block layer (k + d).  Setting j = k + d gives j >= k.  Therefore:

    j == k : collision at the *final resting position* -- the two layers
             occupy the same height at the same time.
    j  > k : collision along the *descent path* -- new-block layer k passes
             through the height of existing-block layer j while still
             being lowered (sweep collision).

  The same j >= k rule applies symmetrically to crane exit (ascending motion).
  The condition j < k never causes a collision because in that case the
  existing layer j is *below* the descending new layer k and is never reached.

  Asymmetric layer counts:
    If the new block has more layers than the existing block (n_new > n_exist),
    the new block's upper layers (k >= n_exist) have no existing layer at or
    above them, so no collision check is needed for those layers.
    If the existing block has more layers (n_exist > n_new), those upper
    existing layers (j >= n_new) are fully checked for every new layer k,
    because the new block passes through those heights during descent.

===============================================================================
MAIN CLASSES
===============================================================================

Bay              : Rectangular bay dimensions and boundary helpers.
Block            : One block placed at (x, y) with a given orientation.
                   World-coordinate layer polygons are cached at construction.
CollisionResult  : Spatial overlap between two co-present blocks at one layer.
EntryObstruction : Crane-path obstruction when inserting or removing a block
                   (covers both final-position and sweep collisions).

===============================================================================
MAIN FUNCTIONS
===============================================================================

check_collisions(bay, blocks) -> list[CollisionResult]
    Spatial overlap check among all block pairs.  Uses AABB early-exit before
    full Shapely intersection.  Only layer k of block A vs layer k of block B
    is compared (same-height model).  Returns [] if no collisions.

check_entry(bay, blocks, new_block) -> list[EntryObstruction]
    Crane-entry feasibility: can the crane lower new_block into the bay without
    hitting any block in `blocks`?  Applies the j >= k descent-path rule.
    Also checks that new_block fits within the bay boundary.
    Returns [] if entry is feasible.

check_exit(bay, blocks, target_block) -> list[EntryObstruction]
    Crane-exit feasibility: can the crane lift target_block out of the bay
    without hitting surrounding blocks?  Same j >= k rule as check_entry.
    Returns [] if exit is feasible.

check_feasibility(prob_info, solution) -> dict
    Validates a complete solver solution through five ordered stages and
    computes the objective value if all stages pass.

===============================================================================
EXAMPLE
===============================================================================

    from utils import Bay, Block, check_collisions

    bay = Bay(width=100, height=20)

    b0 = Block(block_id=0, block_data=instance["blocks"][0], x=0, y=0, orient_idx=0)
    b1 = Block(block_id=1, block_data=instance["blocks"][1], x=5, y=3, orient_idx=0)

    results = check_collisions(bay, [b0, b1])
    for r in results:
        print(r)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


# -----------------------------------------------------------------------------
# Internal geometry helpers
# -----------------------------------------------------------------------------

def _resolve_layers(raw_layers: list) -> list:
    """
    Return a copy of raw_layers as a plain Python list-of-lists.

    Each element of raw_layers is a polygon vertex list [[x,y], ...].
    All layers are expected to be explicitly provided (non-empty).
    The function is kept for interface compatibility; it performs a
    shallow copy so callers cannot accidentally mutate instance data.
    """
    return [list(layer) for layer in raw_layers if layer]


def _anchor_verts(verts: list) -> list:
    """
    Translate a single vertex list so its bounding-box min corner is at (0, 0).

    Subtracts min_x from all x-coordinates and min_y from all y-coordinates.
    Used when anchoring a single layer in isolation (e.g. for bounding-box
    estimation).  When multiple layers must stay aligned to each other use
    _anchor_layers instead.
    """
    if not verts:
        return verts
    min_x = min(v[0] for v in verts)
    min_y = min(v[1] for v in verts)
    return [[x - min_x, y - min_y] for x, y in verts]


def _anchor_layers(layers: list) -> list:
    """
    Translate all layers so that the union bounding-box min corner is at (0, 0).

    The translation is derived from the combined vertex set of all layers, so
    every layer is shifted by the same (min_x, min_y) offset.  This preserves
    the relative positions between layers -- anchoring each layer individually
    would shift them by different amounts and break the inter-layer alignment
    that is required for correct crane-path geometry.

    Returns a new list of layers; the original data is not modified.
    """
    all_verts = [v for l in layers for v in l]
    if not all_verts:
        return [list(l) for l in layers]
    min_x = min(v[0] for v in all_verts)
    min_y = min(v[1] for v in all_verts)
    return [[[x - min_x, y - min_y] for x, y in l] for l in layers]


def _translate_verts(verts: list, dx: float, dy: float) -> list:
    """Shift every vertex in verts by (dx, dy) and return a new list."""
    return [[x + dx, y + dy] for x, y in verts]


@lru_cache(maxsize=8192)
def _poly_from_verts_cached(verts_tuple: tuple) -> Optional[ShapelyPolygon]:
    """
    Build and cache a Shapely Polygon from an immutable tuple of (x, y) pairs.

    The result is memoised by verts_tuple so repeated calls with the same
    vertex sequence (common during iterative crane-path checking) return the
    already-constructed object without re-entering Shapely.

    Validity repair:
      Shapely may produce an invalid polygon for self-touching or nearly
      self-intersecting vertex sequences.  buffer(0) is a standard Shapely
      idiom that rebuilds the geometry into a valid simple polygon (or
      MultiPolygon) without changing the covered area.  If the result is still
      empty (degenerate input such as collinear points) None is returned so
      callers can skip the intersection test.

    Returns None for fewer than 3 vertices or for degenerate geometry.
    """
    if len(verts_tuple) < 3:
        return None
    try:
        p = ShapelyPolygon(verts_tuple)
        if not p.is_valid:
            p = p.buffer(0)
        return p if not p.is_empty else None
    except Exception:
        return None


def _poly_from_verts(verts: list) -> Optional[ShapelyPolygon]:
    """
    Convert a mutable vertex list to a cached Shapely Polygon.

    Converts verts to a hashable tuple-of-tuples so it can be used as the
    cache key for _poly_from_verts_cached.  Returns None for empty or
    degenerate input (fewer than 3 vertices).
    """
    if not verts or len(verts) < 3:
        return None
    return _poly_from_verts_cached(tuple(tuple(v) for v in verts))


def _bounding_box(verts: list) -> tuple[float, float, float, float]:
    """
    Return (min_x, min_y, max_x, max_y) of a vertex list.

    Used to build axis-aligned bounding boxes (AABB) for fast overlap
    pre-screening before the more expensive Shapely intersection tests.
    """
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return min(xs), min(ys), max(xs), max(ys)


# -----------------------------------------------------------------------------
# Bay
# -----------------------------------------------------------------------------

@dataclass
class Bay:
    """
    Dimensions of a single rectangular bay.

    The bay coordinate system has its origin at the bottom-left corner.
    x runs horizontally (0 ... width) and y runs vertically (0 ... height).
    All block positions are expressed in this coordinate system.

    Parameters
    ----------
    width  : Bay width in grid units (positive integer).
    height : Bay height in grid units (positive integer).
    id     : Bay identifier (0-based list index, assigned at load time).
    """
    width: int
    height: int
    id: int = 0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Bay dimensions must be positive integers: width={self.width}, height={self.height}")

    @classmethod
    def from_dict(cls, d: dict, idx: int = 0) -> "Bay":
        """Create a Bay from a bays[i] entry in the instance JSON."""
        return cls(width=int(d["width"]), height=int(d["height"]), id=idx)

    def contains_block(self, block: "Block") -> bool:
        """
        Return True if the block's entire footprint lies within this bay.

        Uses the block's axis-aligned bounding rectangle (which encloses the
        actual polygon) to check 0 <= min_x and max_x <= width, and similarly
        for y.  Because the bounding rect is a superset of the actual polygon,
        any block that passes this check is guaranteed to be fully inside the
        bay.
        """
        bb = block.bounding_rect()
        return (
            bb[0] >= 0 and bb[1] >= 0
            and bb[2] <= self.width and bb[3] <= self.height
        )


# -----------------------------------------------------------------------------
# Block
# -----------------------------------------------------------------------------

@dataclass
class Block:
    """
    A single block placed at position (x, y) in a bay with a chosen orientation.

    Parameters
    ----------
    block_id   : Index into instance["blocks"] -- uniquely identifies the block.
    block_data : The full instance["blocks"][i] dict (shape, timing, preferences).
    x          : x coordinate of the reference point in bay coordinates (integer
                 grid units).  The reference point is the first vertex of the first
                 layer in local coordinates, which is guaranteed
                 to be (0, 0).
    y          : y coordinate of the reference point in bay coordinates (integer
                 grid units).  See x above.
    orient_idx : 0-based index into block_data["shape"] selecting the orientation.

    Caching:
      _layers_cache is computed once in __post_init__ and holds the world-
      coordinate polygon vertex lists for every layer of the chosen orientation.
      This avoids repeated translate calls during the many check_entry / check_exit
      evaluations that happen in Phase 1 and repair.
    """
    block_id:   int
    block_data: dict
    x:          int = 0
    y:          int = 0
    orient_idx: int = 0

    def __post_init__(self):
        # The reference point is the first vertex of the first layer, which the
        # instance generator guarantees to be (0, 0) in local coordinates.
        # Translate all layers by (x, y) so the reference point maps to (x, y)
        # in bay coordinates.  All layers shift by the same offset, preserving
        # their relative positions (inter-layer alignment is unchanged).
        layers = _resolve_layers(self.block_data["shape"][self.orient_idx]["layers"])
        if layers:
            ref_x, ref_y = layers[0][0] if layers[0] else (0.0, 0.0)
            object.__setattr__(self, '_layers_cache',
                               [_translate_verts(l, self.x - ref_x, self.y - ref_y)
                                for l in layers])
        else:
            object.__setattr__(self, '_layers_cache', [])

    # -- Query properties -----------------------------------------------------
    @property
    def orientations(self) -> list:
        """All orientation entries from block_data["shape"]."""
        return self.block_data["shape"]

    @property
    def num_orientations(self) -> int:
        """Number of available orientations for this block."""
        return len(self.orientations)

    @property
    def current_orient(self) -> dict:
        """The shape entry for the currently selected orient_idx."""
        if not (0 <= self.orient_idx < len(self.orientations)):
            raise IndexError(
                f"Block {self.block_id}: orient_idx={self.orient_idx} "
                f"out of range (total {len(self.orientations)} orientations)"
            )
        return self.orientations[self.orient_idx]

    @property
    def orientation_index(self) -> int:
        """The orientation identifier stored in the JSON (shape[i]["orientation"])."""
        return self.current_orient["orientation"]

    # -- Layer computation ----------------------------------------------------
    def resolved_layers(self) -> list[list]:
        """
        Raw layer list for the current orientation (no translation applied).
        Return format: [ [[x, y], ...], [[x, y], ...], ... ]
        """
        return _resolve_layers(self.current_orient["layers"])

    def layers_at_pos(self) -> list[list]:
        """
        World-coordinate layer polygon vertices for this block's current
        (x, y) placement.  Returns the pre-computed _layers_cache built in
        __post_init__; O(1) lookup with no recomputation.

        The vertex lists are in the same format as resolved_layers() but
        translated so that the reference point (first vertex of first layer,
        which is (0, 0) in local coordinates) maps to (x, y) in bay
        coordinates.  All layers share the same translation offset,
        preserving inter-layer alignment.
        """
        return self._layers_cache

    def bounding_rect(self) -> tuple[float, float, float, float]:
        """
        Axis-aligned bounding rectangle of the complete block footprint in
        world coordinates.  Returns (min_x, min_y, max_x, max_y).

        Covers all layers so the returned box is guaranteed to enclose the
        actual polygon at every layer.  Used for fast AABB overlap screening
        in check_collisions, check_entry, and check_exit before the more
        expensive Shapely intersection tests.

        Fallback: if the block has no layers (degenerate shape), returns a
        1*1 bounding box anchored at (x, y) so that AABB tests still work.
        """
        layers = self.layers_at_pos()
        if not layers:
            return (float(self.x), float(self.y),
                    float(self.x) + 1.0, float(self.y) + 1.0)
        all_verts = [v for layer in layers for v in layer]
        return _bounding_box(all_verts)

    # -- Convenience methods --------------------------------------------------
    @classmethod
    def from_instance(cls, block_id: int, instance: dict,
                      x: int = 0, y: int = 0, orient_idx: int = 0) -> "Block":
        """Create a Block directly from an instance JSON dict."""
        return cls(
            block_id=block_id,
            block_data=instance["blocks"][block_id],
            x=x, y=y, orient_idx=orient_idx,
        )

    def __repr__(self) -> str:
        return (f"Block(id={self.block_id}, pos=({self.x},{self.y}), "
                f"orient_idx={self.orientation_index})")


# -----------------------------------------------------------------------------
# CollisionResult
# -----------------------------------------------------------------------------

@dataclass
class CollisionResult:
    """
    Spatial overlap between two blocks at a specific layer.

    Produced by check_collisions when two blocks that share a time interval
    have overlapping footprints at the same layer index.  A single block pair
    may generate multiple CollisionResult records if they overlap at more than
    one layer.

    Attributes
    ----------
    block_a      : First colliding block.
    block_b      : Second colliding block.
    layer_index  : Layer index k at which the overlap occurs (0-based).
                   Layer k of block_a overlaps layer k of block_b.
    intersection : Shapely geometry of the overlapping region.  Can be used
                   to calculate area or to visualise the exact overlap.
    area         : Area of the intersection polygon (auto-computed from
                   intersection in __post_init__).
    """
    block_a:      Block
    block_b:      Block
    layer_index:  int
    intersection: ShapelyPolygon
    area:         float = field(init=False)

    def __post_init__(self):
        self.area = self.intersection.area

    def __repr__(self) -> str:
        return (
            f"CollisionResult("
            f"blocks=({self.block_a.block_id}, {self.block_b.block_id}), "
            f"layer={self.layer_index}, "
            f"area={self.area:.4f})"
        )


# -----------------------------------------------------------------------------
# check_collisions -- core utility function
# -----------------------------------------------------------------------------

def _bb_overlap(a: tuple[float, float, float, float],
                b: tuple[float, float, float, float]) -> bool:
    """
    Return True if two axis-aligned bounding boxes overlap (strict interior).

    Each box is (min_x, min_y, max_x, max_y).  Boxes that share only a single
    edge or corner are *not* considered overlapping (uses strict < comparisons).
    This is used as a fast pre-filter before the more expensive Shapely polygon
    intersection tests.
    """
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def check_collisions(bay: Bay, blocks: list[Block],
                     layer_indices: Optional[set] = None) -> list[CollisionResult]:
    """
    Check for spatial overlaps among all pairs of blocks at each layer.

    Two blocks "collide" when their polygons at the same layer index k have a
    non-zero intersection area.  This represents two objects physically occupying
    the same space at the same height -- a hard constraint violation.

    Algorithm:
      Step 1 -- AABB pre-filter: skip any block pair whose full-footprint
               bounding boxes do not overlap.  This eliminates most pairs in
               O(1) without invoking Shapely.
      Step 2 -- Per-layer Shapely intersection: for pairs that pass Step 1,
               compare layer k of block A against layer k of block B for each
               shared layer index k.  Produces one CollisionResult per (pair,
               layer) combination that overlaps.

    Layer count asymmetry:
      When two blocks have different numbers of layers, only layer indices 0
      through min(n_A, n_B) - 1 are compared.  The additional layers of the
      taller block have no counterpart layer in the shorter block at the same
      height, so no same-height collision is possible for those layers.

    Parameters
    ----------
    bay           : Target bay (used only for its dimensions; not directly
                    accessed in the current implementation but passed for
                    interface consistency).
    blocks        : List of Block instances assumed to be co-present in the bay.
    layer_indices : If provided, only check the specified layer indices.
                    None (default) checks all shared layers.

    Returns
    -------
    list[CollisionResult]
        One entry per (block pair, layer) combination with area > 0.
        Returns [] if no collisions are found.
        If the same pair collides at multiple layers, each layer is a separate
        CollisionResult.
    """
    results: list[CollisionResult] = []
    n = len(blocks)

    # Pre-compute per-block AABB and layer lists (avoid repeated computation)
    bboxes:  list[tuple[float, float, float, float]] = [b.bounding_rect() for b in blocks]
    all_layers: list[list[list]] = [b.layers_at_pos() for b in blocks]

    for i in range(n):
        for j in range(i + 1, n):
            # -- Step 1: Early exit via AABB comparison ----------------------
            if not _bb_overlap(bboxes[i], bboxes[j]):
                continue

            ba = blocks[i]
            bb = blocks[j]
            layers_a = all_layers[i]
            layers_b = all_layers[j]

            # -- Step 2: Per-layer precise intersection check ----------------
            for k in range(min(len(layers_a), len(layers_b))):
                if layer_indices is not None and k not in layer_indices:
                    continue
                poly_a = _poly_from_verts(layers_a[k])
                poly_b = _poly_from_verts(layers_b[k])

                if poly_a is None or poly_b is None:
                    continue

                try:
                    inter = poly_a.intersection(poly_b)
                except Exception:
                    continue

                if not inter.is_empty and inter.area > 0:
                    results.append(CollisionResult(
                        block_a=ba,
                        block_b=bb,
                        layer_index=k,
                        intersection=inter,
                    ))

    return results


# -----------------------------------------------------------------------------
# EntryObstruction / check_entry -- crane entry feasibility check
# -----------------------------------------------------------------------------

@dataclass
class EntryObstruction:
    """
    A single crane-path obstruction encountered during block insertion or removal.

    Produced by check_entry (insertion) and check_exit (removal) when the
    crane's vertical travel path is blocked by an existing block.

    Attributes
    ----------
    existing_block : The block that causes the obstruction.
                     Special case: when existing_block.block_id == new_block.block_id
                     (self-reference), the obstruction indicates that new_block
                     exceeds the bay boundary rather than a collision with another
                     block.  Callers that need to distinguish boundary violations
                     from inter-block collisions should check this condition.
    new_layer      : Layer index k of the block being inserted/removed.
    exist_layer    : Layer index j of existing_block that causes the obstruction.
                     j >= k always holds (see module docstring for the j >= k rule).
    intersection   : Shapely geometry of the overlapping region between the two
                     layer polygons.  Useful for computing overlap area or
                     visualising where the obstruction occurs.
    area           : Area of the intersection polygon (auto-computed from
                     intersection in __post_init__).
    is_sweep       : True when j > k -- the obstruction occurs while the crane
                     is still moving (sweep/descent or ascent collision).
                     False when j == k -- the obstruction is at the final
                     resting position of the new block (same-height collision).
    """
    existing_block: Block
    new_layer:      int
    exist_layer:    int
    intersection:   ShapelyPolygon
    area:           float = field(init=False)

    def __post_init__(self):
        self.area = self.intersection.area

    @property
    def is_sweep(self) -> bool:
        """True if the obstruction is along the crane travel path (j > k), False if at final position (j == k)."""
        return self.exist_layer > self.new_layer

    def __repr__(self) -> str:
        kind = "sweep" if self.is_sweep else "collision"
        return (
            f"EntryObstruction("
            f"existing={self.existing_block.block_id}, "
            f"new_layer={self.new_layer}, exist_layer={self.exist_layer}, "
            f"kind={kind}, area={self.area:.4f})"
        )


def check_entry(bay: Bay, blocks: list[Block],
                new_block: Block,
                fast: bool = False) -> list[EntryObstruction]:
    """
    Check whether the crane can lower new_block into the bay without obstruction.

    Feasibility conditions:
      1. Bay boundary -- new_block must fit entirely within the bay.  If not,
         a sentinel EntryObstruction is returned with existing_block == new_block
         (self-reference) so callers can distinguish boundary violations from
         inter-block collisions.
      2. Final-position collision (j == k) -- layer k of new_block must not
         horizontally overlap layer k of any existing block.  Both layers would
         occupy the same physical height at the same time.
      3. Descent-path collision (j > k) -- while the crane lowers new_block by
         offset d = j - k, new-block layer k sweeps through the same absolute
         height as existing-block layer j.  Any horizontal overlap at this height
         blocks the descent.

    In summary, for each existing block, all (k, j) pairs with j >= k are checked
    for horizontal polygon overlap.  Pairs with j < k are never obstructing because
    the existing layer j would be below new-block layer k throughout descent.

    AABB pre-filter:
      For each existing block, the full-footprint bounding boxes are compared
      first.  If they do not overlap, no layer-level check is needed and the
      block is skipped entirely, avoiding all Shapely polygon construction.

    Layer count asymmetry:
      When new_block has more layers than an existing block (n_new > n_exist),
      new-block layers k >= n_exist have no existing layer at or above them
      (j >= k would require j >= n_exist, but the existing block has no such
      layers), so those new layers are never obstructed by that existing block.

    Parameters
    ----------
    bay       : Target bay for boundary checking.
    blocks    : Blocks already in the bay at the moment of insertion.
    new_block : Block being inserted.  x, y, orient_idx must be set to the
                intended placement position and orientation.
    fast      : If True, return immediately on the first obstruction found
                (useful when only presence/absence of obstruction matters,
                not the full list).

    Returns
    -------
    list[EntryObstruction]
        Empty list  -> entry is feasible.
        Non-empty   -> one record per (existing block, k, j) pair that overlaps.
        Use is_sweep to distinguish descent-path collisions (True) from
        final-position collisions (False).
    """
    results: list[EntryObstruction] = []

    # -- Condition 1: bay boundary --------------------------------------------
    if not bay.contains_block(new_block):
        # Compute the area of the block footprint that lies outside the bay.
        # The result is stored as a sentinel EntryObstruction with
        # existing_block == new_block so callers can identify boundary violations.
        bb = new_block.bounding_rect()
        bay_poly = _poly_from_verts([
            [0, 0], [bay.width, 0], [bay.width, bay.height], [0, bay.height]
        ])
        new_poly = _poly_from_verts([
            [bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]
        ])
        if bay_poly is not None and new_poly is not None:
            outside = new_poly.difference(bay_poly)
            if not outside.is_empty and outside.area > 0:
                results.append(EntryObstruction(
                    existing_block=new_block,  # self-reference sentinel for boundary violation
                    new_layer=0,
                    exist_layer=0,
                    intersection=outside,
                ))
        return results

    # -- Conditions 2 & 3: crane-path collision against each existing block ---
    new_layers = new_block.layers_at_pos()
    new_bbox   = new_block.bounding_rect()
    n_new      = len(new_layers)

    for exist in blocks:
        # AABB pre-filter: skip blocks whose footprint bounding boxes don't overlap
        if not _bb_overlap(new_bbox, exist.bounding_rect()):
            continue

        exist_layers = exist.layers_at_pos()
        n_exist      = len(exist_layers)

        # Build Shapely polygons for each new-block layer once and reuse across all j
        new_polys = [_poly_from_verts(new_layers[k]) for k in range(n_new)]

        for k in range(n_new):
            poly_new = new_polys[k]
            if poly_new is None:
                continue

            # j >= k covers: j==k (final position) and j>k (descent path sweep)
            # j < k never obstructs because existing layer j is below new layer k
            for j in range(k, n_exist):
                poly_exist = _poly_from_verts(exist_layers[j])
                if poly_exist is None:
                    continue

                try:
                    inter = poly_new.intersection(poly_exist)
                except Exception:
                    continue

                if not inter.is_empty and inter.area > 0:
                    obs = EntryObstruction(
                        existing_block=exist,
                        new_layer=k,
                        exist_layer=j,
                        intersection=inter,
                    )
                    if fast:
                        return [obs]
                    results.append(obs)

    return results


def check_exit(bay: Bay, blocks: list[Block],
               target_block: Block,
               fast: bool = False) -> list[EntryObstruction]:
    """
    Check whether the crane can lift target_block out of the bay without obstruction.

    The geometry is identical to check_entry: the j >= k rule applies in both
    directions because the crane moves purely vertically.  During ascent,
    target-block layer k sweeps through the height of surrounding-block layer j
    (j > k) in exactly the same way as during descent, just in the opposite
    direction.

    Feasibility conditions:
      1. Final-position collision (j == k) -- layer k of target_block must not
         currently overlap layer k of any surrounding block.  An existing
         overlap at the resting position means the crane cannot even start
         lifting.
      2. Ascent-path collision (j > k) -- while lifting target_block by offset
         d = j - k, target-block layer k sweeps through the height of
         surrounding-block layer j.  Any horizontal overlap at that height
         blocks the ascent.

    Self-exclusion:
      target_block itself may be included in `blocks` (e.g. when check_exit is
      called with the full list of blocks present in the bay).  It is
      automatically skipped by comparing block_id.  Callers do not need to
      remove target_block from the list before calling.

    Parameters
    ----------
    bay          : Target bay (used for interface consistency; bay boundary is
                   not re-checked here since the block is already placed).
    blocks       : All blocks currently present in the bay, including
                   target_block itself (automatically excluded internally).
    target_block : The block to be removed by the crane.
    fast         : If True, return on the first obstruction found.

    Returns
    -------
    list[EntryObstruction]
        Empty list  -> exit is feasible.
        Non-empty   -> one record per (surrounding block, k, j) pair that overlaps.
        Use is_sweep to distinguish ascent-path collisions (True) from
        final-position collisions (False).
    """
    results: list[EntryObstruction] = []

    target_layers = target_block.layers_at_pos()
    target_bbox   = target_block.bounding_rect()
    n_target      = len(target_layers)

    # Build Shapely polygons for each target layer once and reuse across all surrounding blocks
    target_polys = [_poly_from_verts(target_layers[k]) for k in range(n_target)]

    for exist in blocks:
        # Skip target block itself (callers may include it in the blocks list)
        if exist.block_id == target_block.block_id:
            continue

        # AABB pre-filter: skip blocks whose footprint bounding boxes don't overlap
        if not _bb_overlap(target_bbox, exist.bounding_rect()):
            continue

        exist_layers = exist.layers_at_pos()
        n_exist      = len(exist_layers)

        for k in range(n_target):
            poly_target = target_polys[k]
            if poly_target is None:
                continue

            # j >= k covers: j==k (final position) and j>k (ascent path sweep)
            for j in range(k, n_exist):
                poly_exist = _poly_from_verts(exist_layers[j])
                if poly_exist is None:
                    continue

                try:
                    inter = poly_target.intersection(poly_exist)
                except Exception:
                    continue

                if not inter.is_empty and inter.area > 0:
                    obs = EntryObstruction(
                        existing_block=exist,
                        new_layer=k,
                        exist_layer=j,
                        intersection=inter,
                    )
                    if fast:
                        return [obs]
                    results.append(obs)

    return results


# -----------------------------------------------------------------------------
# Convenience function: batch creation of Bay + Block list from instance JSON
# -----------------------------------------------------------------------------

def blocks_from_instance(instance: dict, bay_idx: int,
                         positions: Optional[list[tuple[int, int]]] = None,
                         orient_indices: Optional[list[int]] = None) -> list[Block]:
    """
    Construct a list of Block objects for blocks whose highest bay preference is bay_idx.

    This is a convenience helper for quick visual tests and unit tests.  It selects
    blocks by bay preference (mimicking the greedy algorithm's bay assignment),
    then places each block at a caller-supplied position or, if none is given, at an
    automatically computed grid position that avoids obvious overlaps.

    Block selection:
      A block is included iff  argmax(bay_preferences) == bay_idx.
      Blocks are returned in their original index order from instance["blocks"].

    Automatic grid placement (positions=None):
      The bounding box of orientation 0's first layer is used to estimate the block
      footprint.  Blocks are arranged left-to-right, top-to-bottom in columns of
      width (bw + 2), ensuring a 2-unit gap between adjacent blocks.  Positions are
      clamped to [0, bay_width - bw] * [0, bay_height - bh] to keep blocks inside
      the bay.  The grid layout does not guarantee collision-free placement; it is
      only intended to produce reasonable starting positions for visualisation.

    Parameters
    ----------
    instance       : Instance dict loaded from a problem instance JSON file.
    bay_idx        : Index of the target bay.  Only blocks preferring this bay
                     are included in the result.
    positions      : Optional list of (x, y) integer positions, one per selected
                     block (in selection order).  If shorter than the block list,
                     auto-placement is used for the remaining blocks.
    orient_indices : Optional list of orient_idx values, one per selected block.
                     Missing entries default to 0.

    Returns
    -------
    list[Block]
        Block objects in selection order, each with x/y/orient_idx set.
        block_id matches the original index in instance["blocks"].
    """
    bays = instance["bays"]
    bay_w = bays[bay_idx]["width"]
    bay_h = bays[bay_idx]["height"]

    # Select blocks whose argmax bay_preference equals bay_idx.
    # prefs.index(max(prefs)) returns the *first* index that achieves the maximum,
    # so ties are broken in favour of the lower bay index -- consistent with the
    # greedy algorithm's bay assignment logic.
    selected: list[tuple[int, dict]] = []
    for bi, blk_data in enumerate(instance["blocks"]):
        prefs = blk_data["bay_preferences"]
        if prefs.index(max(prefs)) == bay_idx:
            selected.append((bi, blk_data))

    result: list[Block] = []
    for seq, (bi, blk_data) in enumerate(selected):
        oi = 0 if orient_indices is None else orient_indices[seq] if seq < len(orient_indices) else 0

        if positions is not None and seq < len(positions):
            px, py = positions[seq]
        else:
            # Automatic grid placement: estimate footprint from the first layer's
            # bounding box and pack blocks left-to-right with a 2-unit gap.
            layers = _resolve_layers(blk_data["shape"][oi]["layers"])
            if layers:
                bb = _bounding_box(_anchor_verts(layers[0]))
                bw = max(1, int(math.ceil(bb[2])))
                bh = max(1, int(math.ceil(bb[3])))
            else:
                bw, bh = 5, 5
            cols = max(1, bay_w // max(bw + 2, 1))
            col = seq % cols
            row = seq // cols
            px = min(col * (bw + 2), bay_w - bw)
            py = min(row * (bh + 2), bay_h - bh)

        result.append(Block(
            block_id=bi,
            block_data=blk_data,
            x=int(px), y=int(py),
            orient_idx=oi,
        ))

    return result


# -----------------------------------------------------------------------------
# check_feasibility -- solution validity check + objective computation
# -----------------------------------------------------------------------------

def check_feasibility(prob_info: dict, solution: dict) -> dict:
    """
    Validate a solution through five ordered stages and compute the objective.

    Stages are checked in sequence.  The first stage with any violation causes
    an immediate return; later stages are not evaluated.  This means the
    returned stage number is the *earliest* stage that fails, not necessarily
    the only one.

    -- Stage 1: assignment validity --------------------------------------------
    Every block in prob_info["blocks"] must appear in exactly one ENTRY
    operation and one EXIT operation.  Additionally:
      * bay_id must be in [0, n_bays).
      * orient_idx must be in [0, len(shape)).
      * entry_time >= release_time  (with 1e-6 tolerance for floating-point).
      * exit_time - entry_time >= processing_time  (same tolerance).
    A block_id appearing in more than one ENTRY or more than one EXIT operation
    is an immediate Stage-1 violation; no silent overwrite occurs.

    -- Stage 2: crane entry feasibility ----------------------------------------
    For each block's ENTRY event at time entry_time, check_entry is called with
    the set of blocks that were already present in the bay before this block
    arrives (strictly earlier entry, not yet departed):
      present_at_entry = { b : b.entry_time < entry_time < b.exit_time }
    Blocks entering at the same time are excluded -- their relative ordering
    within the same time point is validated by Stage 5.

    -- Stage 3: crane exit feasibility -----------------------------------------
    For each block's EXIT event at time exit_time, check_exit is called with
    the set of blocks that were already present in the bay before this block
    begins its ascent (strictly earlier entry, not yet departed):
      present_at_exit = { b : b.entry_time < exit_time < b.exit_time }
      | { target_block itself }
    Blocks entering or exiting at the same time are excluded -- their relative
    ordering within the same time point is validated by Stage 5.
    (The target block is always included; check_exit internally skips it.)

    -- Stage 4: spatial collision and boundary ----------------------------------
    For every pair of blocks (in the same bay) whose time intervals overlap,
    check_collisions is called to detect same-height polygon overlaps.  Also
    checks that every block lies within its bay boundary via contains_block.
    Time overlap uses the half-open interval model: [a, e) & [a', e') != {}
    iff a < e' and a' < e.

    -- Stage 5: sequential operation feasibility --------------------------------
    Operations are replayed in time order.  A bay_present set per bay tracks
    which blocks are currently in the bay.  At each time point:
      * All EXIT ops must precede all ENTRY ops (violation if an EXIT appears
        after any ENTRY at the same time point).
      * EXIT: the block must be in bay_present; check_exit is called against
        the current bay_present set.  On success the block is removed from
        bay_present.
      * ENTRY: check_entry is called against the current bay_present set.
        On success the block is added to bay_present.
    This stage is stricter than Stage 2/3 in detecting ordering violations
    within a time point, but more lenient for concurrent ENTRY/EXIT events
    because EXIT operations are processed first (removing blocks before new
    ones arrive).

    -- Objective computation (only when all stages pass) -----------------------
    obj1 = Sigma max(0, exit_time_i - due_date_i)          (total tardiness)
    obj2 = max_{j1!=j2} |u_j1 * load_j1 - u_j2 * load_j2|  (normalized imbalance)
             where u_j = avg_bay_area / (W_j * H_j)
    obj3 = Sigma_i (S_i_max - S_i_bay_i)                    (preference penalty)
             where S_i_max = max_j bay_preferences_i[j]
    objective = w1*obj1 + w2*obj2 + w3*obj3

    Parameters
    ----------
    prob_info : Instance JSON dict with keys "bays", "blocks", "weights".
    solution  : Solver output dict:
                {"operations": {str(time_int): [op_dict, ...]}}
                ENTRY op_dict keys: "type", "block_id", "bay_id", "x", "y", "orient_idx"
                EXIT  op_dict keys: "type", "block_id", "bay_id"
                entry_time = int(t_str) for ENTRY; exit_time = int(t_str) for EXIT.

    Returns
    -------
    dict with keys:
        feasible   : bool         -- True if all five stages pass.
        stage      : int          -- Failing stage index (1-4), or 5 if all pass.
        violations : list[str]    -- Human-readable violation descriptions.
        objective  : float|None   -- Total objective value; None if infeasible.
        obj1       : float|None   -- Tardiness component; None if infeasible.
        obj2       : float|None   -- Load-balance component; None if infeasible.
        obj3       : float|None   -- Bay-preference component; None if infeasible.
    """
    _INFEASIBLE = {"feasible": False, "stage": 0, "violations": [],
                   "objective": None, "obj1": None, "obj2": None, "obj3": None}

    # -- Argument safeguard ---------------------------------------------------
    if not isinstance(prob_info, dict):
        return {**_INFEASIBLE, "violations": [
            f"prob_info must be a dict, got {type(prob_info).__name__}"
        ]}
    if not isinstance(solution, dict):
        return {**_INFEASIBLE, "violations": [
            f"solution must be a dict, got {type(solution).__name__}"
        ]}
    for key in ("blocks", "bays"):
        if key not in prob_info:
            return {**_INFEASIBLE, "violations": [f"prob_info missing required key '{key}'"]}
        if not isinstance(prob_info[key], list):
            return {**_INFEASIBLE, "violations": [
                f"prob_info['{key}'] must be a list, got {type(prob_info[key]).__name__}"
            ]}
    if len(prob_info["bays"]) == 0:
        return {**_INFEASIBLE, "violations": ["prob_info['bays'] is empty"]}

    blocks_data = prob_info["blocks"]
    bays_data   = prob_info["bays"]
    n_blocks    = len(blocks_data)
    n_bays      = len(bays_data)

    # Reconstruct per-block assignment records from the flat operations dict.
    # ENTRY ops carry the placement data (bay_id, x, y, orient_idx, entry_time);
    # EXIT ops supply the matching exit_time.  Each block_id must appear in
    # exactly one ENTRY and one EXIT operation; duplicates are Stage-1 violations.
    raw_operations = solution.get("operations", {})
    if not isinstance(raw_operations, dict):
        return {**_INFEASIBLE, "violations": [
            f"solution['operations'] must be a dict, got {type(raw_operations).__name__}"
        ]}

    operations: dict = {}
    _asgn_tmp: dict[int, dict] = {}
    _entry_count: dict[int, int] = {}
    _exit_count: dict[int, int] = {}
    for _t_str, _ops_at_t in raw_operations.items():
        try:
            _t = float(int(_t_str))
        except (ValueError, TypeError):
            return {**_INFEASIBLE, "violations": [
                f"operations key '{_t_str}' cannot be converted to an integer time"
            ]}
        if not isinstance(_ops_at_t, list):
            return {**_INFEASIBLE, "violations": [
                f"operations['{_t_str}'] must be a list, got {type(_ops_at_t).__name__}"
            ]}
        operations[_t_str] = _ops_at_t
        for _op in _ops_at_t:
            if not isinstance(_op, dict):
                return {**_INFEASIBLE, "violations": [
                    f"operations['{_t_str}'] contains a non-dict entry: {_op!r}"
                ]}
            for _req in ("type", "block_id", "bay_id"):
                if _req not in _op:
                    return {**_INFEASIBLE, "violations": [
                        f"operations['{_t_str}']: op missing required key '{_req}': {_op!r}"
                    ]}
            _bid = _op["block_id"]
            if _op["type"] == "ENTRY":
                _entry_count[_bid] = _entry_count.get(_bid, 0) + 1
                _asgn_tmp[_bid] = {
                    "block_id":   _bid,
                    "bay_id":     _op["bay_id"],
                    "x":          _op.get("x", 0.0),
                    "y":          _op.get("y", 0.0),
                    "orient_idx": _op.get("orient_idx", 0),
                    "entry_time": _t,
                    "exit_time":  None,  # filled in when the EXIT op is encountered
                }
            elif _op["type"] == "EXIT":
                _exit_count[_bid] = _exit_count.get(_bid, 0) + 1
                if _bid in _asgn_tmp:
                    _asgn_tmp[_bid]["exit_time"] = _t
    assignments = list(_asgn_tmp.values())

    violations: list[str] = []

    # -- Stage 1: every block assigned to exactly one bay --------------------
    # Check for duplicate ENTRY or EXIT operations before anything else.
    for _bid, _cnt in sorted(_entry_count.items()):
        if _cnt > 1:
            violations.append(
                f"Stage1: block {_bid} has {_cnt} ENTRY operations (expected exactly 1)"
            )
    for _bid, _cnt in sorted(_exit_count.items()):
        if _cnt > 1:
            violations.append(
                f"Stage1: block {_bid} has {_cnt} EXIT operations (expected exactly 1)"
            )
    if violations:
        return {"feasible": False, "stage": 1, "violations": violations,
                "objective": None, "obj1": None, "obj2": None, "obj3": None}

    # Build the set of block_ids that appear in at least one ENTRY op.
    # Any block_id missing from this set was never placed, which is a violation.
    assigned_ids = [a["block_id"] for a in assignments]
    seen: set[int] = set(assigned_ids)
    for bi in range(n_blocks):
        if bi not in seen:
            violations.append(f"Stage1: block {bi} is not assigned")
    # Validate index ranges and timing constraints for each assigned block.
    # exit_time may still be None here if no EXIT op was found for the block.
    for a in assignments:
        if not (0 <= a["bay_id"] < n_bays):
            violations.append(
                f"Stage1: block {a['block_id']} assigned to invalid bay {a['bay_id']}"
            )
        if not (0 <= a["orient_idx"] < len(blocks_data[a["block_id"]]["shape"])):
            violations.append(
                f"Stage1: block {a['block_id']} has invalid orient_idx {a['orient_idx']}"
            )
        if a["exit_time"] is None:
            violations.append(
                f"Stage1: block {a['block_id']} has no EXIT operation"
            )
            continue
        ei, ai, pi = a["exit_time"], a["entry_time"], blocks_data[a["block_id"]]["processing_time"]
        if ei - ai < pi - 1e-6:
            violations.append(
                f"Stage1: block {a['block_id']} exit-entry={ei-ai:.2f} < processing_time={pi}"
            )
        if ai < blocks_data[a["block_id"]]["release_time"] - 1e-6:
            violations.append(
                f"Stage1: block {a['block_id']} entry_time={ai} < release_time="
                f"{blocks_data[a['block_id']]['release_time']}"
            )

    if violations:
        return {"feasible": False, "stage": 1, "violations": violations,
                "objective": None, "obj1": None, "obj2": None, "obj3": None}

    # Group assignments and their Block objects by bay index.
    # bay_asgns[j] / bay_blocks[j] are parallel lists: index k in both refers
    # to the same block.  This pairing is used heavily in Stages 2, 3, and 4
    # to look up the Block object from the assignment dict and vice-versa.
    bay_asgns: list[list[dict]] = [[] for _ in range(n_bays)]
    bay_blocks: list[list[Block]] = [[] for _ in range(n_bays)]
    for a in assignments:
        j = a["bay_id"]
        bi = a["block_id"]
        bay_asgns[j].append(a)
        bay_blocks[j].append(Block(
            block_id=bi,
            block_data=blocks_data[bi],
            x=int(round(a["x"])),
            y=int(round(a["y"])),
            orient_idx=a["orient_idx"],
        ))

    def _time_overlaps(a1: float, e1: float, a2: float, e2: float) -> bool:
        """True if intervals [a1, e1) and [a2, e2) overlap."""
        return a1 < e2 and a2 < e1

    # -- Stage 2: crane entry feasibility at entry_time ----------------------
    # For each block, collect the blocks that were already present in the same
    # bay before the new block arrives.  The condition is:
    #   a_k < ai < e_k
    # where ai is entry_time of the new block, a_k and e_k are entry/exit of
    # an existing block k.  Both bounds are strict:
    #   * a_k < ai  : excludes blocks entering at the same time as the new
    #                 block -- their relative ordering is handled by Stage 5.
    #   * ai < e_k  : excludes blocks departing at exactly ai -- those have
    #                 already been removed by the crane before this insertion.
    for j in range(n_bays):
        bay = Bay.from_dict(bays_data[j], j)
        for idx, a in enumerate(bay_asgns[j]):
            new_blk = bay_blocks[j][idx]
            ai, ei = a["entry_time"], a["exit_time"]
            present = [
                bay_blocks[j][k]
                for k, other in enumerate(bay_asgns[j])
                if k != idx
                and other["entry_time"] < ai < other["exit_time"]
            ]
            obs = check_entry(bay, present, new_blk)
            if obs:
                for o in obs:
                    if o.existing_block.block_id == new_blk.block_id:
                        violations.append(
                            f"Stage2: t={int(ai)}: block {a['block_id']} exceeds bay boundary "
                            f"(area={o.area:.3f})"
                        )
                    else:
                        kind = "sweep" if o.is_sweep else "collision"
                        violations.append(
                            f"Stage2: t={int(ai)}: block {a['block_id']} entry obstructed by "
                            f"block {o.existing_block.block_id} "
                            f"(new_layer={o.new_layer}, exist_layer={o.exist_layer}, "
                            f"kind={kind}, area={o.area:.3f})"
                        )

    if violations:
        return {"feasible": False, "stage": 2, "violations": violations,
                "objective": None, "obj1": None, "obj2": None, "obj3": None}

    # -- Stage 3: crane exit feasibility at exit_time ------------------------
    # For each block, collect the blocks that were already present in the same
    # bay before the target block begins its ascent.  The condition is:
    #   a_k < ei < e_k   (strictly between, both ends excluded)
    # Both bounds are strict:
    #   * a_k < ei  : excludes blocks entering at exactly ei -- those have not
    #                 yet arrived, and their ordering relative to this EXIT is
    #                 handled by Stage 5.
    #   * ei < e_k  : excludes blocks departing at exactly ei -- their relative
    #                 ordering with this EXIT is handled by Stage 5.
    # The target block itself is always included in present_at_exit so that
    # check_exit can skip it by block_id; callers do not pre-filter it out.
    for j in range(n_bays):
        bay = Bay.from_dict(bays_data[j], j)
        for idx, a in enumerate(bay_asgns[j]):
            target_blk = bay_blocks[j][idx]
            ai, ei = a["entry_time"], a["exit_time"]
            present_at_exit = [
                bay_blocks[j][k]
                for k, other in enumerate(bay_asgns[j])
                if bay_blocks[j][k].block_id == target_blk.block_id
                or (other["entry_time"] < ei < other["exit_time"])
            ]
            obs = check_exit(bay, present_at_exit, target_blk)
            if obs:
                for o in obs:
                    kind = "sweep" if o.is_sweep else "collision"
                    violations.append(
                        f"Stage3: t={int(ei)}: block {a['block_id']} exit obstructed by "
                        f"block {o.existing_block.block_id} "
                        f"(target_layer={o.new_layer}, exist_layer={o.exist_layer}, "
                        f"kind={kind}, area={o.area:.3f})"
                    )

    if violations:
        return {"feasible": False, "stage": 3, "violations": violations,
                "objective": None, "obj1": None, "obj2": None, "obj3": None}

    # -- Stage 4: no spatial collisions + all blocks within bay boundary -----
    for j in range(n_bays):
        bay = Bay.from_dict(bays_data[j], j)

        # 4-a: Bay boundary check via bounding box.  Any block whose axis-aligned
        # bounding rectangle extends outside the bay's [0,width]*[0,height] region
        # fails this check.  This is a necessary but not sufficient condition: it
        # does not catch non-convex shapes that protrude at the polygon level while
        # the bounding box still fits -- but such shapes cannot arise from the
        # placement model used in this solver.
        for idx, a in enumerate(bay_asgns[j]):
            blk = bay_blocks[j][idx]
            if not bay.contains_block(blk):
                bb = blk.bounding_rect()
                violations.append(
                    f"Stage4: block {a['block_id']} bounding box {bb} "
                    f"exceeds bay {j} ({bay.width}*{bay.height})"
                )

        # 4-b: Spatial collision check between every pair of blocks in this bay
        # that co-exist in time.  Two blocks with non-overlapping time intervals
        # can never be simultaneously present, so only pairs satisfying
        #   [a_p, e_p) & [a_q, e_q) != {}   (i.e., a_p < e_q and a_q < e_p)
        # are passed to check_collisions.  This avoids O(n^2) polygon checks for
        # blocks that are never in the bay at the same time.
        n = len(bay_asgns[j])
        for p in range(n):
            for q in range(p + 1, n):
                ap, aq = bay_asgns[j][p], bay_asgns[j][q]
                if not _time_overlaps(ap["entry_time"], ap["exit_time"],
                                      aq["entry_time"], aq["exit_time"]):
                    continue  # disjoint time intervals -> spatial collision impossible
                results = check_collisions(bay,
                                           [bay_blocks[j][p], bay_blocks[j][q]])
                for r in results:
                    violations.append(
                        f"Stage4: block {ap['block_id']} and block {aq['block_id']} "
                        f"collide in bay {j} at layer {r.layer_index} "
                        f"(area={r.area:.3f})"
                    )

    if violations:
        return {"feasible": False, "stage": 4, "violations": violations,
                "objective": None, "obj1": None, "obj2": None, "obj3": None}

    # -- Stage 5: operations sequential feasibility ---------------------------
    # Replay all operations in chronological order, maintaining a bay_present
    # set per bay that tracks which block_ids are physically in the bay right now.
    # At each time point, all EXIT ops must appear before any ENTRY op -- if an
    # EXIT is listed after an ENTRY at the same time step, that is a violation
    # because the departing block would still be present when the new one arrives.
    # For each EXIT: the block must be in bay_present, and check_exit must pass
    # against the current bay_present set.  For each ENTRY: check_entry must pass
    # against the current bay_present set.  Successful ops update bay_present.
    asgn_by_id: dict[int, dict] = {a["block_id"]: a for a in assignments}
    bay_present: list[set[int]] = [set() for _ in range(n_bays)]

    for t_str in sorted(operations, key=lambda s: int(s)):
        ops = operations[t_str]
        # First pass: scan for EXIT-after-ENTRY ordering violations at this time point.
        seen_entry = False
        for op in ops:
            if op["type"] == "EXIT" and seen_entry:
                violations.append(
                    f"Stage5: t={t_str}: EXIT block {op['block_id']} listed "
                    f"after an ENTRY operation (EXIT must precede ENTRY)"
                )
            if op["type"] == "ENTRY":
                seen_entry = True

        for op in ops:
            kind = op["type"]
            bid  = op["block_id"]
            jay  = op["bay_id"]
            if bid not in asgn_by_id:
                violations.append(
                    f"Stage5: t={t_str}: operation references unassigned block {bid}"
                )
                continue
            a = asgn_by_id[bid]
            if a["bay_id"] != jay:
                violations.append(
                    f"Stage5: t={t_str}: operation bay_id={jay} does not match "
                    f"assignment bay_id={a['bay_id']} for block {bid}"
                )
                continue

            bay = Bay.from_dict(bays_data[jay], jay)
            target_blk = Block(
                block_id=bid,
                block_data=blocks_data[bid],
                x=int(round(a["x"])),
                y=int(round(a["y"])),
                orient_idx=a["orient_idx"],
            )

            if kind == "ENTRY":
                present_blks = [
                    Block(block_id=k,
                          block_data=blocks_data[k],
                          x=int(round(asgn_by_id[k]["x"])),
                          y=int(round(asgn_by_id[k]["y"])),
                          orient_idx=asgn_by_id[k]["orient_idx"])
                    for k in bay_present[jay]
                ]
                obs = check_entry(bay, present_blks, target_blk)
                if obs:
                    for o in obs:
                        if o.existing_block.block_id == bid:
                            violations.append(
                                f"Stage5: t={t_str}: ENTRY block {bid} "
                                f"exceeds bay boundary"
                            )
                        else:
                            kind_str = "sweep" if o.is_sweep else "collision"
                            violations.append(
                                f"Stage5: t={t_str}: ENTRY block {bid} "
                                f"obstructed by block {o.existing_block.block_id} "
                                f"({kind_str})"
                            )
                else:
                    bay_present[jay].add(bid)

            else:  # EXIT
                if bid not in bay_present[jay]:
                    violations.append(
                        f"Stage5: t={t_str}: EXIT block {bid} is not present "
                        f"in bay {jay} at this point"
                    )
                    continue
                present_blks = [
                    Block(block_id=k,
                          block_data=blocks_data[k],
                          x=int(round(asgn_by_id[k]["x"])),
                          y=int(round(asgn_by_id[k]["y"])),
                          orient_idx=asgn_by_id[k]["orient_idx"])
                    for k in bay_present[jay]
                ]
                obs = check_exit(bay, present_blks, target_blk)
                if obs:
                    for o in obs:
                        kind_str = "sweep" if o.is_sweep else "collision"
                        violations.append(
                            f"Stage5: t={t_str}: EXIT block {bid} "
                            f"obstructed by block {o.existing_block.block_id} "
                            f"({kind_str})"
                        )
                else:
                    bay_present[jay].discard(bid)

    if violations:
        return {"feasible": False, "stage": 5, "violations": violations,
                "objective": None, "obj1": None, "obj2": None, "obj3": None}

    # -- Objective function computation ---------------------------------------
    # Weights default to 1.0 if not present in prob_info["weights"].
    w1 = prob_info.get("weights", {}).get("w1", 1.0)
    w2 = prob_info.get("weights", {}).get("w2", 1.0)
    w3 = prob_info.get("weights", {}).get("w3", 1.0)

    obj1 = 0.0           # total tardiness: Sigma max(0, exit_time - due_date)
    bay_loads = [0.0] * n_bays  # accumulated workload per bay for obj2
    obj3 = 0.0           # preference penalty: Sigma (S_i_max - S_i_bay_i)

    for a in assignments:
        bi = a["block_id"]
        bj = a["bay_id"]
        blk = blocks_data[bi]
        obj1 += max(0.0, a["exit_time"] - blk["due_date"])
        bay_loads[bj] += blk["workload"]
        s_max = max(blk["bay_preferences"])
        obj3 += s_max - blk["bay_preferences"][bj]

    # obj2: maximum normalized workload imbalance across all bay pairs.
    # u_j = avg_bay_area / (W_j * H_j) -- smaller u for larger bays (less congested).
    bay_areas = [bays_data[j]["width"] * bays_data[j]["height"] for j in range(n_bays)]
    avg_area  = sum(bay_areas) / n_bays
    u = [avg_area / a for a in bay_areas]
    if n_bays >= 2:
        obj2 = math.floor(max(
            abs(u[j1] * bay_loads[j1] - u[j2] * bay_loads[j2])
            for j1 in range(n_bays) for j2 in range(n_bays)
            if j1 != j2
        ))
    else:
        obj2 = 0.0

    objective = w1 * obj1 + w2 * obj2 + w3 * obj3

    return {
        "feasible":  True,
        "stage":     5,
        "violations": [],
        "objective": objective,
        "obj1":      obj1,
        "obj2":      obj2,
        "obj3":      obj3,
    }

