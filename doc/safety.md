# src 以下の panic / 実行時エラー候補

## 対象

提出物の実行時に通る `src` 以下を対象に、問題入力に依存して panic・実行時エラーになりうる箇所を整理する。

提出 wrapper や `tools` はここでは対象外。

## 優先度高

### 1. `collision.rs`: layer の凸分解失敗

該当箇所：

- `build_shape_geom()`
- `build_convex_parts()`
- `make_convex_part()`
- `rasterize_convex_pair()`

主な panic 候補：

```rust
assert!(!parts.is_empty(), ...)
assert!(points.len() >= 3 && points.len() <= MAX_CONVEX_VERTS)
assert!(hull.len >= 3, ...)
```

原因：

- layer が退化している。
- 頂点数が少ない。
- ear clipping が数値誤差や複雑形状で失敗する。
- 凸部品の頂点数が想定を超える。
- Minkowski difference hull が退化する。

問題文上は実インスタンスの shape は通常有効と考えられるが、hidden を含むあらゆる入力を想定すると、ここは最も入力依存で panic しやすい。

修正方針：

- 退化 layer は panic せず skip する。
- `build_convex_parts()` が空になった場合は、bbox 矩形を保守的な convex part として使う。
- bbox も退化している場合は、その layer を無視する。
- `rasterize_convex_pair()` は `hull.len < 3` なら何も追加せず return するか、bbox ベースの保守的 interval に fallback する。

保守的に collision を多めに判定する方向なら、解の実行可能性を壊しにくい。

### 2. `precompute.rs`: 空 orientation の bbox

該当箇所：

- `orientation_bbox_bounds()`
- `orientation_bbox()`
- `build_orientation_neighbors()`
- `build_other_block_neighbors()`

現在の挙動：

- 全 layer が空、または全 layer が skip されるような orientation だと、bbox が `INFINITY` / `NEG_INFINITY` のまま返る可能性がある。
- その後の bbox 面積、IoU、offset 計算で NaN や極端な値が混入しうる。

修正方針：

- 点が 1 つもない orientation は `Boundsf { min_x: 0.0, min_y: 0.0, max_x: 0.0, max_y: 0.0 }` を返す。
- `collision.rs` の empty orientation bbox と挙動を揃える。
- 可能なら `Precompute::build()` の最初で「各 block に少なくとも 1 orientation、各 orientation に有効 layer がある」ことを検証し、無効なら solver error にする。

## 優先度中

### 3. `solver.rs`: 時刻計算の overflow

該当箇所：

- `try_place_block()`
- `insert_greedy()`
- `old_time_info()`
- `add_forbidden_from_hit_state()`
- `schedule_to_solution()`

主な危険な計算：

```rust
entry_time + block.processing_time
original.exit_time - block.due_date
old.entry_time - process_t + 1
old.exit_time - 1
info.old_exit_time - info.new_process_time - 1
info.old_exit_time - info.new_process_time + 1
allow_l - 1
allow_r + 1
```

通常の問題スケールでは問題になりにくいが、極端に大きい `release_time`, `due_date`, `processing_time` が来ると debug/release に関わらず不正挙動や panic の原因になる。

修正方針：

- `max_t` を `i64::MAX` ではなく `i64::MAX - processing_time` に制限する。
- `entry_time + processing_time` は `checked_add()` または `saturating_add()` を使う。
- forbidden interval の境界計算は `saturating_add()` / `saturating_sub()` を使う。
- `processing_time <= 0` は問題として不正なので、事前 validation で弾く。

### 4. `solver.rs`: y-event 座標計算の overflow

該当箇所：

- `insert_greedy()`
- `push_y_event()`

危険な計算：

```rust
old.y - hi
old.y - lo
old.y + lo
old.y + hi
base_x + ddx
base_y + ddy
old.x + dx + ddx
old.y + dy + ddy
```

通常の bay 座標は小さい想定だが、極端な bbox / 座標が来ると overflow しうる。

修正方針：

- y-event の区間変換は `i128` で計算してから `fit_range` に clamp する。
- clamp 後に `i64` 範囲外なら捨てる。
- shift/rotate/swap の候補座標も `checked_add()` か `saturating_add()` にする。

### 5. `solver.rs`: `apply_y_event()` の内部不変条件依存 unwrap

該当箇所：

```rust
let pos = active_pos[old_idx].take().unwrap();
let last = active_old_ids.pop().unwrap();
```

原因：

- y-event の開始・終了イベントの対応が壊れた場合に panic する。
- 現在の `push_y_event()` と sort/sweep が正しければ基本的には起きない。

修正方針：

- ここは入力直接依存というより実装不変条件。
- panic 回避を優先するなら、`debug_assert!` にして、release では不整合イベントを無視する。
- ただし不整合を無視すると探索結果が壊れうるため、まずは event 生成側の overflow 対策を優先する。

## 優先度低

### 6. `collision.rs`: block/orientation index の `expect()` / 添字アクセス

該当箇所：

```rust
self.block_pair_index[moving.block_id * self.n + fixed.block_id]
    .expect("collision block pair should be precomputed")

let moving_geom = &self.geoms[moving.block_id][moving.orient_idx];
let fixed_geom = &self.geoms[fixed.block_id][fixed.orient_idx];
```

原因：

- `ScheduledBlock` の `block_id` / `orient_idx` が壊れている場合に panic する。

現在の評価：

- schedule は solver 内部で `problem.blocks` と `pre` から生成しているため、通常は安全。
- 入力 JSON の solution を読む処理はないので、外部から壊れた index は入らない。

修正方針：

- 必須ではない。
- 気にするなら `debug_assert!` を追加し、public API を `Option`/`Clear` 返却にする。ただし呼び出し側が増えるので YAGNI 寄り。

### 7. `solver.rs`: `pref_penalty[s.block_id][s.bay_id]`

該当箇所：

- `score_schedule()`
- `insert_greedy()`
- `try_move_neighbor()`
- `choose_removed_blocks()`

原因：

- `bay_preferences.len() < bays.len()` の入力だと、`pref_penalty[block_id][bay_id]` で panic しうる。

問題文上は各 block に全 bay 分の preference がある想定。

修正方針：

- `solve()` の冒頭で `validate_problem()` を追加し、`bay_preferences.len() == bays.len()` を確認する。
- 不正なら panic ではなく `Err(String)` を返す。

### 8. `sample_neighbor()` の `unwrap()`

該当箇所：

```rust
NEIGHBOR_PROBS.last().unwrap().0
```

原因：

- `NEIGHBOR_PROBS` が空なら panic。

現在の評価：

- const で非空なので実質安全。

修正方針：

- 修正不要。
- どうしても消すなら `NEIGHBOR_PROBS[0].0` も同じなので意味は薄い。

### 9. `util.rs` の未使用 sampler 系 assert

該当箇所：

- `XorShift32::new()`
- `BufferedRandom::new()`
- `DiscreteSampler::new()`
- `ContinousSampler::new()`

現在の評価：

- 現 solver の主要経路ではほぼ未使用。
- 入力依存の提出時 panic 候補としては優先度低。

修正方針：

- 放置でよい。
- 使用するようになった時点で、引数 validation または `Result` 化を検討する。

## 追加したい validation

`solve()` 冒頭、または `Precompute::build()` 前に最低限以下を確認すると、panic ではなく明示的な error にできる。

```text
- bays が空でない
- blocks が空でない、または空でも許容するなら空 solution を返す
- bay.width > 0, bay.height > 0
- 各 block の processing_time > 0
- 各 block の shape が空でない
- 各 block の bay_preferences.len() == bays.len()
- 各 orientation に有効な layer が少なくとも 1 つある
- 各有効 layer の頂点数が 3 以上
- 座標が finite である
```

ただし、提出用 solver としては「不正入力を弾く」より「有効入力で panic しない」ことが重要なので、まずは幾何処理と overflow 対策を優先する。

## 推奨修正順

1. `precompute.rs` の empty bbox 対応。
2. `collision.rs` の凸分解失敗 panic を保守的 fallback に変更。
3. `solver.rs` の時刻計算を overflow 安全化。
4. `solver.rs` の y-event / 近傍候補座標計算を overflow 安全化。
5. 必要なら `validate_problem()` を追加して、入力形式不整合を `Err(String)` にする。
