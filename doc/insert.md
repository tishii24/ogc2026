# insert-greedy 高速化方針

## 目的

現状の `insert_greedy` は、各 `(bay, orient, x, y)` ごとに `get_insert_t()` を呼び、同じ bay の全ブロックに対して衝突判定と forbidden interval の sort/merge を行っている。

これを、固定した `(bay, orient, x)` ごとに `y` 方向の衝突イベントを作ってスイープし、衝突状態が変わる点だけ評価する形に変更する。

## 前提

- `x_step == 1`, `y_step == 1` 前提。
- `new_old_clear` と `old_new_clear` は方向付きで区別する。
- forbidden intervals は同時に少数、おおむね 1〜5 個程度と想定する。
- segment tree は使わない。
- sort/merge も基本的には避け、少数 interval 向けの線形探索を使う。

## 全体像

```text
for bay_id
  bay にある old blocks を集める
  for orient_idx
    fit_range を取得
    for x in fit_range.min_x..=fit_range.max_x
      old blocks から y イベントを作る
      y を小さい順にイベントスイープ
        hit state を更新
        active old blocks から forbidden intervals を full-build
        first feasible time を sort なしで取得
        候補を評価
```

現在の `anchor_x` による早期打ち切りは維持する。

## collision.rs に追加する API

既存の `CollisionGrid` は、`dx` 列ごとに merged 済みの `dy` intervals を持っている。
これを外から取得できる API を `CollisionPrecompute` に追加する。

```rust
pub struct BlockOrient {
    pub block_id: usize,
    pub orient_idx: usize,
}

pub fn crane_dy_intervals(
    &self,
    moving: BlockOrient,
    fixed: BlockOrient,
    dx: i64,
) -> &[(i64, i64)]
```

`crane_dy_intervals()` は座標を持たず、ordered pair / orientation key と `dx = fixed.x - moving.x` だけで該当 `dx` 列の interval slice を返す。

実装イメージ：

```rust
impl CollisionPrecompute {
    pub fn crane_dy_intervals(
        &self,
        moving: BlockOrient,
        fixed: BlockOrient,
        dx: i64,
    ) -> &[(i64, i64)] {
        if moving.block_id == fixed.block_id {
            return &[];
        }

        let pair_idx = self.block_pair_index[moving.block_id * self.n + fixed.block_id]
            .expect("collision block pair should be precomputed");
        let pair = &self.block_pairs[pair_idx];
        let key = moving.orient_idx * pair.fixed_orients + fixed.orient_idx;
        let orient_pair = self.get_or_build_orient_pair(moving, fixed, pair_idx, key);

        orient_pair.crane.dy_intervals(dx)
    }
}

impl CollisionGrid {
    fn dy_intervals(&self, dx: i64) -> &[(i64, i64)] {
        if dx < self.delta.min_dx || self.delta.max_dx < dx {
            return &[];
        }
        let col = (dx - self.delta.min_dx) as usize;
        let begin = self.column_offsets[col];
        let end = self.column_offsets[col + 1];
        &self.intervals[begin..end]
    }
}
```

計算量：

- 初回: `grid 構築コスト + O(1) + O(k)`
- キャッシュ済み: `O(1) + O(k)`

`k` は該当 `dx` 列の `dy` interval 数。
戻り値取得自体は `O(1)` で、呼び出し側で intervals を走査する分が `O(k)`。

## new_old_clear / old_new_clear の区別

`pre.collision.crane(moving, fixed)` は方向付き。
内部では以下を使う。

```rust
dx = fixed.x - moving.x;
dy = fixed.y - moving.y;
```

そのため、`new` を `(x, y)` に置く場合は 2 種類を別々に扱う。

### new -> old

```rust
moving = new
fixed  = old

dx = old.x - x
dy = old.y - y
```

`crane_dy_intervals(new_orient, old_orient, dx)` が `[lo, hi]` を返した場合、

```text
lo <= old.y - y <= hi
```

なので、`y` の衝突区間は

```text
old.y - hi <= y <= old.y - lo
```

### old -> new

```rust
moving = old
fixed  = new

dx = x - old.x
dy = y - old.y
```

`crane_dy_intervals(old_orient, new_orient, dx)` が `[lo, hi]` を返した場合、

```text
lo <= y - old.y <= hi
```

なので、`y` の衝突区間は

```text
old.y + lo <= y <= old.y + hi
```

## y イベント

各 old block について、固定した `x` に対して以下のイベントを作る。

```rust
struct YEvent {
    y: i64,
    old_idx: usize,
    dir: HitDir,
    delta: i8,
}

enum HitDir {
    NewOld,
    OldNew,
}
```

`delta = +1` が衝突区間開始、`delta = -1` が終了。
終了イベントは閉区間 `[l, r]` に対して `r + 1` に置く。

```rust
// new -> old
for &(lo, hi) in pre.collision.crane_dy_intervals(new_orient, old_orient, old.x - x) {
    let l = old.y - hi;
    let r = old.y - lo;
    push_interval_event(l, r, old_idx, HitDir::NewOld, &mut events);
}

// old -> new
for &(lo, hi) in pre.collision.crane_dy_intervals(old_orient, new_orient, x - old.x) {
    let l = old.y + lo;
    let r = old.y + hi;
    push_interval_event(l, r, old_idx, HitDir::OldNew, &mut events);
}
```

fit range 外のイベントは不要なので、`[fit_range.min_y, fit_range.max_y]` に clamp してから追加する。

```rust
fn push_interval_event(
    l: i64,
    r: i64,
    old_idx: usize,
    dir: HitDir,
    events: &mut Vec<YEvent>,
    min_y: i64,
    max_y: i64,
) {
    let l = l.max(min_y);
    let r = r.min(max_y);
    if l > r {
        return;
    }
    events.push(YEvent { y: l, old_idx, dir, delta: 1 });
    if r < i64::MAX {
        events.push(YEvent { y: r + 1, old_idx, dir, delta: -1 });
    }
}
```

`events` は `y` 昇順に sort する。
ここはイベント数に対する sort なので、現行の「各 y ごとに forbidden sort」より軽い想定。

## hit state

各 old block について方向別の active count を持つ。

```rust
#[derive(Clone, Copy, Default)]
struct HitState {
    new_old: u8,
    old_new: u8,
}
```

bool ではなく count にする理由は、同じ方向で複数の `dy` interval が重なる可能性があるため。

イベント適用：

```rust
match event.dir {
    HitDir::NewOld => {
        if event.delta > 0 {
            states[event.old_idx].new_old += 1;
        } else {
            states[event.old_idx].new_old -= 1;
        }
    }
    HitDir::OldNew => {
        if event.delta > 0 {
            states[event.old_idx].old_new += 1;
        } else {
            states[event.old_idx].old_new -= 1;
        }
    }
}
```

`active_old_ids` も持つと、forbidden full-build 時に全 old block を見ずに済む。
ただし old block 数が十分小さい場合は、まず全 old block を走査する実装でよい。

## y スイープの評価点

`hit state` が変わらない y 区間では、以下は変わらない。

- forbidden intervals
- first feasible time
- tardiness
- score delta

変わるのは tie-break に使う `bbox_top` だけ。
そのため、その区間の最小 `y` だけ評価すればよい。

評価する y は：

- `fit_range.min_y`
- 各イベントの `y`

ただし `fit_range.max_y` を超えるイベントは評価しない。

スイープイメージ：

```rust
events.sort_unstable_by_key(|e| e.y);

let mut event_pos = 0;
let mut y = range.min_y;

loop {
    while event_pos < events.len() && events[event_pos].y == y {
        apply_event(events[event_pos], &mut states);
        event_pos += 1;
    }

    evaluate_y(y, &states);

    if event_pos >= events.len() {
        break;
    }
    y = events[event_pos].y;
    if y > range.max_y {
        break;
    }
}
```

注意：`y = range.min_y` より前に始まっているイベントは clamp により `range.min_y` に開始イベントとして入るため、初期状態は all zero でよい。

## forbidden interval の計算

現行の `add_forbidden_intervals_for_old()` から collision 判定を抜き、方向別 hit state を入力にする。

事前に old block ごとの時間情報を作る。

```rust
struct OldTimeInfo {
    a: i64,  // old.entry_time
    b: i64,  // old.exit_time
    p: i64,  // new processing_time
    ol: i64,
    or: i64,
}
```

`ol, or` は現行と同じ。

```rust
ol = old.entry_time - p + 1;
or = old.exit_time - 1;
```

`[min_t, max_t]` で clamp し、空ならその old block は時間的に影響しない。

```rust
fn add_forbidden_from_hit_state(
    info: OldTimeInfo,
    new_old_hit: bool,
    old_new_hit: bool,
    forbidden: &mut Vec<Interval>,
) {
    let new_old_clear = !new_old_hit;
    let old_new_clear = !old_new_hit;

    if new_old_clear && old_new_clear {
        return;
    }

    let ol = info.ol;
    let or = info.or;

    if !new_old_clear && !old_new_clear {
        forbidden.push((ol, or));
        return;
    }

    let (allow_l, allow_r) = if new_old_clear {
        ((info.a + 1).max(ol), (info.b - info.p - 1).min(or))
    } else {
        ((info.b - info.p + 1).max(ol), (info.a - 1).min(or))
    };

    if allow_l > allow_r {
        forbidden.push((ol, or));
        return;
    }
    if ol < allow_l {
        forbidden.push((ol, allow_l - 1));
    }
    if allow_r < or {
        forbidden.push((allow_r + 1, or));
    }
}
```

現行の分岐と対応：

```rust
new_old_clear && old_new_clear      => forbidden なし
!new_old_clear && !old_new_clear    => overlap 全体 forbidden
new_old_clear && !old_new_clear     => new -> old だけ通れる
!new_old_clear && old_new_clear     => old -> new だけ通れる
```

## first feasible time

interval 数が少ない前提なので、sort/merge せずに未ソート interval を線形探索する。

```rust
fn first_feasible_time_small(
    forbidden: &[Interval],
    min_t: i64,
    max_t: i64,
) -> Option<i64> {
    let mut t = min_t;

    loop {
        let mut next_t = t;

        for &(l, r) in forbidden {
            if l <= t && t <= r {
                if r == i64::MAX {
                    return None;
                }
                next_t = next_t.max(r + 1);
            }
        }

        if next_t == t {
            return Some(t);
        }
        if next_t > max_t {
            return None;
        }

        t = next_t;
    }
}
```

計算量は forbidden interval 数を `m` として最悪 `O(m^2)`。
ただし `m` が 1〜5 程度なら、`sort_unstable + merge` より定数倍で軽い可能性が高い。

## allocation 方針

内側ループで allocation しない。

- `events: Vec<YEvent>` は `x` ごとに `clear()` して再利用する。
- `forbidden: Vec<Interval>` は `evaluate_y()` ごとに `clear()` して再利用する。
- `states: Vec<HitState>` は bay 内 old block 数ぶん確保し、`x` ごとに zero clear する。

例：

```rust
let mut events = Vec::with_capacity(64);
let mut forbidden = Vec::with_capacity(16);
let mut states = vec![HitState::default(); bay_old_blocks.len()];
```

## insert_greedy 内の実装イメージ

```rust
for bay_id in 0..problem.bays.len() {
    let bay_old_blocks: Vec<ScheduledBlock> = schedule
        .iter()
        .copied()
        .filter(|old| old.bay_id == bay_id)
        .collect();

    let old_time_infos = build_old_time_infos(&bay_old_blocks, process_t, min_t, max_t);

    for &orient_idx in &pre.orientation_order_by_bbox[block_id] {
        let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
            continue;
        };

        let mut anchor_x = None;
        for x in range.min_x..=range.max_x {
            if let Some(anchor_x) = anchor_x {
                if x > anchor_x + params.x_buffer {
                    break;
                }
            }

            events.clear();
            states.fill(HitState::default());

            let new_orient = BlockOrient { block_id, orient_idx };

            for (old_idx, &old) in bay_old_blocks.iter().enumerate() {
                if old_time_infos[old_idx].is_none() {
                    continue;
                }

                let old_orient = BlockOrient {
                    block_id: old.block_id,
                    orient_idx: old.orient_idx,
                };

                for &(lo, hi) in pre.collision.crane_dy_intervals(new_orient, old_orient, old.x - x) {
                    push_interval_event(old.y - hi, old.y - lo, old_idx, HitDir::NewOld, &mut events, range.min_y, range.max_y);
                }
                for &(lo, hi) in pre.collision.crane_dy_intervals(old_orient, new_orient, x - old.x) {
                    push_interval_event(old.y + lo, old.y + hi, old_idx, HitDir::OldNew, &mut events, range.min_y, range.max_y);
                }
            }

            events.sort_unstable_by_key(|e| e.y);
            sweep_y_and_evaluate(...);
        }
    }
}
```

`crane_dy_intervals()` は `BlockOrient` を受けるため、未使用の `x`/`y` を持つダミー `BlockPlacement` は不要。
`dx` は呼び出し側から明示的に渡す。

## 注意点

- `new -> old` と `old -> new` の `dy` から `y` への変換式を混同しない。
- event sort は同じ `y` のイベントをすべて適用してから評価する。
- `range.min_y` より前から続く衝突は、イベントを clamp して `range.min_y` 開始にする。
- `anchor_x` の挙動は探索品質に影響するため、まずは現状維持する。
- `first_feasible_time_small()` は `forbidden` が少数という仮定に依存する。もし大きく増えるケースが見えたら、sort/merge 版と閾値で切り替える。

## 実装順

1. `collision.rs` に `crane_dy_intervals()` と `CollisionGrid::dy_intervals()` を追加。
2. `solver.rs` に `HitDir`, `YEvent`, `HitState`, `OldTimeInfo` を追加。
3. `add_forbidden_from_hit_state()` と `first_feasible_time_small()` を追加。
4. `insert_greedy` の内側を y event sweep に置き換える。
5. 現行の `get_insert_t()` と結果が大きくズレないことを小さなケースで確認する。
