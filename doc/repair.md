# 軽量 repair 近傍の実装方針

`src/solver.rs` の破壊再構築近傍を、bay 割当列挙や time candidate 探索に頼らない軽量な貪欲 repair に置き換える。

## 目的

現状の `try_remove_reinsert` は以下が重く、tardiness が発生しているケースで特に遅くなりやすい。

- bay assignment の DFS 列挙
- `allow_tardiness=false/true` の2 phase
- 少数の `time_candidates` を順に試す構造
- fixed bay 失敗後の全 bay fallback

新方針では、各 block の再挿入時に全 bay / orientation / `(x, y)` を試し、固定配置ごとに forbidden time intervals を作って、最小 feasible `t` を直接選ぶ。

## 削除・不要化するもの

新方式へ完全移行する場合、以下は不要になる。

- `BayAssignment`
- `enumerate_bay_assignments`
- `MAX_ENUMERATED_BAY_ASSIGNMENTS`
- `MAX_BAY_ASSIGNMENTS`
- `MAX_TIME_CANDIDATES`
- `LATE_TIME_EXTRA_MARGIN`
- `time_candidates`
- `allow_tardiness` 引数・2 phase

## precompute に追加するもの

`src/precompute.rs` の `Precompute` に block 面積を追加する。

```rust
pub block_area: Vec<f64>,
```

面積は block の全 orientation / layer の polygon area の最大値とする。

```text
block_area[block_id] = max(area(layer)) over all orientations and layers
```

polygon area は shoelace formula で計算する。向きに依存しないように絶対値を取る。

この値は removed block の再挿入順を決めるために使う。

## 新しい try_remove_reinsert の流れ

```text
1. schedule から removed_ids を取り除き、base と removed に分ける
2. removed の数が removed_ids と一致しなければ None
3. removed を block_area 降順に並べる
4. cur = base
5. loads = cur の bay workload
6. removed を順に処理する
   1. find_best_insert_position で block の最良挿入位置を探す
   2. 見つからなければ None
   3. 見つかれば cur に push し、loads を更新
7. Some(cur)
```

removed の並べ替えは以下のようにする。

```rust
removed.sort_by(|a, b| {
    pre.block_area[b.block_id]
        .total_cmp(&pre.block_area[a.block_id])
        .then(a.block_id.cmp(&b.block_id))
});
```

## block ごとの挿入探索

`find_best_insert_position` のような関数を作る。

入力のイメージ:

```rust
fn find_best_insert_position(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    schedule: &[ScheduledBlock],
    loads: &[f64],
) -> Option<ScheduledBlock>
```

探索順:

```text
for bay_id in 0..problem.bays.len():
  for (orient_rank, orient_idx) in pre.orientation_order_by_bbox[block_id]:
    fit_range がなければ continue
    for y in min_y..=max_y:
      for left in [true, false]:
        for x in left/right に寄せる順:
          fixed placement に対して最小 feasible t を探す
          見つかったら candidate として best 更新
```

`left/right に寄せる順` は既存実装と同じでよい。

```rust
let x = if left {
    range.min_x + xi as i64
} else {
    range.max_x - xi as i64
};
```

## 時刻範囲

各 block の探索範囲は以下に固定する。

```rust
let lo = block.release_time;
let cur_max_exit = schedule.iter().map(|s| s.exit_time).max().unwrap_or(lo);
let hi = cur_max_exit.max(lo);
```

`hi` に余分な margin は足さない。`t = max(cur_max_exit, release_time)` が safe entry になり、既存 block と時間的に重ならないため、配置可能性のためにそれより後ろを見る必要はない。

## 固定配置に対する feasible t の探し方

`bay_id`, `orient_idx`, `x`, `y` を固定したら、その配置で NG になる entry time の閉区間 `[l, r]` を列挙する。

対象は同じ bay の既存 block のみ。

```rust
for old in schedule.iter().filter(|old| old.bay_id == bay_id) {
    ...
}
```

新 block:

```text
new = [t, t + p)
```

既存 block:

```text
old = [a, b)
```

時間が重なり得る entry time 全体は、整数 t の閉区間で:

```text
overlap = [a - p + 1, b - 1]
```

この区間が `[lo, hi]` と交差しなければ、その old は無視できる。

### crane 判定

固定配置に対して以下を計算する。

```rust
let new_old_clear = pre.collision.crane(new_place, old_place) == CollisionResult::Clear;
let old_new_clear = pre.collision.crane(old_place, new_place) == CollisionResult::Clear;
```

両方 clear なら、その old から forbidden interval は発生しない。

### 時間関係ごとの forbidden interval

現行 `can_insert` と同じ意味になるように、次の3種類を扱う。

#### 1. old contains new

```text
a < t && t + p < b
=> t in [a + 1, b - p - 1]
```

この区間では `new -> old` だけ必要。したがって `!new_old_clear` なら forbidden に追加する。

#### 2. new contains old

```text
t < a && b < t + p
=> t in [b - p + 1, a - 1]
```

この区間では `old -> new` だけ必要。したがって `!old_new_clear` なら forbidden に追加する。

#### 3. ABAB / same-time conservative

上記2種類以外で overlap する区間。現行実装と同じく、同時刻の ENTRY/EXIT は操作順に依存するため保守的に両方向を見る。

```text
overlap - old_contains_new - new_contains_old
```

この部分は `!new_old_clear || !old_new_clear` なら forbidden に追加する。

実装では、まず `overlap` を `[lo, hi]` に clamp し、その中から old-contains-new / new-contains-old の区間を切り出す。残りの区間が ABAB/same-time 部分になる。

### forbidden intervals から t を選ぶ

1. forbidden intervals を `[lo, hi]` に clamp する
2. 空区間は捨てる
3. `l` 昇順で sort
4. merge する
5. `[lo, hi]` のうち forbidden に含まれない最小の `t` を返す

`tardiness = max(0, t + p - due_date)` は `t` に対して単調非減少なので、最小 feasible `t` が最小 tardiness の `t` になる。

返り値は `Option<i64>` でよい。

### forbidden interval 生成の実装手順

実装では、閉区間を `(i64, i64)` で表す。

```rust
type Interval = (i64, i64);
```

まず、区間を `[lo, hi]` に clamp する小さい helper を用意するとよい。

```rust
fn clamp_interval(l: i64, r: i64, lo: i64, hi: i64) -> Option<Interval> {
    let l = l.max(lo);
    let r = r.min(hi);
    if l <= r { Some((l, r)) } else { None }
}
```

old block 1個から forbidden intervals を追加するときは、`overlap` を containment 区間の境界で分割する。各小区間では時間関係が一定なので、代表点 `t = l` で必要な crane 方向を判定すればよい。禁止区間はすべて `forbidden` に push し、最後にまとめて sort + merge する。

```rust
fn add_forbidden_intervals_for_old(
    pre: &Precompute,
    new_block: ScheduledBlock,
    old: ScheduledBlock,
    lo: i64,
    hi: i64,
    forbidden: &mut Vec<Interval>,
) {
    let p = new_block.exit_time - new_block.entry_time;
    let a = old.entry_time;
    let b = old.exit_time;

    let Some((ol, or)) = clamp_interval(a - p + 1, b - 1, lo, hi) else {
        return;
    };

    let new_place = BlockPlacement {
        block_id: new_block.block_id,
        orient_idx: new_block.orient_idx,
        x: new_block.x,
        y: new_block.y,
    };
    let old_place = BlockPlacement {
        block_id: old.block_id,
        orient_idx: old.orient_idx,
        x: old.x,
        y: old.y,
    };

    let new_old_clear = pre.collision.crane(new_place, old_place) == CollisionResult::Clear;
    let old_new_clear = pre.collision.crane(old_place, new_place) == CollisionResult::Clear;
    if new_old_clear && old_new_clear {
        return;
    }

    let mut points = vec![ol, or + 1];

    if let Some((l, r)) = clamp_interval(a + 1, b - p - 1, ol, or) {
        points.push(l);
        points.push(r + 1);
    }
    if let Some((l, r)) = clamp_interval(b - p + 1, a - 1, ol, or) {
        points.push(l);
        points.push(r + 1);
    }

    points.sort_unstable();
    points.dedup();

    for w in points.windows(2) {
        let l = w[0];
        let r = w[1] - 1;
        if l > r {
            continue;
        }

        let t = l;
        let ok = if a < t && t + p < b {
            new_old_clear
        } else if t < a && b < t + p {
            old_new_clear
        } else {
            new_old_clear && old_new_clear
        };

        if !ok {
            forbidden.push((l, r));
        }
    }
}
```

`new_block.entry_time` はこの時点では未定なので、呼び出し側で一時的に `entry_time=0, exit_time=p` のように入れてよい。上の処理では `p` と配置情報だけを使う。

`or + 1` や `r + 1` は問題制約上 overflow しない想定でよい。気になる場合は `saturating_add(1)` を使う。

全 old から intervals を集めたら、merge する。

```rust
fn merge_intervals(intervals: &mut Vec<Interval>) {
    intervals.sort_unstable_by_key(|&(l, r)| (l, r));
    let mut merged: Vec<Interval> = Vec::new();
    for &(l, r) in intervals.iter() {
        if let Some(last) = merged.last_mut() {
            if l <= last.1 + 1 {
                last.1 = last.1.max(r);
                continue;
            }
        }
        merged.push((l, r));
    }
    *intervals = merged;
}
```

最小 feasible `t` は以下で取る。

```rust
fn first_feasible_time(mut forbidden: Vec<Interval>, lo: i64, hi: i64) -> Option<i64> {
    merge_intervals(&mut forbidden);
    let mut t = lo;
    for (l, r) in forbidden {
        if t < l {
            return Some(t);
        }
        if t <= r {
            t = r + 1;
        }
        if t > hi {
            return None;
        }
    }
    if t <= hi { Some(t) } else { None }
}
```

固定配置に対する時刻探索は、これらを組み合わせる。

```rust
fn best_time_for_fixed_placement(
    pre: &Precompute,
    new_block: ScheduledBlock,
    schedule: &[ScheduledBlock],
    lo: i64,
    hi: i64,
) -> Option<i64> {
    let mut forbidden = Vec::new();
    for &old in schedule.iter().filter(|old| old.bay_id == new_block.bay_id) {
        add_forbidden_intervals_for_old(pre, new_block, old, lo, hi, &mut forbidden);
    }
    first_feasible_time(forbidden, lo, hi)
}
```

実装後の検証として、debug build では `best_time_for_fixed_placement` で得た `t` を入れた `ScheduledBlock` について、旧 `can_insert(pre, scheduled, schedule)` が `true` になることを `debug_assert!` で確認するとよい。

## candidate の評価・tie-break

各 `(bay_id, orient_idx, x, y)` で feasible `t` が見つかったら candidate を作る。

主評価は tardiness。

```rust
let tardiness = (t + block.processing_time - block.due_date).max(0);
```

同点 tie-break は以下の順に小さい方を選ぶ。

```text
1. tardiness
2. block を追加することによる weighted delta(obj2 + obj3)
3. entry_time
4. orientation bbox 小さい順の rank
5. x
6. y
```

`delta(obj2 + obj3)` は solver の score と合わせて重み付きにする。

```rust
let current_obj2 = normalized_imbalance(pre, loads);
let mut next_loads = loads.to_vec();
next_loads[bay_id] += block.workload as f64;
let next_obj2 = normalized_imbalance(pre, &next_loads);

let delta_obj2 = problem.weights.w2 * (next_obj2 - current_obj2);
let delta_obj3 = problem.weights.w3 * pre.pref_penalty[block_id][bay_id] as f64;
let delta_obj23 = delta_obj2 + delta_obj3;
```

効率化のため、`current_obj2` は block ごとの探索開始時に一度だけ計算してよい。

`delta_obj23` は `f64` なので比較には `total_cmp` を使う。

candidate 用に小さい struct を作ると実装しやすい。

```rust
struct InsertCandidate {
    scheduled: ScheduledBlock,
    tardiness: i64,
    delta_obj23: f64,
    orient_rank: usize,
}
```

比較順:

```text
tardiness
then delta_obj23 by total_cmp
then scheduled.entry_time
then orient_rank
then scheduled.x
then scheduled.y
```

## repair 中の loads 更新

`try_remove_reinsert` では `base` から `loads` を作る。

block を1つ挿入したら、必ず以下を更新する。

```rust
loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
cur.push(scheduled);
```

`find_best_insert_position` には現在の `loads` を渡し、候補評価の `delta_obj2` に使う。

## 既存 can_insert との整合性

新方式では `can_insert` を時刻候補ごとに呼ばないが、forbidden interval の意味は現行 `can_insert` と一致させる。

特に注意する点:

- block の占有時間は半開区間 `[entry, exit)`
- forbidden interval は整数 entry time の閉区間 `[l, r]`
- 同時刻 ENTRY/EXIT が絡む ABAB/same-time は保守的に両方向 crane を要求する
- old contains new / new contains old の不等号は現行 `can_insert` の strict containment と合わせる

実装後、可能なら debug 用に新方式で得た `scheduled` が旧 `can_insert(pre, scheduled, schedule)` を満たすことを確認するとよい。

## 懸念点

### 早く置きすぎる可能性

`t` の評価が tardiness のみで、同点では `entry_time` 小を優先するため、on-time の範囲ではかなり早く置く傾向がある。後続 block の邪魔になる可能性はあるが、まずはこの仕様で実装する。

### greedy repair の順序依存

removed を面積降順で貪欲に入れるため、後続 block の自由度は直接考慮しない。軽量近傍として割り切る。

### hi を広げないことによる探索範囲

`hi = max(cur_max_exit, release_time)` とする。safe entry は必ず含まれるため、配置可能性のための margin は不要。ただし、将来的に「少し後ろにずらすことで後続 removed block を置きやすくする」効果を狙う場合は、別途 margin を再検討する。

### 区間実装のバグリスク

最もバグりやすいのは forbidden interval の生成。特に off-by-one と同時刻処理に注意する。
