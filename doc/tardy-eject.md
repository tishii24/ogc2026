# tardy-eject 近傍 実装方針

## 目的

tardiness が発生しているブロックを、より早い entry time に押し込む。
その際、干渉する既存ブロックを一時削除する。最初に選んだ tardy block だけ tardiness 改善を必須にし、追加で削除されたブロックは面積が大きい順に通常の `insert_greedy` で再挿入する。

狙いは、大きく遅れているブロックを局所的に前倒しし、干渉の連鎖を小さく抑えながら少しずつ tardiness を改善すること。

## 基本方針

`insert_greedy` の探索ループを使い回せるように、以下に分離する。

- `(bay, orient, x, y)` の候補列挙
- old block ごとの crane 干渉状態の sweep
- forbidden interval の構築
- 候補評価

通常の `insert_greedy` は、候補評価として「干渉ゼロの最早 entry time を選ぶ」評価を渡す。
tardy-eject 近傍は、候補評価として「tardiness が改善する entry time のうち、削除面積が小さいものを選ぶ」評価を渡す。

既存の `insert_greedy` の外部挙動は変えない。

## 追加する定数

`src/solver.rs` 上部に置く。

```rust
const TARDY_EJECT_AREA_POWER: f64 = 1.5;
const TARDY_EJECT_MAX_REMOVED: usize = 4;
const TARDY_EJECT_MAX_QUEUE: usize = 10;
const TARDY_EJECT_POOL_SIZE: usize = 16;
```

`TARDY_EJECT_AREA_POWER` は `sum(area^a)` の `a`。
まずは `1.5` 程度でよい。

`TARDY_EJECT_MAX_REMOVED` は一回の近傍で追加 eject してよい最大ブロック数。最初に選んで `cur` から外す tardy block は数えない。
`TARDY_EJECT_MAX_QUEUE` は削除キューの暴走防止。
`TARDY_EJECT_POOL_SIZE` は tardiness 上位から対象を選ぶための pool サイズ。

`NeighborKind` / `NEIGHBOR_PROBS` / `NEIGHBOR_KIND_COUNT` を更新し、近傍確率はまず `0.2` にする。

## データ構造

### forbidden interval に干渉元を持たせる

現状の `Interval = (i64, i64)` だけだと、どの old block が原因か分からない。
候補評価で削除対象を復元するため、以下を追加する。

```rust
struct BlockForbidden {
    old_idx: usize,
    interval: Interval,
}
```

`old_idx` は `bay_old_blocks` 内の index。
削除対象の `block_id` は `bay_old_blocks[old_idx].block_id` から取る。

通常の `insert_greedy` では `interval` だけ見ればよい。

### search_insert の評価結果

候補評価が「配置」と「削除対象」を返せるようにする。

```rust
struct InsertSearchResult {
    scheduled: ScheduledBlock,
    removed_ids: Vec<usize>,
    key: InsertSearchKey,
}
```

`InsertSearchKey` は通常挿入と tardy-eject で別にしてもよい。
YAGNI を優先するなら、既存の `InsertCandidate` は通常挿入用に残し、tardy-eject 用には別 struct を作る。

```rust
struct TardyEjectCandidate {
    scheduled: ScheduledBlock,
    removed_ids: Vec<usize>,
    removed_area_sum: f64,
    after_tardiness: i64,
    bbox_right: f64,
    bbox_top: f64,
}
```

## insert_greedy のリファクタ

### 目標の関数境界

探索本体を `search_insert` に切り出す。

```rust
fn search_insert<E>(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    schedule: &[ScheduledBlock],
    loads: &[f64],
    params: InsertSearchParams,
    eval: E,
) -> Option<InsertSearchResult>
where
    E: FnMut(InsertEvalInput) -> Option<InsertSearchResult>,
```

ただし Rust の lifetime や borrow が面倒なら、最初は callback 化しすぎなくてよい。
以下のように「候補列挙 helper」を作って、通常版と tardy-eject 版でそれぞれ評価する形でもよい。

```rust
fn enumerate_insert_candidates(
    ...,
    on_candidate: impl FnMut(InsertEvalInput),
)
```

重要なのは、`bay / orient / x / y` の列挙と y sweep を二重実装しないこと。

### InsertEvalInput

候補評価に必要な情報を渡す。

```rust
struct InsertEvalInput<'a> {
    scheduled_base: ScheduledBlock,
    block: &'a Block,
    bay_old_blocks: &'a [ScheduledBlock],
    forbiddens: &'a [BlockForbidden],
    delta_obj23: f64,
    bounds: Boundsf,
}
```

`scheduled_base` は `entry_time` / `exit_time` 未確定でもよい。
`block.processing_time` から `exit_time = entry_time + processing_time` を作る。

`forbiddens` は、その `(bay, orient, x, y)` における old block 由来の禁止 interval 群。

### 通常 insert_greedy の評価

現状と同じ評価を維持する。

```rust
let entry_time = first_feasible_time(intervals, min_t, max_t)?;
let tardiness = (entry_time + process_t - due_date).max(0);
score_delta = w1 * tardiness + delta_obj23;
key = (score_delta, entry_time, bbox_right, bbox_top, block_id);
```

`first_feasible_time` には `forbiddens.iter().map(|f| f.interval)` を渡す。

既存の `anchor_y` による早期打ち切りは通常挿入・tardy-eject 用探索のどちらでも維持してよい。

## tardy-eject の候補評価

### 改善条件

対象ブロックの現在 tardiness を `before_tardiness` とする。

```rust
let before_tardiness = (old.exit_time - block.due_date).max(0);
```

候補 entry time `t` は以下を満たす必要がある。

```text
after_tardiness = max(0, t + processing_time - due_date)
after_tardiness < before_tardiness
```

`before_tardiness == 0` のブロックは tardy-eject 評価の対象外。
この改善条件を要求するのは、最初に選んだ tardy block だけにする。追加 eject されたブロックは、同じ改善条件を要求せず通常の `insert_greedy` で戻す。

探索する `t` の範囲は以下。

```rust
let min_t = block.release_time;
let max_t = old.exit_time - block.processing_time - 1;
```

これは `after_tardiness < before_tardiness` の十分な上限になる。
より直接的には、`after_tardiness < before_tardiness` を評価時に判定する。

### 評価キー

候補はまず tardiness 改善を必須条件にする。
その上で以下の順に小さいものを選ぶ。

```text
(removed_area_sum, after_tardiness, entry_time, bbox_right, bbox_top, block_id)
```

`removed_area_sum` は、その `t` に刺さっている forbidden interval の干渉元 block の面積和。

```rust
removed_area_sum = sum(pre.block_area[removed_id].powf(TARDY_EJECT_AREA_POWER))
```

同じ block が複数 interval で刺さる可能性があるため、`removed_ids` は重複排除する。
小さい配列なので `Vec` + `contains` でよい。

### t の選び方

全日付を舐めず、`block.release_time..=latest_t` の範囲で forbidden interval の active 集合が変わる時刻だけを見る。

```rust
let min_t = block.release_time;
let max_t = old.exit_time - block.processing_time - 1;
```

各 `BlockForbidden { old_idx, interval: (l, r) }` を `[min_t, max_t]` に clip し、以下のイベントを作る。

```text
l     : old_idx を active に追加
r + 1 : old_idx を active から削除
```

sweep 中は `old_idx` ごとの refcount を持つ。同じ block が複数 interval で刺さる可能性があるため、refcount が `0 -> 1` になった時だけ削除対象へ追加し、`1 -> 0` になった時だけ外す。
`removed_area_sum` もこのタイミングで増減すればよい。

候補時刻は以下だけ評価する。

- `min_t`
- `[min_t, max_t]` 内の各イベント時刻

同じ active 集合が続く区間内では、`after_tardiness` は `t` が小さいほど良い。そのため各区間の左端、つまりイベント時刻だけ見れば十分。

各候補時刻 `t` について:

1. その時刻のイベントを適用し、active な `old_idx` 集合を更新する
2. `after_tardiness` を計算し、改善しなければ捨てる
3. active な `old_idx` から `removed_ids` を作る
4. `removed_area_sum` を使って評価キーで best 更新

干渉ゼロで改善できる候補は `removed_area_sum = 0` になり、自然に最優先される。

## tardy-eject 近傍本体

関数名案:

```rust
fn try_tardy_eject_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>>
```

### 1. 対象ブロック選択

`schedule` から tardiness 正のブロックだけを集める。

最初は以下の単純な選び方でよい。

- tardiness が大きい順に上位 pool を作る
- pool からランダム選択

例:

```rust
const TARDY_EJECT_POOL_SIZE: usize = 16;
```

評価対象:

```rust
tardiness = (s.exit_time - problem.blocks[s.block_id].due_date).max(0)
```

### 2. cur と削除キューを作る

選んだ tardy block を `cur` から外し、再挿入対象にする。

```text
cur = schedule - selected
queue = [selected]
removed_total = 0
```

queue は block id ではなく元の `ScheduledBlock` を持つ。
bay/orient/x/y は初期値として使える。

### 3. 面積降順で再挿入

削除キューが空でない間、面積が大きい block から取り出す。

```text
while queue not empty:
  old = pop largest area
  if old is initial tardy block:
    result = search_tardy_eject_insert(...)
    cur から result.removed_ids を削除
    removed blocks を queue に追加
  else:
    result.scheduled = insert_greedy(...)
  result.scheduled を cur に追加
```

`search_tardy_eject_insert` は新規に追加する helper 名の案。
`insert_greedy` と同じ候補列挙を使い、初回 tardy block 用の評価として `TardyEjectCandidate` を返す。

重複管理:

- `in_queue[block_id]`
- `removed_or_processing[block_id]`

すでに queue にある block を再追加しない。
すでに処理済み、処理中、または `cur` に存在しない block を削除しようとする候補は無効扱いにする。

### 4. 失敗条件

以下の場合は `None`。

- tardy block が存在しない
- 挿入候補が見つからない
- 追加 eject 数が `TARDY_EJECT_MAX_REMOVED` を超えた
- queue 長が `TARDY_EJECT_MAX_QUEUE` を超えた
- 最終 schedule の長さが `problem.blocks.len()` と一致しない

### 5. 最終採用判定

他の近傍と同様に採用・判定する。

## 実装順序

1. `BlockForbidden` を追加する
2. `add_forbidden_from_hit_state` の結果を `BlockForbidden` として作れるようにする
3. `insert_greedy` の探索本体を候補列挙 helper に切り出す
4. 通常 `insert_greedy` を helper 経由にして、スコアが変わらないことを確認する
5. tardy-eject 用の候補評価を追加する
6. `try_tardy_eject_neighbor` を追加する
7. `NeighborKind` / `NEIGHBOR_PROBS` / `NEIGHBOR_KIND_COUNT` を更新する

## 注意点

- `entry_time == exit_time` にはならない。`exit_time = entry_time + processing_time` を常に守る。
- 同じ時刻の EXIT before ENTRY という前提は、既存の forbidden interval ロジックに合わせる。
- `removed_ids` に自分自身を入れない。
- `BlockForbidden.old_idx` は `bay_old_blocks` の index なので、`cur` の index と混同しない。
- `score_schedule` は feasibility を検査しない。近傍の途中で干渉を許しても、最終 schedule では全ブロックを通常挿入または tardy-eject 評価で再挿入済みにする。
- 最初は obj2 / obj3 の細かい差分最適化を入れない。最後の `score_schedule` 改善で採用する。
