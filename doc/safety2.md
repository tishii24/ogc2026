### 1. 優先度が高い箇所

| 箇所 | 発生条件 | 修正案 |
|---|---|---|
| `src/collision.rs:395` `build_shape_geom()` | layerが3点未満、面積0、自己交差などで凸分割に失敗 | `assert!`を削除し、`build_shape_geom() -> Result<_, String>`にする。block/orientation/layer番号付きでエラーを返す |
| `src/collision.rs:580`, `src/collision.rs:710` | 退化polygonやMinkowski hullが3点未満 | 同様にcollision precompute全体をfallibleにし、`Precompute::build() -> Result`として`solver::solve()`まで伝播 |
| `src/annealing.rs:552` | reheat有効だが、現在regimeの `reheat_local_best_score_per_block_scale`が未設定 | パラメータ検証でpositive/zeroの両方を必須にする。実行側も`unwrap()`せず`Option`を処理 |
| `src/preoptimize.rs:447` | blocksが空で、空scheduleに対して`.max().unwrap()` | 問題入力としてblocks非空を検証するか、空問題を明示的に処理 |

### 2. 中～低優先度の箇所

| 箇所 | リスク | 修正案 |
|---|---|---|
| `src/solver.rs:956,988` | seed候補poolが空の場合の`seed_pool[0]` | `.first().copied()?`で近傍失敗として返す |
| `src/solver/optimize.rs:207`、`src/preoptimize.rs:972` | `states.pop().unwrap()` | `AnnealingDelegate::finish()`を`Result`化。ただし現在は必ず1 stateなので低優先度 |
| `src/collision.rs:146,190` | 未前計算pairへの`expect()` | `Option`/`Result`化。ただし現在は内部生成IDにより維持される不変条件 |
| `src/insert.rs:949` | 不正な内部XEventで`unreachable!()` | `Result`化可能。ただしイベント値は同ファイルで限定生成されており通常は安全 |

`last_mut().unwrap()`など、直前の非空確認で安全な箇所もあります。すべて機械的に `.get()`へ変えるより、外部境界を検証し、内部不変条件由来の箇所だけ最後に整理する方が性能・可読性を維持できます。
