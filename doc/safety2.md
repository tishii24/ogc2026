### 1. 優先度が高い箇所

| 箇所 | 発生条件 | 修正案 |
|---|---|---|
| `src/collision.rs:395` `build_shape_geom()` | layerが3点未満、面積0、自己交差などで凸分割に失敗 | `assert!`を削除し、`build_shape_geom() -> Result<_, String>`にする。block/orientation/layer番号付きでエラーを返す |
| `src/collision.rs:580`, `src/collision.rs:710` | 退化polygonやMinkowski hullが3点未満 | 同様にcollision precompute全体をfallibleにし、`Precompute::build() -> Result`として`solver::solve()`まで伝播 |

### 2. 中～低優先度の箇所

| 箇所 | リスク | 修正案 |
|---|---|---|
| `src/solver/optimize.rs:207`、`src/preoptimize.rs:972` | `states.pop().unwrap()` | `AnnealingDelegate::finish()`を`Result`化。ただし現在は必ず1 stateなので低優先度 |
| `src/collision.rs:146,190` | 未前計算pairへの`expect()` | `Option`/`Result`化。ただし現在は内部生成IDにより維持される不変条件 |
| `src/insert.rs:949` | 不正な内部XEventで`unreachable!()` | `Result`化可能。ただしイベント値は同ファイルで限定生成されており通常は安全 |
