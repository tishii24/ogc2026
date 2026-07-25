`main.rs`から通常実行で到達する経路を対象に、`unwrap`・`assert`・配列index・算術境界などを分類します。明示的panicだけでなく、入力や探索失敗で発生し得る箇所も含めて、修正優先度と案を整理します。

初期監査では、公式入力では成立している内部不変条件よりも、未検証パラメータと形状入力が主なpanic要因でした。残りの明示的panic箇所と、入力検証でまとめて防げる範囲を確認します。

結論として、`train`＋`params/default.yaml`の通常実行で直ちに起きそうなpanicは見つかりませんでした。ただし外部入力・パラメータを十分検証していないため、形状不正や設定ミスで確実にpanicする箇所があります。まず入力検証を一か所に集約するのが最も効果的です。

### 1. 優先度が高い箇所

| 箇所 | 発生条件 | 修正案 |
|---|---|---|
| `src/collision.rs:395` `build_shape_geom()` | layerが3点未満、面積0、自己交差などで凸分割に失敗 | `assert!`を削除し、`build_shape_geom() -> Result<_, String>`にする。block/orientation/layer番号付きでエラーを返す |
| `src/collision.rs:580`, `src/collision.rs:710` | 退化polygonやMinkowski hullが3点未満 | 同様にcollision precompute全体をfallibleにし、`Precompute::build() -> Result`として`solver::solve()`まで伝播 |
| `src/params.rs:183` | 温度候補のscaleを全て省略 | `.reduce(...).unwrap()`を `ok_or_else(...)`に変更し、`AnnealingParamsConfig::make() -> Result`にする |
| `src/annealing.rs:357` | `worker_count == 0`または`exchange_interval == 0` | `SolverParams::validate()`で拒否し、`Annealer::run()`の`assert!`も`Result`化 |
| `src/annealing.rs:552` | reheat有効だが、現在regimeの `reheat_local_best_score_per_block_scale`が未設定 | パラメータ検証でpositive/zeroの両方を必須にする。実行側も`unwrap()`せず`Option`を処理 |
| `src/solver.rs:805` | `max_removed_blocks < min_removed_blocks` | パラメータ検証で大小関係を保証 |
| `src/solver.rs:915` | `remove_seed_per_block`の下限が0 | `1 <= min <= max < usize::MAX`を検証。0だと無限ループやunderflowにつながる |
| `src/solver.rs:988` | `remove_entry_seed_candidate_count == 0` | 1以上を必須にするか、空なら近傍生成を`None`で終了 |
| `src/preoptimize.rs:447` | blocksが空で、空scheduleに対して`.max().unwrap()` | 問題入力としてblocks非空を検証するか、空問題を明示的に処理 |
| `src/solver_util.rs:122` | neighbor確率の合計が0、NaN、または全て非正 | 確率を有限・非負・合計正として検証。`sample_neighbor()`も`Option`を返せる形にする |

`train`については、先ほど確認したとおり2点layer・面積0のlayerはありません。ただしデシリアライザは空layerを除外するだけで、3点未満・面積0・自己交差を拒否していないため、一般入力ではcollisionの`assert!`に到達します。

### 2. 一括して追加したい入力検証

`Problem::validate()`を追加し、`solver::solve()`の最初に呼ぶのがよいです。

```text
bays:
  - 1件以上
  - width > 0, height > 0
  - width * heightがoverflowしない

blocks:
  - 1件以上
  - processing_time > 0
  - release_time + processing_timeがoverflowしない
  - bay_preferences.len() == bays.len()
  - shapeが1件以上

layers:
  - 頂点数 >= 3
  - 面積 > EPS
  - 座標が有限
  - 凸分割・三角形分割が可能
```

現在は `PreoptimizePrecompute::build()`に一部検証がありますが、`Precompute::build()`が先に実行されます。そのため、検証は両precomputeより前へ移す必要があります。

### 3. 一括して追加したいパラメータ検証

`SolverParams::load()`の最後に `params.validate()?`を追加します。

```text
runtime:
  worker_count >= 1
  local_search_time_buffer_secondsが有限・非負

annealing:
  exchange_interval >= 1
  温度scaleが各regimeで1つ以上
  全scaleが有限かつ正
  reheatを使う場合:
    stagnation_iterations >= 1
    duration_iterations >= 1
    positive/zero両方のreheat scaleが存在

neighbor:
  min_removed_blocks <= max_removed_blocks
  remove_seed_per_block: 1 <= min <= max
  remove_entry_seed_candidate_count >= 1
  各確率・weightが有限・非負
  neighbor確率の合計 > 0
  各rangeでmin <= max

insert:
  skip probabilityが0～1
  random progress powerが有限・非負
```

CLIの `timelimit`も `parse::<f64>()`だけでは `NaN`や`inf`を受理するため、`is_finite() && timelimit > 0.0`を確認すべきです。`NaN`は時間比較を壊し、ループが終了しなくなる可能性があります。

### 4. 中～低優先度の箇所

| 箇所 | リスク | 修正案 |
|---|---|---|
| `src/tracing.rs:47-98` | `trace-annealing`有効時、権限不足・ディスク容量不足でI/Oの`.unwrap()`がpanic | `AnnealingTraceWriter::new/write`を`io::Result`化 |
| `src/annealing.rs:48-71`、`src/preoptimize.rs:412`、`src/solver.rs:282` | Mutex poisonで二次panic | エラー伝播、または `unwrap_or_else(PoisonError::into_inner)` |
| `src/main.rs:19,49`、`log!` | broken pipeやstderr/stdout書込み失敗 | `writeln!`の結果を`Result`として処理。診断ログは書込み失敗を無視してもよい |
| `src/precompute.rs:270`、`src/preoptimize.rs:72` | `bay.width * bay.height`の整数overflow | 乗算前に`f64`または`i128`へ変換 |
| 各時刻計算 | `entry_time + processing_time`などのoverflow | 時刻計算を`checked_add/sub`、スコア差を`i128`化 |
| `src/solver.rs:956,988` | seed候補poolが空の場合の`seed_pool[0]` | `.first().copied()?`で近傍失敗として返す |
| `src/solver/optimize.rs:207`、`src/preoptimize.rs:972` | `states.pop().unwrap()` | `AnnealingDelegate::finish()`を`Result`化。ただし現在は必ず1 stateなので低優先度 |
| `src/collision.rs:146,190` | 未前計算pairへの`expect()` | `Option`/`Result`化。ただし現在は内部生成IDにより維持される不変条件 |
| `src/insert.rs:949` | 不正な内部XEventで`unreachable!()` | `Result`化可能。ただしイベント値は同ファイルで限定生成されており通常は安全 |

`last_mut().unwrap()`など、直前の非空確認で安全な箇所もあります。すべて機械的に `.get()`へ変えるより、外部境界を検証し、内部不変条件由来の箇所だけ最後に整理する方が性能・可読性を維持できます。

### 5. 推奨する修正順

1. `timelimit`、`Problem::validate()`、`SolverParams::validate()`を追加  
2. collision/precomputeの`assert!`を`Result`へ変更  
3. reheat・温度候補・seed pool・neighbor samplingの`unwrap()`を除去  
4. tracingとstdout/stderrのI/Oエラーを処理  
5. Mutex poisonと内部不変条件の`unwrap()`を整理  

なお、`Cargo.toml`はreleaseで `panic = "abort"`です。releaseでは一度panicすると`catch_unwind`で回収できないため、トップレベルでpanicを捕捉する方法ではなく、panic箇所を`Result`や近傍失敗の`None`へ変える必要があります。
