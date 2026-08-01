結論として、最大の改善余地はSAの細かな温度調整ではなく、**①時刻探索が「最早搬入・最短滞在」に固定されていること、②preoptimizeの良い抽象解を幾何解へ変換する際に大きく劣化すること、③LargeReconstructが試行比率以上にCPU時間を消費していること**です。まず再構築・挿入処理を高速化して tardiness 改善に直結するLNSへ作り替え、その後に時刻表現と並列探索を広げる順序が最も効果的です。

## 1. ログから分かる現状

| 観測 | 内容 |
|---|---|
| 全体性能 | 完全な31ケースでは `074` が objective合計 `87,429,260`、raw tardiness `9,212`。同一パラメータの `074-r` は `89,209,987`、`9,320` で約2%悪化しており、並列実行の揺らぎが大きいです。 |
| tardinessケースへの偏り | `074-r` の tardiness-positive 18ケースだけで objective `88,196,756`、T=`9,320`。zero 13ケースは objective `1,013,231`、T=`0` です。改善余地はほぼpositive側に集中しています。 |
| 反復密度 | positiveケースの中央値は constrained/global が `0.537M / 0.760M` iterations、zeroケースは `1.406M / 1.165M`。難しいケースほど候補生成に時間を使い、反復数が落ちています。 |
| 抽象解からの劣化 | `prob_40` は abstract `1.409M` → 幾何初期解 `2.086M` → 最終 `1.594M`。`prob_38` は `22.875M` → `44.368M` → `34.378M`。短時間性能の最大の損失点は初期幾何再構築です。 |
| まだ探索は進む | 60秒から300秒へ延ばすと、`prob_38` のTは約9.7%、`prob_40` は約12.7%改善しています。完全な局所停止ではなく、現在の改善速度が遅い状態です。 |

なお、目的関数合計は分析用であり、公式順位はケースごとの相対順位です。チューニングではraw合計より、ケース正規化したregretや順位相当指標を重視すべきです。

## 2. iterationを増やすための高速化

最優先は `LargeReconstruct` です。現在の重みでは選択率は約2.94%ですが、`076/prob_40` では近傍CPU時間の **constrainedで約79%、globalで約72%** を消費しています。globalではLargeが約10ms/試行なのに対し、Shiftは約0.048msです。

| 優先度 | 改善 |
|---|---|
| **P0** | `PlacementXScanner::new()` が挿入ごと・ベイごとにschedule全体をfilterして再確保しています（`ogc2026/src/solver/placement_scan.rs:253-301`）。状態側に`bay_members`を持ち、scanner用workspaceをworker単位で再利用します。 |
| **P0** | LargeReconstructでは「多数の固定block＋7〜13個の差分」という構造なので、固定baseに対する禁止区間・配置候補を一度計算し、追加blockのdeltaだけを合成します。`ogc2026/doc/idea.md` の方針が適切です。 |
| **P0** | `first_feasible_time()` はx区間ごとにactive禁止区間を先頭からmergeしています（`placement_scan.rs:168-215`）。時刻端点を圧縮し、区間cover更新＋最初の非cover時刻を返す構造にすると、密集ケースの二乗的走査を抑えられます。 |
| **P1** | 全近傍で`Vec<ScheduledBlock>`を再構築し、候補成功後にobjectiveとhashを全件再計算しています（`optimize.rs:235-285`）。block-id indexed state、loads、Z1/Z3、hashを保持し、変更した1〜k blockだけ差分更新します。 |
| **P1** | lazy collision cacheは両方向を同時構築するのに有向pairを別々に保持し、worker間で同じmissを重複計算し得ます（`collision.rs:121-176`）。unordered pairへの統合と一度だけの初期化が有効です。 |

単純なiteration数ではなく、ログに **「shared-best改善量 / CPU秒」** を近傍別に追加して配分すべきです。`prob_40` ではShiftやMoveの方がLargeより即時改善効率が高く、Largeは停滞時の脱出用に時間予算を限定する構成が合います。

## 3. より広範囲を探索するための根本変更

- **時刻探索空間を広げる**  
  全生成経路が `EXIT = ENTRY + P` で、各幾何配置について最初の実行可能ENTRYしか使いません（`placement_scan.rs:304-328`、`beam_reconstruct.rs:170-195`）。少なくとも「最早時刻・元時刻・`due-P`・禁止区間終了直後」を候補化し、さらにblockerの搬出までEXITを延長する候補を追加すべきです。現在は問題仕様で許される「処理完了後も残して並行処理する解」に到達できません。

- **blocker-aware LNSにする**  
  現在のremoveはtardinessや位置距離を見ますが、実際に早期搬入・搬出を妨げているblockを追っていません（`reconstruct.rs:422-677`）。scannerが最早時刻を押し下げた禁止区間の原因blockを記録し、`tardy block + blocker chain + 移動先候補`をまとめてremoveする方が直接的です。remove数も固定7〜13ではなく、問題サイズ・停滞時間・Tに応じて変えるべきです。

- **4 workerを異なる探索器として使う**  
  現在は全workerが同一初期解・同一温度で、`worker_temperature_scale: 0.0`、reheat未設定、Beamも現在の作業ツリーでは無効です。cold局所探索、hot LNS、precedence制約付き、完全非制約を並走させ、多様性付きarchiveを共有する方が有効です。同一設定の再実行で約2%揺れているため、単一shared-bestへの同期は探索多様性と再現性の両方を悪化させています。

- **Beamはそのまま再有効化しない**  
  現状は部分解の一時的なZ2で枝刈りし、candidate 16件が特定bay/orientationに偏りやすく、挿入順も1本です（`beam_reconstruct.rs:266-505`）。ベイ別quota、残りworkloadを考慮したZ2下界、複数挿入順、全枝で元配置候補を残す修正後に、低頻度で利用するのがよいです。

## 4. tardinessが大きいケースを短時間で改善する方針

最重要なのは、preoptimizeを強化することより、**その結果を幾何解へ壊さず移すこと**です。現在の`build_optimize_state()`はabstract stateから主にbay割当と疎なprecedenceだけを引き継ぎ、各ベイをランダムなtopological orderで最初から作り直しています（`ogc2026/src/solver/reconstruct.rs:94-230`）。

推奨する短時間向け経路は次です。

1. abstractの`entry_time`順を初期挿入順・目標時刻として直接利用する。
2. 幾何挿入時に「abstract時刻からの遅れ」を評価へ入れ、失敗箇所だけ局所repairする。
3. urgent・大体積・blockerになりやすいblockを先に配置し、容易なblockを後から詰める。
4. 初期フェーズではT削減を強く優先し、一定時間後に公式objectiveへ戻す。ログ上も300秒解はZ2/Z3を悪化させながらTを下げており、positiveケースではこの方向が合理的です。
5. `Move`は小面積blockを優先する現行選択（`neighbors.rs:332-350`）をやめ、tardiness限界寄与・blocker回数・workload限界寄与で対象を選ぶ。`prob_40`ではMove成功候補の改善率が約12〜18%あり、対象選択を直した上で比率を上げる価値があります。

長期的には、配置と時刻を直接一体化するより、空間配置から「AをBより先に搬出」「AをBの滞在内に入れる」といったイベント順序制約を作り、差分制約/DAGで最早時刻を再計算する状態表現が有望です。これにより延長EXIT、ABBA型の重なり、同日操作順を自然に探索できます。

## 5. 意図しない挙動・実装順

| 重要度 | 指摘 |
|---|---|
| **高** | preoptimizeの`occupancy`にperimeter等の混雑ペナルティを加えた値を、そのままハード容量制約に使っています（`preoptimize.rs:213-275`）。物理的には単独配置可能でも`occupancy > bay area`となり、初期状態を一件も作れない可能性があります。適格性判定と混雑評価を分離すべきです。 |
| **高** | 最初のcandidate emitはprecompute・preoptimize・幾何初期構築の後です（`solver.rs:48-121`）。`prob_40`では約12秒かかっています。重い内部ループは開始時にしかdeadlineを見ないため、短い制限では候補ゼロや超過の危険があります。absolute deadlineをscanner/reconstructまで渡す必要があります。 |
| **中** | `w3=0`かつzero-tardinessでgeometric温度が`0 * (0/0)^p = NaN`になり得ます（`annealing.rs:100-108`）。問題仕様上weightは0を許すため、ゼロ温度を明示処理すべきです。 |
| **低・潜在** | Left系bbox anchorを選んでもscannerは常に`range.min_x`しか生成しません（`insert.rs:42-59`、`placement_scan.rs:319-327`）。現在はanchor確率0なので未顕在ですが、有効化前に`max_x`候補が必要です。 |
| **仕様差** | 辺接触を問題仕様は許しますが、現在は`intersects`と`1e-6` marginで禁止しています（`collision.rs:743-758`）。過去ログに衝突・境界違反があるため、これは安易に緩めず、公式checkerによる反実仮想検証後に判断すべきです。 |

着手順は、**①計測追加と上記安全性修正 → ②bay index・scanner/base-deltaキャッシュ → ③preoptimizeからの時刻保持repairとblocker-aware LNS → ④複数ENTRY/延長EXIT → ⑤worker portfolio・Beam再設計**を推奨します。

コード変更やsolver実行は行っておらず、現行ソースの静的確認と既存ログの集計によるレビューです。最新`076`は`prob_40`のみfeasible確認済みで、現在のHEADに対する31ケース全体の実行可能性は未確認です。
