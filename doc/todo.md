todo:
- refactor
  - neighborの選択
- constraintをtardiness=0になったら取る
  - constraint-time-ratio=0を試す
- 温度の調整
  - schedule: (cosine, linear)
- seedの選び方を増やす
- 高速化
- safety
- チューニング
  - 温度の調整
  - 時間に応じて近傍サイズを大きくする

safety:
- 最後にpy側にいくつかの解を返して、スコアが良い順にcheck-feasibilityをする
  - feasibleなものが見つかったら返す
  - todo: check-feasibilityの無駄なところを落として高速化
- AIチェック
- panicを排除する
  - fallback

precompute:
- 高速化
- 2つのブロックの有望な隣接位置を計算する
  - 時刻が似ている者同士 & max-prefが一致しているブロック同士を合わせる
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- preoptimizeの改善
- 限界高速化
- 調整
  - dyをnon-positiveに限定する
- reconstructの改善
  - obj2を軽視する
- 強い最適化
  - packing-scoreの計算（bboxの重なりなど）
  - 重なっている面積が少なくなる方に動かす

other:
- データ拡張
- gcloudで実行

pending:
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
- kick
- hashを荒くする
