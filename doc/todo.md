todo:
- 良い配置に向かう
- removeの時間距離をENTRY差から滞在区間gapへ変更
- 入りきらないなら
  - pref/volumeが大きい順に入れる
  - slack/volumeが大きい順に入れる
- target-bay/time-window removeを追加
- beam-reconstructの調整
- 1st(pref)-2nd(pref)をpref-spreadとする
- swapを入れる
- randomnessを高める
- 温度の調整
  - schedule: (cosine, linear)
  - constraint->globalで温度を滑らかにする
- obj2を軽視する
- seedの選び方を増やす
- 高速化
- 多点スタート？
- チューニング
  - 温度の調整
  - 時間に応じて近傍サイズを大きくする

precompute:
- 2つのブロックの有望な隣接位置を計算する
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- preoptimizeの改善
- 限界高速化

other:
- データ拡張
- gcloudで実行

pending:
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
- kick
- hashを荒くする
