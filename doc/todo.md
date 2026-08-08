todo:
- 進捗、問題サイズに応じて近傍サイズを大きくする係数の導入

nits:
- tabuを削除

tuning:
- preoptのパラメータ調整
- horizon-size, horizon-power
- reconstruct
  - remove-distance-powerを大きくするのを試す
  - reconstruct-weightにそれぞれpowerをつける
- SAの調整
  - exchange-threshold, exchange-interval
  - worker-scale

solver:
- 限界高速化

pending:
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- hashを荒くする
- 上から見た高さ（exit-t）を最小化する
- obj2: preserved-loadsを使う

rejected:
- adaptive-annealing
- multi-stage
  - tardiness -> pref -> pref + loads
- preoptimizeの改善
  - 近似方法・パラメータ
- kick
- beam-reconstructの導入
- 時刻の優先度だけでなく、bayの優先度を制約に入れて最適化する
- データ拡張
- caseに応じてtimelimitを設定する
- reconstructでanchorをreconstructごとに固定する
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
- pref/volume, limit-t/volumeの交互作用をorderに入れる
- dtにexit-tも考慮する
- w2を途中から考慮する
- 直方体[x_min,x_max][0,height][0,inf]をremoveする
