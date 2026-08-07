todo:
- ブロックが多くなってもちゃんと入れ替えられるようにしたい
- removeの改善
  - 直方体に含まれるブロックを選ぶ
  - wyは小さくする
  - 無駄なblockをremoveしない
- reconstructのdtにexit-tも使う
- max-pref-bayをskipする

tuning:
- obj2
  - reconstructはpreserved-loadsを使う
  - w2を途中から考慮する
- preoptのパラメータ調整
- exchange-thresholdの調整
- neighbor-ratio
  - shift,swapを減らす
- tardiness>0ならwtを大きく、tardiness=0ならwtは小さくする
- reconstruct-weightのpowerをつける
- 温度の調整
  - worker-scale
- horizonの調整
  - horizonに含まれるblockのscore-13を用いる
- 進捗、問題サイズに応じて近傍サイズを大きくする係数の導入

solver:
- 限界高速化

pending:
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- hashを荒くする
- 上から見た高さ（exit-t）を最小化する

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
