todo:
- horizon x beam-search
  - 複数状態を持って次のhorizonに行く
- base-scheduleを保持して複数のorderを差分計算で試す
- visualizerを見る
- horizon割り当ての改善
  - 中盤にもっと時間を割り当てて良さそう
  - 温度が高そう
  - horizonに含まれるblockのscore-13を用いる
- preopt
  - 近似方法・精度の改善
  - チューニング

nits:
- tl=30sで検証する
- candidate-emitの見直し

ideas:
- exit-tを最小化する
- moveを全てのblockを対象にする
- shift,swapを減らす
- remove-blockはL^aで削除する
- tardiness>0ならwtを大きく、tardiness=0ならwtは小さくする
- reconstruct-weightのpowerをつける
- reconstructはpreserved-loadsを使う
- reconstructでanchorをreconstructごとに固定する
- reconstructのdtにexit-tも使う
- スコアが離れすぎたら、bestをもらってくる
- w2を途中から考慮する

tuning:
- 温度の調整
- 制限時間の調整
- 進捗に応じて近傍サイズを大きくする係数の導入

solver:
- 限界高速化

pending:
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
- kick
- hashを荒くする
- removeの時間距離をENTRY差から滞在区間gapへ変更
- pref/volume, limit-t/volumeの交互作用をorderに入れる
- beam-reconstructの導入
- 時刻の優先度だけでなく、bayの優先度を制約に入れて最適化する
- データ拡張
- caseに応じてtimelimitを設定する

rejected:
- adaptive-annealing
- multi-stage
  - tardiness -> pref -> pref + loads
- preoptimizeの改善
  - 近似方法・パラメータ
