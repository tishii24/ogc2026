todo:
- initial-build
  - rolling horizon
- 長時間の検証
- preopt
  - 近似方法・精度の改善
  - チューニング
- report

nits:
- epsを共通化する
- moveを全てのblockを対象にする
- remove-blockはL^aで削除する
- tardiness>0ならtを大きく、tardiness=0ならtは小さくする
- reconstruct-weightのpowerをつける
- reconstructはpreserved-loadsを使う
- スコアが離れすぎたら、bestをもらってくる

tuning:
- 温度の調整
- 制限時間の調整
- 進捗に応じて近傍サイズを大きくする係数の導入

solver:
- 限界高速化

evaluation:
- データ拡張
- caseに応じてtimelimitを設定する

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

rejected:
- adaptive-annealing
- multi-stage
  - tardiness -> pref -> pref + loads
- preoptimizeの改善
  - 近似方法・パラメータ
