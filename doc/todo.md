todo:
- report
- horizonの改善
- stderrからscoreの遷移を見る
- 焼きなまし過程を見る
- preopt
  - 近似方法・精度の改善
  - チューニング

nits:
- tl=30sで検証する
- loopを削除する

ideas:
- exit-tを最小化する
- 途中からw2を考慮する
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
