課題:
長時間の最適化で、より良いスコアを出す
-> 試行ごとにばらつきがある
-> 全体的な構造が異なり、最も良い状態を目指せていない
-> 質の良い局所的なreconstructを増やせれば少しずつ改善しそうではある？
-> tardiness blockが多いので、敷き詰め効率が悪い
-> 敷き詰め効率を上げるには？
-> 色々試すしかない？

1位になるには？
- 全てのケースで、試行によらず、良い解に向かう必要がある
- どこら辺の配置が良いかはわからないので、幅広く探索する必要がある
- 局所的な改善を繰り返して到達できる場合
  - reconstruct,insertを改善することでOKになる
- 全体的な構造を改善する必要がある場合
  - 全体の解の構造を与える必要がある
    - どのblockを優先するか、どのbayに配置するか
  - ただ、これをヒューリスティックで事前に求めるのは困難そう

- reconstructの改善
- より広範囲を探索する
  - reheatの改善
- 目的状態を規定する
- 初期解を決める

todo:
- multi-stage optimizeの検証・追加
- 長時間の検証環境
- adaptive-annealing
  - refactor: trait
  - reheat、温度設定
- initial-build
  - beam-search
- preopt
  - 近似方法・精度の改善
  - チューニング
- report

nits:
- moveを全てのblockを対象にする
- reconstruct-weightのpowerをつける
- seedの選び方を増やす
  - target-bay/time-window remove
- reconstructはpreserved-loadsを使う
- reconstructでもmulti-stage objectiveで評価する

annealing:
- adaptive-annealing
  - reheat
  - temperature_per_block_scale を取り直す
  - スコアが離れすぎたら、bestをもらってくる
- multi-stage
  - tardiness -> pref -> pref + loads
  - obj2を軽視する

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

- adaptive-annealing
- multi-stage
  - tardiness -> pref -> pref + loads
- preoptimizeの改善
  - 近似方法・パラメータ
- initial-build、constraint-optimizeの改善
  - bay割り当てをもっと重視する
  - constraint-optimizeはbay割当をしばらく固定する
