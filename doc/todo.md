todo:
- beam-reconstructの導入
- 時刻の優先度だけでなく、bayの優先度を制約に入れて最適化する
- reheatの再検証
- obj2を軽視する
- seedの選び方を増やす
  - target-bay/time-window remove

report:
- 可視化
- latexの準備

refactor:
- n/a

nits:
- reconstruct-weightのpowerをつける
- remove-blockの分布を変える
- exchange=1にする

tuning:
- 進捗に応じて近傍サイズを大きくする係数
- 温度の調整
- 制限時間の調整

高速化:
- precompute.crane
- scan-y

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
- removeの時間距離をENTRY差から滞在区間gapへ変更
- pref/volume, limit-t/volumeの交互作用をorderに入れる
