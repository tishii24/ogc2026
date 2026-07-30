todo:
- refactor
- beam-reconstructの導入
- 時刻の優先度だけでなく、bayの優先度を制約に入れて最適化する
- reheatの再検証
- obj2を軽視する
- seedの選び方を増やす
  - target-bay/time-window remove
- 高速化

nits:
- removeの時間距離をENTRY差から滞在区間gapへ変更
- 1st(pref)-2nd(pref)をpref-spreadとする
- pref/volume, limit-t/volumeの交互作用をorderに入れる
- 1:1 moveを入れる
- exchange=1にする

tuning:
- 温度の調整
- 制限時間の調整
- 進捗に応じて近傍サイズを大きくする係数をチューニングする

precompute:
- 取得の高速化

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
