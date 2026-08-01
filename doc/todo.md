todo:
- preopt
- refactor
- 高速化
- report

nits:
- global-cはtardiness=0になったら終了する
- moveを全てのblockを対象にする

tuning:
- caseに応じてtimelimitを設定する
- 進捗に応じて近傍サイズを大きくする係数
- 温度の調整
  - 再加熱
- 制限時間の調整

solver:
- 限界高速化

other:
- データ拡張

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
