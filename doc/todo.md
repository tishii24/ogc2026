todo:
- obj2を軽視する
- dyをnon-positiveに限定する
- swapを入れる
- reconstructの配置をgreedyにやらずに全探索する
- kick
- hashを荒くする
- reheat
- 温度の調整
  - schedule: (cosine, linear)
- seedの選び方を増やす
- 高速化
- safety
- チューニング
  - 温度の調整
  - 時間に応じて近傍サイズを大きくする

safety:
- 定期的にglobal-bestをstdoutに吐いておく
- py側
  - stdoutで受け取った解をscore順に並べて、check-feasibilityをして、feasibleだったら返す
    - todo: check-feasibilityの無駄なところを落として高速化
  - rustのpanicをcatchして、時間が余っていたらretryする
- panicを排除する

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
