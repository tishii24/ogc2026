todo:
- logをDEBUGにして、fallbackは失敗扱いにする
- reconstructの配置をgreedyにやらずに全探索する
- swapを入れる
- 温度の調整
  - schedule: (cosine, linear)
  - constraint->globalで温度を滑らかにする
- obj2を軽視する
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
    - todo: check-feasibilityの無駄なところを落として高速化・時間をみて打ち切る処理も入れる
  - rustのpanicを捕捉して、時間が余っていたらretryする
  - もし何かしら異常が起きていたら（feasibleじゃない解が帰っているなど）
    - 停止する処理をrunner.pyにつける（提出では落ちないようにする）
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
- kick
- hashを荒くする
- reheat
