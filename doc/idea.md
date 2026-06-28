- 前計算でpolygonごとにぎっしり詰められる組み合わせを求めておく
  - 元のpolygonの凸包を作って、それが大きくならない組み合わせを求める
- tardiness>0: tardinessを最小化するパターン
- tardiness=0: でobj2,obj3を最小化する

構造:
- myalgorithm (python: 1 thread):
  - solverをsubprocessで起動する
  - stdinから解を読み取って、feasibilityを計算する
    - feasibilityの計算時間は0.2secくらい
  - feasibleじゃない解だった場合には、それをstdoutに書き出す
- solver (rust: 3 thread):
  - solver(worker-id=1~3)をmultithread（rayonなど）で動かす
  - annealingで解を探索して、一定周期で解をstdoutに書き出す
  - stdinからmyalgorithmが計算したfeasibilityの結果を読み取り、feasibleでない場合はrollbackする

- ファイル出力でも良いかも
