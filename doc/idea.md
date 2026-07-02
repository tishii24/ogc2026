feasibilityのチェック:
- myalgorithm (python: 1 thread):
  - solverをsubprocessで起動する
  - stdinから解を読み取って、feasibilityを計算する
    - feasibilityの計算時間は0.2secくらい
  - feasibleじゃない解だった場合には、それをstdoutに書き出す
- solver (rust: 3 thread):
  - solver(worker-id=1~3)をmultithread（rayonなど）で動かす
  - annealingで解を探索して、一定周期で解をstdoutに書き出す
  - stdinからmyalgorithmが計算したfeasibilityの結果を読み取り、feasibleでない場合はrollbackする

初期解
- tardinessが大きいケースは小さい順に入れる
- tardinessが発生することが確定しているブロックよりも、発生しないブロックを優先する

is-tardy:
- remove-block
  - 違反量が大きいブロック
- insert-order
  - tardyな場合
    - obj2は無視する
    - anchorでの打ち切りをしない
  - tardyじゃない場合
    - 事前にbay-idの割り当てを最適化・それ通りにinsertを試す
- insert
  - 違反量が閾値を超える場合は打ち切る
