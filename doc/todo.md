- reconstructの改善
- 初期解の改善
- 並列annealing
- safetyを追加する
- パラメータ調整

safety:
- 定期的にpy側でfeasibilityをチェックする
  - check-feasibilityの高速化
- 空のblockのチェック
- 縦長のbayではxとyを入れ替える
- AIチェック
  - panicを排除する
  - fallback

precompute:
- 衝突判定の高速化
- 2つのブロックの有望な隣接位置を計算する
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- 局所探索の改善
  - reconstructの改善
  - shift-k
- 初期解の改善
  - 順番を変える
    - tardinessが大きいケースは小さい順に入れる
    - tardinessが発生することが確定しているブロックよりも、発生しないブロックを優先する
    - (loadsが大きい順、面積が大きい順、偏りが大きい順、締切が早い) の重みを探索する
  - 多点スタートする
  - ベストなパラメータを少しブラして実行する
- 大域的最適化
  - obj2,obj3の下界を求めて、それに合わせてbay-idを決める
- t-intervalをmergeして、候補tがなくなったら打ち切る
- 高速化
  - insert-greedyのチューニング
  - 移動してもスコアが良くならないbay-idは試さない
- 温度調整
  - reannealing
- 並列annealing
- 強い最適化
  - insertを評価して、ベストなinsertを探す
  - packing-scoreの計算（bboxの重なりなど）
    - 置ける面積を具体的に計算
  - 重なっている面積が少なくなる方に動かす
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
  - change-entry-tの追加

stats:
- ベストを一番下に表示

other:
- 難しいケースをaugmentationして評価
- 時間を延ばして評価
- 空のブロックがある場合を考慮
