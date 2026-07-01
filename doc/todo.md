safety:
- 定期的にpy側でfeasibilityをチェックする
  - check-feasibilityの高速化
- 空のblockがある場合の検証
- AIチェック
  - panicを排除する
  - fallback
- py側にいくつか解を返して、feasibilityをチェックする
- 縦長のbayではxとyを入れ替える

precompute:
- 衝突判定の高速化
- 2つのブロックの有望な隣接位置を計算する
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- 局所探索の改善
  - reconstructの改善
  - 小さい近傍の追加
  - 重複除去
- 初期解の改善
  - (loadsが大きい順、面積が大きい順、偏りが大きい順、締切が早い) の重みを探索する
  - 重みをランダムにサンプリングして探索する
  - 良いスコアとなったパラメータを保持しておき、たまにそこからサンプリングして、その重みをブラして再度探索する
  - 外側でorderを構築して、orderを与えるようにする
  - orderをhashしておいて重複除去
  - note:
    - tardinessが大きいケースは小さい順に入れる
    - tardinessが発生することが確定しているブロックよりも、発生しないブロックを優先する
- 大域的最適化
  - obj2,obj3の下界を求めて、それに合わせてbay-idを決める
- 高速化
  - insert-greedyのチューニング
- 調整
  - start-temp,end-tempの推定
  - dx,dyをnon-positiveに限定する
  - パラメータ
- 強い最適化
  - packing-scoreの計算（bboxの重なりなど）
  - 重なっている面積が少なくなる方に動かす
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
  - change-entry-tの追加

stats:
- ベストを一番下に表示

other:
- 全てのケースで評価
- ケースをaugmentationして評価
- 時間を延ばして評価
