- targeted-reconstruct
- reconstructの改善
  - obj2を軽視する
- 高速化
- MILPの導入
- safety
- チューニング

safety:
- 定期的にpy側でfeasibilityをチェックする
  - check-feasibilityの高速化
- 空のblockがある場合の検証
- AIチェック
  - panicを排除する
  - fallback
- py側にいくつか解を返して、feasibilityをチェックする
- 縦長のbayではxとyを入れ替える
- 固定長配列をやめる
- orient-indexの確認

precompute:
- 衝突判定の高速化
- 2つのブロックの有望な隣接位置を計算する
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- 初期解改善
- reconstructの改善
  - obj2を軽視する
- bestを取ってくるときにkickの追加
- 小さい近傍の追加
- 大域的最適化
  - obj2,obj3の下界を求めて、それに合わせてbay-idを決める
  - bay-idの割り当てを事前に山登りする
  - obj2は最後でだけ気にする
- 高速化
  - insert-greedy
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
- スコア遷移を見る
