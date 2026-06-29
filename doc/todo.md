- panicを排除する
- boundsを統一する
- annealingを改善する
- safetyを追加する
- 下界を求めて最適化

safety:
- feasibilityをチェックする
  - check-feasibilityの高速化
- panicを排除する
- 空のlayerに対する処理を追加する

precompute:
- 衝突判定の高速化
- ブロックごとにIoUが大きい(orientation,dx,dy)を前計算
  - bboxだけで良いかも
- 2つのブロック間でIoUが大きい(orientation,dx,dy)を前計算
  - (t,area)が似ているブロック間のみ
- 2つのブロックの有望な隣接位置を計算する
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- 初期解を軽くする
- 初期解のビームサーチ
- 大域的最適化
  - obj2,obj3の下界を求めて、それに合わせてbay-idを決める
- t-intervalをmergeして、候補tがなくなったら打ち切る
- 高速化
  - get-insert-tを差分評価する
- 近傍を小さくする
- 詰める順番を工夫する
  - 縦長のbayでは上から詰める
- 強い最適化
  - insertを評価して、ベストなinsertを探す
  - packing-scoreの計算（bboxの重なりなど）
    - 置ける面積を具体的に計算
  - 重なっている面積が少なくなる方に動かす
- 同時刻の操作順を考慮する
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
- 焼きなまし過程の可視化
- 並列annealing

stats:
- グループ化して平均を表示
- ベストを一番下に表示

other:
- 難しいケースをaugmentationして評価
- 時間を延ばして評価
- 空のブロックがある場合を考慮
