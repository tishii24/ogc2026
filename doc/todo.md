precompute:
- Cを設定 & lazy cache
- 怪しいものはpythonに返してshapelyで厳密に計算する
- 高速化

solver:
- 初期解をfeasibleにする
- 破壊再構築を小さくする
  - 大きいブロックから試す
  - time->bayの順で挿入を試したいです
    - (time,bay)の順で小さいものを選びたいです
- 大域的最適化
  - 緩和解を求めて、それに合わせて詰めに行きたい
- 小さい近傍
- 縦長のbayでは上から詰める
- 強い最適化
  - insertを評価して、ベストなinsertを探す
  - 置ける面積を具体的に計算
  - 操作順を考慮する
- 焼きなまし過程の可視化
- 並列焼きなまし

runner:
- dry-run

visualizer:
- max-prefでないblockもリストで表示
- w{1~3}を表示

stats:
- ケースごとの特徴をグループ化して平均を表示

other:
- 外側でコア数などを揃えて評価
- 難しいケースをaugmentationして評価
- 時間を延ばして評価
