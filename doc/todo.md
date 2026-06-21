precompute:
- Cを設定 & lazy cache
- 怪しいものはpythonに返してshapelyで厳密に計算する
- 高速化

solver:
- obj2,obj3の大域的最適化
- 削除するブロックを適切に選ぶ
  - 位置
- 小さい近傍
- 縦長のbayでは上から詰める
- 強い最適化
  - insertを評価して、ベストなinsertを探す
  - 置ける面積を具体的に計算
  - 操作順を考慮する
- 焼きなまし過程の可視化
- 並列焼きなまし

visualizer:
- max-prefでないblockもリストで表示
- weightを表示

stats:
- 順位スコアを計算

other:
- 外側でコア数などを揃えて評価
- 難しいケースをaugmentationして評価
- 時間を延ばして評価
