precompute
- Cを設定 & lazy cache
- 高速化
- 怪しいものはpythonに返してshapelyで厳密に計算する

solver
- entry/exit条件を考える
  - entry/exit-time の順序関係を見ることで、ABAB か ABBA のいずれかを判定して、必要なcheckだけをすれば良いはず（AABBの場合は衝突しない）
- obj2,obj3の大域的最適化
- 削除するブロックを適切に選ぶ
  - 位置
- 縦長のbayでは上から詰める
- 強い最適化
  - insertを評価して、ベストなinsertを探す
  - 置ける面積を具体的に計算
- 並列焼きなまし

other
- 外側でコア数などを揃えて評価
- augmentationして評価
