precompute
- Cを設定 & lazy cache
- 高速化

solver
- entry/exit条件を考える
  - 衝突判定はもっと正しくできそう
  - entry/exit-time の順序関係を見ることで、ABAB か ABBA のいずれかを判定して、必要なcheckだけをすれば良いはずです（AABBの場合は衝突しない）
- 大域的最適化
  - obj2,obj3の最適化ができていない
  - 下界に近いスコアを求めたい
    1. scheduleを無視した下界
    2. schedule, areaを考慮した下界
- k個のブロックを削除・再挿入
  1. bay-idの最適な割り当てを求める
  2. leftから順に詰める
- 配置の厳密な最適化
- 細長いものが多いなら、l,rから入れた方が良いかも
- 並列焼きなまし

other
- 外側でコア数などを揃えて評価
