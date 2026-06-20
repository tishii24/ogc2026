precompute
- Cを設定 & lazy cache
- 高速化

solver
- u_kを前計算
- entry/exit条件を考える
  - 衝突判定はもっと正しくできそう
  - entry/exit-time の順序関係を見ることで、ABAB か ABBA のいずれかを判定して、必要なcheckだけをすれば良いはずです（AABBの場合は衝突しない）
- obj2, obj3 を考慮した貪欲
- k個のブロックを削除・再挿入
  1. bay-idの最適な割り当てを求める
    - obj2, obj3の最適化
- 並列焼きなまし
- 細長いものが多いなら、l,rから入れた方が良いかも

- 配置が微妙
- obj2,obj3の最適化ができていない

other
- 外側でコア数などを揃えて評価
