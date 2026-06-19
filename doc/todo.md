precompute
- Cを設定 & lazy cache
- 高速化

solver
- u_kを前計算
- entry/exit条件を考える
  - entry/exitのtを決める
  - 干渉するブロックを決める
　　- ABBA、ABABのいずれでも、entry/exitができるかどうかは判定できるはず
  - 衝突判定はもっと正しくできそうです
  - entry/exit-time の順序関係を見ることで、ABAB か ABBA のいずれかを判定して、必要なcheckだけをすれば良いはずです（AABBの場合は衝突しない）
- obj2, obj3 を考慮した貪欲
  - どのbayに挿入すべきかだけを求める
- k個のブロックを削除・再挿入
  1. bay-idの最適な割り当てを求める
    - obj2, obj3の最適化
  2. tardinessが発生しないようにブロックを挿入する
    - k個のブロックを削除・ランダムな順序で挿入を繰り返せば良さそう
      - 4方向のいずれかから、全てのorientationについて
      - 低確率でランダムな挿入
    - 1で求めたbay-idに入れられなかったら制約をつけてbay-idを割り当て直す
- 並列焼きなまし

other
- 外側でコア数などを揃えて評価
