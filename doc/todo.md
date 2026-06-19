precompute
- Cを削除
- lazy cache

solver
- entry/exit条件を考える
　　- 運び出す時刻を固定すれば、必要なentry/exitがわかるはず
- obj2, obj3 を考慮した貪欲
- k個のブロックを削除・ランダムな順序で挿入を繰り返せば良さそう
  - 4方向のいずれかから、全てのorientationについて
  - 低確率でランダムな挿入
- 並列焼きなまし

other
- コア数などを揃えて評価
