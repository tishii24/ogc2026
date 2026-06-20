precompute
- Cを設定 & lazy cache
- 高速化
- 怪しいものはpythonに返してshapelyで厳密に計算する

solver
- なるべく斜めじゃない・好ましいorientationを計算しておく
- entry/exit条件を考える
  - 衝突判定はもっと正しくできそう
  - entry/exit-time の順序関係を見ることで、ABAB か ABBA のいずれかを判定して、必要なcheckだけをすれば良いはず（AABBの場合は衝突しない）
- obj2,obj3の大域的最適化
- 近傍
  - k個のブロックを削除・再挿入
    1. bay-idの最適な割り当てを求める
    2. leftから順に詰める
  - 小さい近傍
    - 少しずらす
  - ベストなinsertを探す
- 細長いものが多いなら、l,rから入れた方が良いかも
- magic numberはconstにする
- 並列焼きなまし

other
- 外側でコア数などを揃えて評価
- augmentationして評価
