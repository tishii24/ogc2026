targeted-reconstruct:
- tardinessが大きいブロックを一つ選ぶ
- ブロックのentry-t=release-tに固定して、最も影響が少ない箇所への挿入を試す
  - 影響は干渉するブロックの area　の総和
- 干渉するブロックを全て削除する
- 削除されたブロックの再挿入を、reconstructと同様に順番を決めてから、insert-greedyで行う

reduce-tardiness:
- entry-tが近いブロックを削除する
- bayは区別しない
- 大幅に前に戻せる場合があるかもなので、randomに前の方も削除する

reduce-obj23:
- bay割当てを最適化する
- obj2を強く考慮しない
- bayごとの占有面積を一致させるように選ぶ必要がある
  - swap？
- 全体では良いが、block単位では損をする組合せを選ぶ必要がある
  - kick
  - 温度を高める
  - rough hash + tabu

### 温度

要求
- global-bestとの差が `min(w1,w3*50)` 以上になったら追従したい
-　`w3*20` くらいの悪化は許して高めの温度で探索したいが、`z1`が支配的な間は採用したくない

案
- tardinessの値に応じて温度を変える
  - z1>0: 0.1*w1 -> 0.01*w1
  - z1=0: 10*w3  -> w3
- exchange-threshold: `min(0.5*w1,w3*50)`
- 一度global-bestから取得していて、取得した以降でglobal-bestが更新されていなければ取得しない
