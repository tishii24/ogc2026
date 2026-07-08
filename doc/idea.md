- LPで下界を求めて、その割り当てに近くなるように評価項を入れる
- LPで一部のbay-id割り当てだけ最適化

targeted-reconstruct
- tardinessが大きいblockを一つ選ぶ
- blockのentry-t=release-tに固定して、最も影響が少ない箇所への挿入を試す
  - 影響は干渉するブロックの area　の総和
- 干渉するブロックを全て削除する
- 削除されたブロックの再挿入を、reconstructと同様に順番を決めてから、insert-greedyで行う
