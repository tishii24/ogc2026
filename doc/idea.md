- LPで下界を求めて、その割り当てに近くなるように評価項を入れる
- LPで一部のbay-id割り当てだけ最適化

targeted-reconstruct:
- tardinessが大きいブロックを一つ選ぶ
- ブロックのentry-t=release-tに固定して、最も影響が少ない箇所への挿入を試す
  - 影響は干渉するブロックの area　の総和
- 干渉するブロックを全て削除する
- 削除されたブロックの再挿入を、reconstructと同様に順番を決めてから、insert-greedyで行う

preoptimize: MILPソルバーによって近似問題の解を求める
- (bay-id, entry-t)だけを変数とする
  - exit-tはentry-t + process-tとして良い
- 元の問題のスコアを最小化する
- ブロックの位置（x,y）は決めない
- 代わりに、各時刻tにおいて、各bayに存在するブロックの占有面積の総和がbayの面積を超えないことを制約とする
- ブロックiをbay jに配置した時の占有面積s_{i,j}は以下で定義する
  - s_{i,j} := a_i + \alpha * (b_i - a_i) + c_j
    - a_i := ブロックiの各layerのunionを取った図形の面積
    - b_i := ブロックiの各layerのunionを取った図形のbboxの面積
    - c_j := bay jに一つ配置することで増える占有面積（多いブロックほど配置しづらくなることを表す）
      - c_j := \beta * min(bay.width, bay.height)
- 占有面積のパラメータとして、(\alpha, \beta)がある
- MILPソルバーによって、(\alpha, \beta)の元での(bay-id, entry-t)が得られる
- note: \alpha = \beta = 0 とすると、充填率100%となり、厳密な下界が得られるはず？

最適化
1. (\alpha, \beta)を適当な値に設定して、preoptimizeを実行することで誘導解Sを得る
2. 実行可能な初期解をSに近づけるように貪欲法で作成する
  - bayの割り当てを固定して、ブロックの挿入順序を調整して挿入する
3. 局所探索
  - E := (1-\gamma) * D(s, S) + \gamma * E(s)
    - D(s, S) := 誘導解Sとの距離
    - E(s) := 実スコア
  - 実行可能性は崩さないまま探索をする
  -　E_sがpreoptimizeで得られた解のスコアを達成した場合
    - より良いスコアを持つ解が得られるまで(\alpha, \beta)を小さくしてpreoptimizeを実行して、Sを更新する
  - D(s,S) := \sum_{i} (s.entry_t[i] - S.entry_t[i])^2 + (if s.bay-id[i] != S.bay-id[i] then \lambda else 0)
    - とりあえず、\gamma=0,1の時だけ考え、0の時にはbayを跨がない移動とする
  - \gamma(p) \in [0,1]
    - 最初はスケジュールせず、一定時間過ぎたら0->1にする
