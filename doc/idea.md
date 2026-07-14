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
- 占有面積のパラメータとして、Pがある
- MILPソルバーによって、Pの元での(bay-id, entry-t)が得られる
- note: \alpha = \beta = 0 とすると、充填率100%となり、厳密な下界が得られるはず？

課題:
- Pの設定
  - 無駄な誘導解Sに向かう時間を無くしたい
  - 実行不可能な(alpha,beta)に向かう時間も避けたい
- いつ制約を無くすべきか
- どのように誘導解に向かうか

解法:
P = (alpha, beta)
1. Pを適当な値に設定して、preoptimizeを実行することで誘導解Sを得る
  - 評価: (E(s'), D(s,s'))
    - E(s) := 状態sの生スコア
    - D(s,s') := 状態s,s'の距離
      - bayごとに異なる
  - 現在の状態から離れ過ぎないように正則化をかける
  - TODO: 詰めやすさをタイブレークのスコアとして導入する
2. 順序制約を計算する
  - is_before[i][j] := (i,j)について、end[i]<=start[j]なら順序を固定する
  - befores[i] := iより前におく必要があるブロック
3. bayごとに、前から順に詰めて貪欲解を作成する
  - 順序制約を守った範囲内で、ブロックの順番をランダムに選ぶ
4. bayごとに独立にannealing
  - 順序制約を守る
  - 目標tardinessに達成したら、そのbayでの探索は行わない
5. bay間の移動も許してannealing
  - TODO: より良いスコアを持つ解が得られるまでPを小さくして preoptimize を実行して、Sを更新する

note:
- 3.まではbayの大きさ・ブロックの数ごとにリソースを比例して与えられる

```
initialize P
current-state := None
while elapsed-time < deadline {
  abstract-state := preoptimize(\alpha, \beta, state)

  if state is None {
    state = initialize(abstract-state)
  }

  while state.score > abstract-state.score {
    optimize with abstract-state
  }
  
  P <- \eta * P
}
```
