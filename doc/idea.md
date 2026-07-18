targeted-reconstruct:
- tardinessが大きいブロックを一つ選ぶ
- ブロックのentry-t=release-tに固定して、最も影響が少ない箇所への挿入を試す
  - 影響は干渉するブロックの area　の総和
- 干渉するブロックを全て削除する
- 削除されたブロックの再挿入を、reconstructと同様に順番を決めてから、insert-greedyで行う

解法:
P = (alpha, beta)
S = 状態
s = 抽象解（bay-id, entry-t）
1. Pを適当な値に設定して、preoptimizeを実行することで抽象解sを得る
  - 各時刻でベイごとの面積の総和を占有面積が超えないようにする
  - ベイごとの面積は固定paddingを持たせて計算する
  - 評価: 状態sの生スコア + 余裕
2. sをもとに順序制約を計算する
  - (i,j)について、end[i]+D<=start[j]なら順序を固定する
  - befores[i] := iより前におく必要があるブロック/
  - afters[i] := iより後におく必要があるブロック
  - D := 余裕を持たせるパラメータ
3. 前から順に詰めて貪欲解を作成する
4. annealing
  - 途中まで順序制約を持たせて探索する
  - tardiness=0の場合
    - TODO: 以下を定期的に繰り返す
      - k個を取り出して、obj2,obj3の最小化をするbay-idの組合せtarget-bayを求める
      - k個の取り出し方はいくつか試して、現在の状態からの差分とスコアの改善幅のバランスで良いものを選ぶ
      - target-bayを固定して、挿入先をそれに固定してしばらく探索する
  - tardiness>0の場合
    - global-annealing

- tardiness=0
  - bayを緩和ソルバーで求めた方が、最適解を得やすい
  - が、なるかならないかと一緒に対応したい
- tardiness=0になるかならないか・頑張ってもtardiness>0
  - bayを固定しない方がtardinessを小さくできる・0にできる場合がある
  - global-searchになってから初めてtardinessを0にできるため、事前にbayごとに最適化するメリットが薄そう
