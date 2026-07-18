targeted-reconstruct:
- tardinessが大きいブロックを一つ選ぶ
- ブロックのentry-t=release-tに固定して、最も影響が少ない箇所への挿入を試す
  - 影響は干渉するブロックの area　の総和
- 干渉するブロックを全て削除する
- 削除されたブロックの再挿入を、reconstructと同様に順番を決めてから、insert-greedyで行う

解法:
S = 状態

1. 貪欲解を作成して、preoptimizeを実行するか決める
  - 一定時間を使用する
  - tardiness=0の解ができれば、preoptimizeを実行せずに、annealingの初期解として使用する
2. preoptimizeを実行する場合
  1. preoptimizeを実行することで抽象解s(entry-t)を得る
    - 各時刻で「ベイごとの面積の総和」を「その時刻に存在するブロックの占有面積の総和」が超えないようにする
    - ブロックをどのbayに置くかは区別しない
    - ベイごとの面積は固定paddingを持たせて計算する
      - ベイごとの面積の総和 = \sum_bay (bay.width-padding) * (bay.height-padding)
    - tardiness=0が達成できたら、min(cur-t+buffer-t,deadline)に終了する
    - 評価: 状態sのtardiness + 余裕
  2. sをもとに順序制約を計算する
    - (i,j)について、end[i]+D<=start[j]なら順序を固定する
    - befores[i] := iより前におく必要があるブロック
    - afters[i] := iより後におく必要があるブロック
    - D := 余裕を持たせるパラメータ
  3. 順序制約を守りながら貪欲解を作成する
3. annealing
  - tardiness=0の場合
    - TODO: 以下を定期的に繰り返す
      - k個を取り出して、obj2,obj3の最小化をするbay-idの組合せtarget-bayを求める
      - k個の取り出し方はいくつか試して、現在の状態からの差分とスコアの改善幅のバランスで良いものを選ぶ
      - target-bayを固定して、挿入先をそれに固定してしばらく探索する
  - tardiness>0の場合
    - 途中まで順序制約を持たせて探索する
    - global-annealing

- tardiness=0
  - bayを緩和ソルバーで求めた方が、最適解を得やすい
  - が、なるかならないかと一緒に対応したい
- tardiness=0になるかならないか・頑張ってもtardiness>0
  - bayを固定しない方がtardinessを小さくできる・0にできる場合がある
  - global-searchになってから初めてtardinessを0にできるため、事前にbayごとに最適化するメリットが薄そう
