todo:
- refactor
- shiftはxを全て試す
- 流動性の高いブロックを選ぶ（pref-spreadが小さい）
- large-reconstructのremove-blockで、seedを選ぶ部分と、その後にseedの周りのblockを回収する部分を分けて、seedを選ぶ部分に以下を追加したいです
方針を検討してください
- 温度の調整
  - ベスト解をもらってくる確率をスコアの差に応じて確率的にする
- ランダム性を高める
  - timeのwindowを2つにする？
  - 最初のブロックほどランダム性を高める
  - ベイ、オリエンテーションをランダムにする
  - 確率的に探索範囲を絞って多様化
  - 元の位置から離れているほど高い評価
- 残りk個になったら、改善するベイの組合せのみ探索する
- preoptimizeの改善
  - 途中で打ち切る
- 高速化
- safety
- 時間ごとのチューニング

safety:
- 最後にpy側にいくつかの解を返して、スコアが良い順にcheck-feasibilityをする
  - feasibleなものが見つかったら返す
  - todo: check-feasibilityの無駄なところを落として高速化
- AIチェック
- panicを排除する
  - 空のblockがある場合（2点しかない場合）
  - fallback
  - 固定長配列をやめる
- tardiness>0に絶対になるケースがあるか調べる

precompute:
- 高速化
- 2つのブロックの有望な隣接位置を計算する
  - 時刻が似ている者同士 & max-prefが一致しているブロック同士を合わせる
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- 同じような時刻のブロックを近くに集めると、後のブロックを入れやすくなる？
- 限界高速化
- 調整
  - dx,dyをnon-positiveに限定する
- reconstructの改善
  - targeted-reconstruct
  - obj2を軽視する
- 強い最適化
  - packing-scoreの計算（bboxの重なりなど）
  - 重なっている面積が少なくなる方に動かす

other:
- 全てのケースで評価
- データ拡張
- 時間を延ばして評価
- スコア遷移を見る
- gcloudで実行

pending:
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
- kick
- Hashを荒くする
