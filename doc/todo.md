todo:
- refactor
- target-bay
- 良い解ではブロック同士が重なっているか確認する
- precompute-pair-wise
- 温度の調整
- preoptimizeを途中で打ち切る
- pre.block-areaの修正
  - unionを使う
- reconstructの改善
  - targeted-reconstruct
  - obj2を軽視する
- preoptimizeの改善
- 高速化
- kick
- safety
- パラメータチューニング

safety:
- 定期的・最後にpy側にいくつかの解を返して、check-feasibilityをする
  - todo: check-feasibilityの高速化
- AIチェック
- panicを排除する
  - 空のblockがある場合（2点しかない場合）
  - fallback
  - 固定長配列をやめる

precompute:
- 高速化
- 2つのブロックの有望な隣接位置を計算する
  - 時刻が似ている者同士 & max-prefが一致しているブロック同士を合わせる
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- 同じような時刻のブロックを近くに集めると、後のブロックを入れやすくなる？
- 調整
  - dx,dyをnon-positiveに限定する
- 強い最適化
  - packing-scoreの計算（bboxの重なりなど）
  - 重なっている面積が少なくなる方に動かす

other:
- 全てのケースで評価
- データ拡張
- 時間を延ばして評価
- スコア遷移を見る

pending:
- 同時刻の操作順を考慮する
  - block-id順で出す、とすれば半分くらいは考慮できる
- 取り出す時刻を変えてABBA <->　ABABを入れ替える
  - change-entry-tの追加
- Pの外側ループ
