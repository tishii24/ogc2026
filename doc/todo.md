課題:
- 大きいケースで良い解を作る
  - 良い初期解を作る
    - beam-search?
    - パラメータを増やす
  - 高速化
- 最適解が得やすいケースを小さいケースを落とさない
  - 幅広く探索する

todo:
- preblock-area
- reconstructの改善
  - targeted-reconstruct
  - obj23を軽視する
  - 高速化
- 順番の改善
  - volume
  - limit-t
- shiftを元の位置も含める（change-entry-t）
- 小さいケース
  - kick
  - 解の入れ替え条件を改善する
  - MILPの導入
- safety
- チューニング

safety:
- 定期的・最後にpy側にいくつかの解を返して、check-feasibilityをする
  - todo: check-feasibilityの高速化
- AIチェック
- panicを排除する
  - 空のblockがある場合（2点しかない場合）
  - fallback
  - 固定長配列をやめる

precompute:
- 衝突判定の高速化
- 2つのブロックの有望な隣接位置を計算する
  - 辺の角度を合わせる
  - 凸包を作って、面積が大きくならない組み合わせを求める

solver:
- 初期解改善
- 高速化
  - insert-greedy
- 評価の改善
  - 同じような時刻のブロックを近くに集めると、後のブロックを入れやすくなる
- 調整
  - start-temp,end-tempの推定
  - dx,dyをnon-positiveに限定する
  - パラメータ調整
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
