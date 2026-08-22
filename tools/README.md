# Tools

各コマンドはリポジトリルートから実行する。

## `composer.py`

- 解法コードを提出可能な形式（zipにする前のフォルダ）にする
- `version: str` と `--params PATH` を引数として受け取り、`solutions/{version}` に保存する
- 指定したYAMLパラメータは `solutions/{version}/params.yaml` に同梱する

## `myalgorithm.py`

- 提出物から呼び出される解法のエントリーポイント
- 問題JSONを標準入力で `solver` に渡し、出力された解候補をスコア順に処理する
- 重複候補を除外して実行可能性を検証し、制限時間内に確認できた最良候補を返す

## `runner.py`

- 提出可能な形式にされたフォルダと、timelimitを指定して実行する
- `--params` 未指定時は `solutions/{version}/params.yaml` を使用する
- `(テストケース, timelimit)` に対する実行結果を `log/score.csv` に保存する
- 実行結果は `log/{version}/{timelimit}/{testcase}` に保存する

## `cloud_runner.py`

- Linux向けの提出物を作成し、suiteをGoogle Cloud Run Jobsで並列実行する
- `setup` で必要なGoogle Cloudリソースを作成・更新し、`update` で既存Jobの設定を更新する
- `run` で実行データをCloud Storageへ送り、完了後にスコアとログをローカルの `log/` へ統合する
- `logs` で指定した実行、または最新の実行のログを表示する
- 設定は `cloud/config.yaml` と、必要に応じて `cloud/config.local.yaml` から読み込む

## `tune_params.py`

- YAMLで定義したパラメータ候補をstepごとに評価する
- 各候補を `runner.py` または `--cloud` 指定時は `cloud_runner.py` で実行し、`stats.py` のrelative scoreで比較する
- 候補パラメータと結果を `tuning/{name}/` に保存する

## `stats.py`

- `log/score.csv` をもとに、`(version, timelimit)` ごとにスコアを集計する

## `stats_server.py`

- ブラウザから条件を指定し、reloadごとに `tools/stats.py` を再実行して表示する

## `visualizer.py`

- `runner.py` が作成した実行結果を可視化する

## `plot_horizon_score.py`

- `runner.py` が保存した `stderr.log` から、rolling horizonごとのworkerのcurrent scoreと温度を可視化する
- horizonの開始時刻、initial score、shared bestも同じグラフに表示する

## `anneal_visualizer.py`

- `anneal-visualizer` featureで出力したworker別JSONLスナップショットから、焼きなまし過程を確認する2D HTML viewerを生成する
- worker、近傍、スコア、変更・再構築対象ブロックなどを時系列で表示する

## `anneal_visualizer_3d.py`

- worker別JSONLスナップショットから、ブロック形状を立体表示する3D HTML viewerを生成する
- `--sampling-ratio` でHTMLへ格納するスナップショット数を間引ける
- 形状処理にShapelyを使用する

## 使い方

```bash
VERSION=v1
TIMELIMIT=60
PARAMS=params/default.yaml

# 提出ファイルの作成
python tools/composer.py $VERSION --params $PARAMS

# 単一ケースだけ実行
python tools/runner.py $VERSION --case train/prob_1.json --timelimit $TIMELIMIT

# suite の実行（別のパラメータを使う場合）
python tools/runner.py $VERSION --suite suites/half.json --timelimit $TIMELIMIT --params $PARAMS

# ビジュアライザの作成
python tools/visualizer.py log/$VERSION/$TIMELIMIT

# パラメータチューニング
# ローカル
python tools/tune_params.py params/tune-example.yaml
# Cloud Run Jobs
python tools/tune_params.py params/tune-example.yaml --cloud

# stderr.logのscoreと温度を可視化
python tools/plot_horizon_score.py log/$VERSION/$TIMELIMIT/prob_1/stderr.log
# 出力先を指定する場合
python tools/plot_horizon_score.py log/$VERSION/$TIMELIMIT/prob_1/stderr.log --out horizon-score.png
# versionごとにまとめた画像を出力する場合
python tools/plot_horizon_score.py log/$VERSION/$TIMELIMIT

# 統計情報の表示
python tools/stats.py --suite suites/half.json

# version, timelimit ごとのケース別スコア表示
python tools/stats.py --suite suites/half.json --matrix

# 統計情報をブラウザで表示
python tools/stats_server.py
# http://127.0.0.1:8000 を開く
```

## DockerでLinux向け提出物を作る

ジャッジ環境に近い Ubuntu 24.04 / x86_64 環境で `solver` をビルドする。
`myalgorithm.py` は `solver` に標準入力で問題JSONを渡すため、一時ファイルは作らない。

```bash
# Docker image のビルド（Dockerfile 更新後も再実行する）
docker build --platform linux/amd64 -t ogc2026-judge .

# 提出物の作成
VERSION=submit-v001
PARAMS=./params/default.yaml

docker run --rm \
  --platform linux/amd64 \
  -v /Users/tatsuyaishii/dev/heuristics-contests/ogc2026:/work/ogc2026 \
  -w /work/ogc2026 \
  ogc2026-judge \
  python3.12 tools/composer.py $VERSION --params $PARAMS --platform linux

(cd "solutions/$VERSION" && zip -r "../../$VERSION.zip" .)
zipinfo "$VERSION.zip"
```

## 焼きなまし過程の可視化

```bash
PROB=in/train-final/prob_40.json
cargo run --bin ogc2026 --release --features anneal-visualizer -- \
  --input $PROB \
  --params params/default.yaml \
  --timelimit 300 \
  --visualize tmp/anneal > /dev/null

python3 tools/anneal_visualizer.py \
  $PROB \
  tmp/anneal/optimize \
  --worker-id 0

python tools/anneal_visualizer_3d.py \
  $PROB \
  tmp/anneal/optimize \
  --worker-id 0
```
