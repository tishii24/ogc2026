# OGC 2026

## src, Cargo.toml
- 解法コード

## tools/composer.py
- 解法コードを提出可能な形式（zipにする前のフォルダ）にする
- `version: str`と`--params PATH`を引数として受け取って、`solutions/{version}`に保存する
- 指定したYAMLパラメータは`solutions/{version}/params.yaml`に同梱する

## tools/runner.py
- 提出可能な形式にされたフォルダと、timelimitを指定して実行する
- `--params`未指定時は`solutions/{version}/params.yaml`を使用する
- (テストケース, timelimit) に対する実行結果を`log/score.csv`に保存する
- 実行結果は`log/{version}/{timelimit}/{testcase}`に保存する

## tools/stats.py
- `log/score.csv`をもとに、（version, timelimit）ごとにスコアを集計する

## tools/stats_server.py
- ブラウザから条件を指定し、reloadごとに`tools/stats.py`を再実行して表示する

## tools/visualizer.py
- runner.pyが作成した実行結果を可視化する

## tools/plot_horizon_score.py
- runner.pyが保存した`stderr.log`から、rolling horizonごとのworkerのcurrent scoreと温度を可視化する
- horizonの開始時刻、initial score、shared bestも同じグラフに表示する
- matplotlibが必要

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

## Docker で Linux 向け提出物を作る

ジャッジ環境に近い Ubuntu 24.04 / x86_64 環境で `solver` をビルドする。
`myalgorithm.py` は `solver` に標準入力で問題 JSON を渡すため、一時ファイルは作らない。

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
PROB=in/preliminary_test/prob_5.json
cargo run --bin ogc2026 --release --features anneal-visualizer -- \
  --input $PROB \
  --params params/default.yaml \
  --timelimit 300 \
  --visualize tmp/anneal > /dev/null

python3 tools/anneal_visualizer.py \
  $PROB \
  tmp/anneal/global \
  --worker-id 0
```
