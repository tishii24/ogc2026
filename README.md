# OGC 2026

## src, Cargo.toml
- 解法コード

## tools/composer.py
- 解法コードを提出可能な形式（zipにする前のフォルダ）にする
- `version: str`を引数として受け取って、`solutions/{version}`に保存する

## tools/runner.py
- 提出可能な形式にされたフォルダと、timelimitを指定して実行する
- (テストケース, timelimit) に対する実行結果を`log/score.csv`に保存する
- 実行結果は`log/{version}/{timelimit}/{testcase}`に保存する

## tools/stats.py
- `log/score.csv`をもとに、（version, timelimit）ごとにスコアを集計する

## tools/visualizer.py
- runner.pyが作成した実行結果を可視化する

## テスト

```bash
VERSION=v1
python tools/composer.py $VERSION

python tools/runner.py $VERSION --case train/prob_1.json --timelimit 60
python tools/runner.py $VERSION --cases train/*.json --timelimit 60

python tools/stats.py
```

## 可視化

```bash
python tools/visualizer.py log/local-test/60/prob_1
```

## Docker で Linux 向け提出物を作る

ジャッジ環境に近い Ubuntu 24.04 / x86_64 環境で `solver` をビルドする。

```bash
docker build --platform linux/amd64 -t ogc2026-judge .

VERSION=submit-v001

docker run --rm \
  --platform linux/amd64 \
  -v /Users/tatsuyaishii/dev/heuristics-contests/ogc2026:/work/ogc2026 \
  -w /work/ogc2026 \
  ogc2026-judge \
  python3.12 tools/composer.py $VERSION --platform linux
```
