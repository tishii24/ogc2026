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

## ジャッジ環境の再現

```
VERSION=linux-check
python tools/composer.py $VERSION --platform linux
docker run --rm \
  --platform linux/amd64 \
  --network=none \
  --cpus=4 \
  --memory=16g \
  --memory-swap=16g \
  -v "$(pwd)":/work \
  -w /work \
  ogc2026-judge \
  python tools/runner.py $VERSION --cases train/*.json --timelimit 60
```

## 可視化

```bash
python tools/visualizer.py log/local-test/60/prob_1
```

## 提出

```bash
rustup target add x86_64-unknown-linux-musl

python tools/composer.py submit-v001 --platform linux

cd solutions/submit-v001
zip -r ../../submission-v001.zip .

zipinfo ../../submission-v001.zip
```
