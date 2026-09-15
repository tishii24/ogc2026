# Cloud Run Jobsでの実行

Cloud Run Jobsを4 task並列で実行し、各taskでsolverの4 workerを使用する。
待機中のCloud Run compute料金は発生しない。

## 構成

- `Dockerfile`: Linux版solverのbuildとCloud Run task実行に使うbase image
- `config.yaml`: 共通設定
- `config.local.yaml`: `setup`が生成するproject・bucket設定（Git管理外）
- `task_runner.py`: suiteをtaskごとに分割して`tools/runner.py`を実行する
- `../cloud_runner.py`: setup、実行、ログ取得を行うローカルCLI

## 前提

- Google Cloudのbillingが有効であること
- `gcloud`とDockerがインストールされていること
- `gcloud auth login`が完了していること
- PyYAMLがローカルPython環境にインストールされていること

使用するprojectを先に設定する。

```bash
gcloud config set project PROJECT_ID
```

## 初回セットアップ

```bash
python tools/cloud_runner.py setup
```

以下を自動で行う。

- 必要なGoogle Cloud APIの有効化
- Artifact Registry repositoryの作成
- Cloud Storage bucketの作成
- Cloud Run Job用service accountの作成とbucket権限の設定
- base imageのbuildとArtifact Registryへのpush
- `tools/cloud/config.yaml`に基づくCloud Run Jobの作成
- DockerのArtifact Registry認証設定

project、bucket、regionは引数で指定できる。

```bash
python tools/cloud_runner.py setup \
  --project PROJECT_ID \
  --bucket BUCKET_NAME \
  --region asia-northeast1
```

`--bucket`未指定時は`PROJECT_ID-ogc2026-runs`を使用する。
確定した設定は`tools/cloud/config.local.yaml`に保存される。
`tools/cloud/Dockerfile`やイメージ自体を変更した場合は、もう一度`setup`を実行する。
`tools/cloud/config.yaml`のtask数、parallelism、CPU、メモリ、timeoutだけを変更した場合は、次のコマンドでDocker buildなしに反映できる。

```bash
python tools/cloud_runner.py update
```

## suiteの実行

```bash
VERSION=082
TIMELIMIT=120
PARAMS=params/default.yaml

python tools/cloud_runner.py run $VERSION \
  --params $PARAMS \
  --suite suites/full-hard.json \
  --timelimit $TIMELIMIT
```

処理の流れは以下の通り。

1. `tools/cloud/config.yaml`のJob設定を既存Jobへ同期する
2. base imageをローカルDockerで実行し、Linux版`solutions/{version}`を作成する
3. solution、checker、対象caseをbundleにしてCloud Storageへ送る
4. Cloud Run Jobsを設定されたtask数・parallelismで実行する
5. suiteのcaseをtaskへround-robinで割り当てる
6. task別のログをCloud Storageへ送る
7. ローカルへ取得し、既存の`log/score.csv`と`log/{version}/{timelimit}`へ統合する
8. 取得に成功したCloud Storage上の一時ファイルを削除する

実行に失敗した場合も、Cloud Storageへ保存できたログはローカルへ取得する。
一部taskが失敗した場合、コマンドはログ統合後に非ゼロで終了する。

## 結果の確認

通常のローカル実行と同じツールを使用できる。

```bash
python tools/stats.py --suite suites/full-hard.json --tl 120
python tools/visualizer.py log/081/120
```

Cloud Run側の標準出力はGoogle Cloud Consoleまたは次のコマンドでリアルタイムに確認できる。
実行名を省略すると、最新のExecutionを追尾する。

```bash
python tools/cloud_runner.py logs
```

複数のExecutionを並行実行している場合は、対象の実行名を指定する。

```bash
gcloud run jobs executions list \
  --job ogc2026-runner \
  --region asia-northeast1
python tools/cloud_runner.py logs EXECUTION_NAME
```

## 設定変更

task数、CPU、memory、timeoutは`tools/cloud/config.yaml`で変更し、`setup`を再実行する。

```yaml
tasks: 4
cpu: 4
memory: 4Gi
task_timeout: 24h
```

solverは`tools/runner.py`により最大4 threadに制限されるため、通常は`cpu: 4`のままでよい。
