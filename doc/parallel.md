# Parallel validation architecture

## 目的

Rust の局所探索を 3 worker で進めつつ、Python 側で非同期に feasibility validation を行う。CPU 使用は Rust 探索 3 core + Python validation 1 core を想定する。最終的に Python 側が validation 済みの best feasible solution を返す。

## プロセス構成

- `myalgorithm.py` が Rust solver を subprocess として起動する。
- Python は `asyncio` で solver の stdout/stdin/stderr を扱う。
- Rust solver は 3 worker で探索する。
- Rust のログは stderr に出す。
- Rust stdout/stdin は JSON Lines の通信専用にする。
- 最終解は Rust stdout ではなく、Python の `algorithm()` が返す。

## CPU 設定

Python が solver 起動時に環境変数を設定する。

```python
env["RAYON_NUM_THREADS"] = "3"
env["OMP_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["BLIS_NUM_THREADS"] = "1"
env["VECLIB_MAXIMUM_THREADS"] = "1"
env["NUMEXPR_NUM_THREADS"] = "1"
```

Rust 側も worker 数を 3 に制限する。

```rust
const SOLVER_WORKER_COUNT: usize = 3;
```

## 通信プロトコル

### Rust -> Python

Rust は validation したい候補解を stdout に 1 行 1 JSON で出力する。

```json
{"id":1,"worker_id":0,"score":123.0,"solution":{}}
```

- `id`: candidate id。全 worker で一意。
- `worker_id`: 候補を出した worker。
- `score`: Rust 側の目的関数値。
- `solution`: OGC の solution JSON。

### Python -> Rust

Python は validation 結果を solver stdin に 1 行 1 JSON で返す。

```json
{"id":1,"worker_id":0,"feasible":true,"objective":123.0}
```

infeasible の場合:

```json
{"id":1,"worker_id":0,"feasible":false}
```

- `id`, `worker_id` は Rust から受け取った値をそのまま返す。
- `objective` は feasible の場合のみ必要。

## Python 側

### 役割

- Rust solver を起動する。
- Rust stdout から candidate を読み続ける。
- candidate を 1 本の validation worker で検証する。
- validation 結果を Rust stdin に返す。
- feasible な candidate のうち最良の `solution` を保持する。
- 制限時間になったら solver を止め、best feasible solution を返す。

### asyncio task

- `read_candidates`: solver stdout を `readline()` し、candidate queue に入れる。
- `validate_loop`: candidate queue から取り出し、`check_feasibility` を実行する。
- `write_results`: validation 結果を solver stdin に書く。

### validation

`check_feasibility` は event loop 上で直接呼ばず、1 worker の executor に逃がす。

```python
executor = ThreadPoolExecutor(max_workers=1)
result = await loop.run_in_executor(executor, validate_candidate, prob_info, candidate)
```

### queue

candidate queue と result queue は小さい上限を持つ。

```python
candidate_queue = asyncio.Queue(maxsize=3)
result_queue = asyncio.Queue(maxsize=3)
```

### best solution

Python は feasible な candidate だけを best として保持する。

```python
if feasible and objective < best_objective:
    best_objective = objective
    best_solution = candidate["solution"]
```

`algorithm()` は `best_solution` を返す。

## Rust 側

### 役割

- 探索 worker を 3 本動かす。
- 各 worker は定期的に candidate を送る。
- 各 worker は validation 結果を非同期に受け取り、infeasible なら rollback する。
- Rust 側では最終解を stdout に出さない。

### channel 構成

- worker -> writer thread: `CandidateMsg`
- reader thread -> worker: `ValidationResult`

writer thread は `CandidateMsg` を stdout JSON Lines にする。
reader thread は stdin JSON Lines を読み、対応する `worker_id` の channel に流す。

### 型イメージ

```rust
struct CandidateMsg {
    id: u64,
    worker_id: usize,
    score: f64,
    solution: Solution,
}

struct ValidationResult {
    id: u64,
    worker_id: usize,
    feasible: bool,
    objective: Option<f64>,
}

struct PendingSnapshot {
    id: u64,
    worker_id: usize,
    schedule: Vec<ScheduledBlock>,
    score: f64,
}
```

candidate id は worker内 で一意にする。

```rust
let id = next_id + 1;
```

### worker の状態

worker は最低限以下を持つ。

```rust
let mut current: Vec<ScheduledBlock>;
let mut current_score: f64;
let mut last_valid: Option<(Vec<ScheduledBlock>, f64)>;
let mut pending: Option<PendingSnapshot>;
let mut restart_count: usize;
let mut last_validation_time: f64;
```

`last_valid.is_none()` は、まだ validation 済み feasible 解がない状態を表す。

### 初期解

- worker は初期解を生成したら、通常 candidate と同じ形式で Python に送る。
- Python は通常通り validation 結果を返す。
- Rust は初期解 validation の返答を待つ間も探索を開始してよい。

初期解 candidate 送信後の状態:

```text
last_valid == None
pending == Some(initial_snapshot)
```

初期解が feasible の場合:

```rust
last_valid = Some((pending.schedule, pending.score));
pending = None;
```

初期解が infeasible の場合:

```rust
restart_count += 1;
pending = None;
last_valid = None;
// current を破棄し、初期解生成からやり直す
```

初期解を再生成するときは `restart_count` を seed や挿入順に混ぜ、同じ失敗を繰り返さないようにする。

### candidate 送信

candidate は以下を満たすときだけ送る。

```rust
last_valid.is_some()
    && pending.is_none()
    && elapsed - last_validation_time >= VALIDATION_INTERVAL_SECONDS
    && current_score + EPS < last_valid_score
```

設定例:

```rust
const VALIDATION_INTERVAL_SECONDS: f64 = 1.0;
const EPS: f64 = 1e-9;
```

送信時は full snapshot を保存する。

```rust
let schedule = current.clone();
pending = Some(PendingSnapshot {
    id,
    schedule: schedule.clone(),
    score: current_score,
});
send(CandidateMsg {
    id,
    worker_id,
    score: current_score,
    solution: schedule_to_solution(&schedule),
});
last_validation_time = elapsed;
```

### validation 結果の反映

worker は各 iteration の先頭などで `try_recv()` により validation 結果を処理する。

結果の `id` が `pending.id` と一致しない場合は無視する。

feasible の場合:

```rust
last_valid = Some((pending.schedule, pending.score));
pending = None;
```

infeasible かつ `last_valid.is_some()` の場合:

```rust
current = last_valid_schedule.clone();
current_score = last_valid_score;
pending = None;
```

infeasible かつ `last_valid.is_none()` の場合は初期解 invalid とみなし、初期解生成からやり直す。

## 探索ループ概要

```text
初期解を作る
初期解 candidate を送る
loop until deadline:
  validation result を処理する
  last_valid == None && pending == None なら初期解を再生成して送る
  annealing を 1 step 進める
  条件を満たせば candidate を送る
```

## 制限時間

- Python 側が全体の制限時間を管理する。
- Rust 側も `deadline` を持って探索を止める。
- Python は終了直前に solver を terminate し、保持している best feasible solution を返す。
