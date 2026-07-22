Rustから最終annealingのworker別best解をまとめて返し、`myalgorithm.py`側でRustスコア順に高速なfeasibility checkを行う方針がよいです。対話型にはせず、探索終了後にまとめて処理すれば実装も単純です。

1. **Rustで複数候補を保持**
   - `Annealer`の各workerで、スコアだけでなく`local_best`状態も保持します。
   - 終了時に以下をdelegateへ渡します。
     - shared best
     - 各workerのlocal best
   - Global annealingではスケジュールhashで重複を除き、スコア昇順に並べます。
   - Preoptimizeは従来どおりshared bestだけを使用します。

2. **solverの出力を複数候補に変更**
   ```json
   {
     "candidates": [
       {
         "score": 12345.0,
         "solution": {"operations": {}}
       }
     ]
   }
   ```
   - constrained annealingでは候補中のbestだけを次の初期解にします。
   - 最終global annealingの候補をPythonへ返します。
   - `worker_count: 4`なら、重複を除いて最大5候補程度になります。

3. **Python側でスコア順に確認**
   `tools/composer.py`が生成する`myalgorithm.py`で次を行います。
   ```python
   candidates.sort(key=lambda candidate: candidate["score"])
   for candidate in candidates:
       if fast_check_feasibility(prob_info, candidate["solution"]):
           return candidate["solution"]
   raise RuntimeError("no feasible solution")
   ```
   `tools/runner.py`やジャッジ側では、返された解に対する公式checkerが改めて実行されます。

4. **高速なfeasibility check**
   提出時の`utils.py`は上書きされるため、直接変更せず、`myalgorithm.py`内にsolver出力専用の軽量checkを実装します。
   - Rust出力では個数・ID・時刻形式が正常と仮定し、Stage 1の詳細な入力検証を省略。
   - Stage 5の時系列replayだけ実行。
   - `Bay`と配置済み`Block`は候補ごとに一度だけ生成して再利用。
   - `check_entry(..., fast=True)`と`check_exit(..., fast=True)`を使用。
   - objectiveは再計算せず、Rustの`score`を候補順にだけ使用。

5. **Stage 5だけでよい理由**
   正常形式のsolver出力なら、Stage 5が他の幾何判定を包含します。
   - Stage 2・3: 実際のENTRY/EXIT順で同じcrane判定を再実行するため不要。
   - Stage 4境界: 各ENTRYの`check_entry`で確認されるため不要。
   - Stage 4衝突: 時間的に重なる各ペアは、後から入るblockのENTRY時に確認されるため不要。

`tools/myalgorithm.py`にある未接続の対話型protocolは使わず、実際の提出物を生成する`tools/composer.py`側を変更するのが最小です。
