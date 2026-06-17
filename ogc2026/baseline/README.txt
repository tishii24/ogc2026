OGC 2026 Baseline Algorithm
============================

Files
-----
  myalgorithm.py      -- YOUR algorithm goes here.  Fill in the algorithm()
                         function; do not change the function signature.
  baseline_greedy.py  -- Reference greedy implementation.  You may use it as
                         a starting point or call it from myalgorithm.py.
  utils.py            -- Feasibility checker and scoring utilities.
                         Do NOT modify this file.

Algorithm interface
-------------------
  myalgorithm.py must define:

      def algorithm(prob_info: dict, timelimit: float) -> dict:
          ...
          return solution

  prob_info  -- dict loaded from the problem instance JSON file
  timelimit  -- wall-clock seconds allowed for computation
  solution   -- dict matching the submission format defined in the problem
                statement

  You may import other modules or define helper functions freely inside
  myalgorithm.py, as long as the algorithm() signature is unchanged.

Requirements
------------
  Install uv if you do not already have it:
    https://docs.astral.sh/uv/getting-started/installation/

  From the repository root, create a virtual environment and install the
  packages needed by the Algorithm Tester (once):
    uv venv --python 3.12
    uv pip install PyQt6 "shapely>=2.1.0"

  If your myalgorithm.py uses additional packages, install them into the same
  environment, for example:
    uv pip install numpy scipy pandas ortools

Testing with Algorithm Tester
------------------------------
  1. Launch alg_tester from the alg_tester directory:
       cd ../alg_tester
       uv run python alg_tester_app.py

  2. Step 1: select a problem instance JSON file.
  3. Step 2: select THIS folder (the one containing myalgorithm.py).
  4. Step 3: set a time limit and click [Run].
     The Solution tab shows the feasibility check result and objective value.

Feasibility check (standalone)
--------------------------------
  from utils import check_feasibility
  result = check_feasibility(prob_info, solution)
  print(result)   # {"stage": "PASS", "objective": <value>} on success
