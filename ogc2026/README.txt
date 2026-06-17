OGC 2026 Optimization Challenge
================================

Contents
--------
  alg_tester/    Algorithm testing tool.
                 See alg_tester/README.txt for details.
  baseline/      Baseline algorithm template.
                 See baseline/README.txt for details.
  ogc2026_env.yml  Legacy environment definition.

Environment setup (once)
------------------------
  Install uv if you do not already have it:
    https://docs.astral.sh/uv/getting-started/installation/

  Create a virtual environment and install the packages needed by the
  Algorithm Tester from this directory:
    uv venv --python 3.12
    uv pip install PyQt6 "shapely>=2.1.0"

  If your myalgorithm.py uses additional packages, install them into the same
  environment, for example:
    uv pip install numpy scipy pandas ortools

Quick start
-----------
  Step 1  Set up the uv environment (see above).
  Step 2  Open baseline/ and edit myalgorithm.py.
  Step 3  Test your algorithm with the Algorithm Tester:
            cd alg_tester
            uv run python alg_tester_app.py
