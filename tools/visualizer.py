# ruff: noqa
#!/usr/bin/env python3
"""Create an interactive HTML viewer from runner output directories."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create viewer.html from log/{version}/{timelimit}/{testcase}.",
    )
    parser.add_argument("run_dir", help="Run result directory, e.g. log/v1/60/prob_1")
    parser.add_argument(
        "--out", help="Output HTML path. default: {run_dir}/viewer.html"
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_case_path(root: Path, case: str) -> Path:
    path = Path(case)
    if not path.is_absolute():
        path = root / path
    return path


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))
    ]


def build_assignments(solution: dict[str, Any]) -> list[dict[str, Any]]:
    assignments: dict[int, dict[str, Any]] = {}
    operations = solution.get("operations", {})

    for time_key, ops in operations.items():
        t = int(time_key)
        for op in ops:
            block_id = int(op["block_id"])
            if op["type"] == "ENTRY":
                assignments.setdefault(block_id, {"block_id": block_id})
                assignments[block_id].update(
                    {
                        "entry": t,
                        "bay_id": int(op["bay_id"]),
                        "x": int(op.get("x") or 0),
                        "y": int(op.get("y") or 0),
                        "orient_idx": int(op.get("orient_idx") or 0),
                    }
                )
            elif op["type"] == "EXIT":
                assignments.setdefault(block_id, {"block_id": block_id})
                assignments[block_id].update(
                    {
                        "exit": t,
                        "exit_bay_id": int(op["bay_id"]),
                    }
                )

    complete = [
        assignment
        for assignment in assignments.values()
        if "entry" in assignment and "exit" in assignment and "bay_id" in assignment
    ]
    complete.sort(key=lambda a: (a["bay_id"], a["entry"], a["exit"], a["block_id"]))
    return complete


def enrich_assignments(
    prob_info: dict[str, Any], assignments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    enriched = []
    for assignment in assignments:
        block_id = int(assignment["block_id"])
        block = prob_info["blocks"][block_id]
        bay_id = int(assignment["bay_id"])
        entry = int(assignment["entry"])
        exit_time = int(assignment["exit"])
        due = int(block.get("due_date", 0))
        release = int(block.get("release_time", 0))
        processing = int(block.get("processing_time", 0))
        prefs = [int(v) for v in block.get("bay_preferences", [])]
        assigned_pref = prefs[bay_id] if 0 <= bay_id < len(prefs) else 0
        max_pref = max(prefs, default=assigned_pref)
        best_bays = [bay for bay, pref in enumerate(prefs) if pref == max_pref]
        stay = exit_time - entry
        enriched.append(
            {
                **assignment,
                "release": release,
                "due": due,
                "processing": processing,
                "workload": int(block.get("workload", 0)),
                "tardiness": max(0, exit_time - due),
                "waiting": max(0, entry - release),
                "stay": stay,
                "extra_stay": max(0, stay - processing),
                "bay_preferences": prefs,
                "assigned_pref": assigned_pref,
                "max_pref": max_pref,
                "best_bays": best_bays,
                "preference_penalty": max_pref - assigned_pref,
            }
        )
    return enriched


def compute_tardiness_summary(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = [
        {
            "block_id": int(a["block_id"]),
            "tardiness": int(a.get("tardiness", 0)),
            "bay_id": int(a.get("bay_id", 0)),
            "entry": int(a.get("entry", 0)),
            "exit": int(a.get("exit", 0)),
            "due": int(a.get("due", 0)),
            "release": int(a.get("release", 0)),
            "processing": int(a.get("processing", 0)),
        }
        for a in assignments
        if int(a.get("tardiness", 0)) > 0
    ]
    blocks.sort(key=lambda a: (-a["tardiness"], a["exit"], a["block_id"]))
    return {
        "total": sum(a["tardiness"] for a in blocks),
        "count": len(blocks),
        "max": max((a["tardiness"] for a in blocks), default=0),
        "blocks": blocks,
    }


def compute_obj13_penalty_summary(
    weights: dict[str, float], assignments: list[dict[str, Any]]
) -> dict[str, Any]:
    w1 = weights["w1"]
    w3 = weights["w3"]
    blocks = []
    total_obj1 = 0.0
    total_obj3 = 0.0
    for a in assignments:
        tardiness = int(a.get("tardiness", 0))
        pref_penalty = int(a.get("preference_penalty", 0))
        obj1_penalty = w1 * tardiness
        obj3_penalty = w3 * pref_penalty
        total = obj1_penalty + obj3_penalty
        total_obj1 += obj1_penalty
        total_obj3 += obj3_penalty
        if total <= 0:
            continue
        blocks.append(
            {
                "block_id": int(a["block_id"]),
                "total": total,
                "obj1_penalty": obj1_penalty,
                "obj3_penalty": obj3_penalty,
                "tardiness": tardiness,
                "preference_penalty": pref_penalty,
                "bay_id": int(a.get("bay_id", 0)),
                "best_bays": a.get("best_bays", []),
                "entry": int(a.get("entry", 0)),
                "exit": int(a.get("exit", 0)),
                "due": int(a.get("due", 0)),
                "release": int(a.get("release", 0)),
                "processing": int(a.get("processing", 0)),
            }
        )

    blocks.sort(
        key=lambda a: (
            -a["total"],
            -a["obj1_penalty"],
            -a["obj3_penalty"],
            a["block_id"],
        )
    )
    return {
        "total": total_obj1 + total_obj3,
        "obj1": total_obj1,
        "obj3": total_obj3,
        "count": len(blocks),
        "blocks": blocks,
    }


def compute_obj2_detail(
    prob_info: dict[str, Any], assignments: list[dict[str, Any]]
) -> dict[str, Any]:
    bays = prob_info.get("bays", [])
    n_bays = len(bays)
    loads = [0.0] * n_bays
    for assignment in assignments:
        bay_id = int(assignment["bay_id"])
        if 0 <= bay_id < n_bays:
            loads[bay_id] += float(assignment.get("workload", 0))

    areas = [float(bay.get("width", 0)) * float(bay.get("height", 0)) for bay in bays]
    avg_area = sum(areas) / n_bays if n_bays else 0.0
    normalized = [
        loads[i] * avg_area / areas[i] if areas[i] > 0 else 0.0 for i in range(n_bays)
    ]
    if n_bays >= 2:
        min_value = min(normalized)
        max_value = max(normalized)
        value = math.floor(max_value - min_value)
        min_bay = normalized.index(min_value)
        max_bay = normalized.index(max_value)
    else:
        min_value = max_value = 0.0
        value = 0
        min_bay = max_bay = 0 if n_bays else None

    per_bay = []
    for bay_id in range(n_bays):
        area = areas[bay_id]
        u = avg_area / area if area > 0 else 0.0
        per_bay.append(
            {
                "bay_id": bay_id,
                "load": loads[bay_id],
                "area": area,
                "u": u,
                "normalized": normalized[bay_id],
                "is_min": bay_id == min_bay,
                "is_max": bay_id == max_bay,
            }
        )

    return {
        "value": value,
        "min": min_value,
        "max": max_value,
        "per_bay": per_bay,
    }


def is_case_run_dir(path: Path) -> bool:
    return (path / "meta.json").is_file() and (path / "solution.json").is_file()


def collect_run_dirs(path: Path) -> list[Path]:
    candidates = []
    if is_case_run_dir(path):
        search_dir = path.parent
        selected = path.resolve()
    else:
        search_dir = path
        selected = None

    if search_dir.is_dir():
        for child in search_dir.iterdir():
            if child.is_dir() and is_case_run_dir(child):
                candidates.append(child)

    if not candidates and is_case_run_dir(path):
        candidates = [path]

    candidates = sorted(
        {candidate.resolve() for candidate in candidates}, key=natural_key
    )
    if selected is not None and selected not in candidates:
        candidates.append(selected)
        candidates.sort(key=natural_key)
    return candidates


def load_view_case(root: Path, run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    solution_path = run_dir / "solution.json"
    result_path = run_dir / "result.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found")
    if not solution_path.is_file():
        raise FileNotFoundError(f"{solution_path} not found")

    meta = load_json(meta_path)
    solution = load_json(solution_path)
    result = load_json(result_path) if result_path.is_file() else None

    case = meta.get("case")
    if not isinstance(case, str) or not case:
        raise ValueError(f"{meta_path} does not contain a valid case")
    case_path = resolve_case_path(root, case)
    if not case_path.is_file():
        raise FileNotFoundError(f"case file not found: {case_path}")
    prob_info = load_json(case_path)

    weights_src = prob_info.get("weights", {})
    weights = {
        "w1": float(weights_src.get("w1", 1.0)),
        "w2": float(weights_src.get("w2", 1.0)),
        "w3": float(weights_src.get("w3", 1.0)),
    }
    assignments = enrich_assignments(prob_info, build_assignments(solution))
    tardiness = compute_tardiness_summary(assignments)
    obj13_penalty = compute_obj13_penalty_summary(weights, assignments)
    t_min = min((int(a["entry"]) for a in assignments), default=0)
    t_max = max((int(a["exit"]) for a in assignments), default=max(t_min, 1))
    name = str(meta.get("testcase") or prob_info.get("name") or run_dir.name)

    return {
        "name": name,
        "run_dir": str(run_dir.relative_to(root))
        if run_dir.is_relative_to(root)
        else str(run_dir),
        "meta": meta,
        "result": result,
        "bays": prob_info.get("bays", []),
        "blocks": prob_info.get("blocks", []),
        "weights": weights,
        "assignments": assignments,
        "tardiness": tardiness,
        "obj13_penalty": obj13_penalty,
        "obj2": compute_obj2_detail(prob_info, assignments),
        "t_min": t_min,
        "t_max": max(t_min, t_max),
    }


def render_html(viewer_data: dict[str, Any]) -> str:
    data_json = json.dumps(viewer_data, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__VIEWER_DATA__", data_json)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OGC 2026 Viewer</title>
<style>
  :root { color-scheme: light; }
  body {
    margin: 0;
    padding: 16px;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f8fafc;
    color: #111827;
  }
  .panel {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
  }
  .topbar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
  }
  label { font-size: 13px; color: #374151; }
  select, button, input[type="range"] { font: inherit; }
  button {
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    border-radius: 6px;
    padding: 4px 10px;
    cursor: pointer;
  }
  button:hover { background: #eef2ff; }
  #summary, #timeLabel, #blockInfo, #tardinessWarning { font-family: Menlo, Consolas, monospace; font-size: 13px; }
  #timeSlider { min-width: 320px; flex: 1; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #374151; }
  .chip { display: inline-flex; align-items: center; gap: 4px; }
  .swatch { width: 18px; height: 12px; border: 1px solid #64748b; border-radius: 2px; display: inline-block; }
  #bays { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
  .bay-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }
  .bay-title { font-weight: 600; font-size: 14px; margin-bottom: 6px; display: flex; justify-content: space-between; }
  canvas { width: 100%; display: block; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; }
  .obj2-row { display: grid; grid-template-columns: 72px 1fr 330px; gap: 8px; align-items: center; margin: 6px 0; font-size: 13px; }
  .bar-bg { background: #e5e7eb; height: 18px; border-radius: 4px; overflow: hidden; }
  .bar { height: 100%; background: #94a3b8; }
  .bar.max { background: #fb7185; }
  .bar.min { background: #60a5fa; }
  .muted { color: #6b7280; }
  .tardy-warning { margin-top: 10px; padding: 8px 10px; border-radius: 6px; border: 1px solid #fecaca; background: #fef2f2; color: #991b1b; font-weight: 600; }
  .tardy-ok { margin-top: 10px; padding: 8px 10px; border-radius: 6px; border: 1px solid #bbf7d0; background: #f0fdf4; color: #166534; }
  .penalty-list { max-height: 420px; overflow-y: auto; border-top: 1px solid #f1f5f9; }
  .tardy-row { display: grid; grid-template-columns: 54px 1fr auto; gap: 8px; align-items: center; padding: 6px 0; border-top: 1px solid #f1f5f9; font-size: 13px; }
  .tardy-row:first-child { border-top: 0; }
  .tardy-rank { font-family: Menlo, Consolas, monospace; font-weight: 600; color: #991b1b; }
  .jump-buttons { display: flex; gap: 4px; flex-wrap: wrap; justify-content: flex-end; }
  .jump-buttons button { padding: 2px 6px; font-size: 12px; }
</style>
</head>
<body>
  <div class="panel">
    <div class="topbar">
      <label>Case <select id="caseSelect"></select></label>
      <button id="playButton">Play</button>
      <label>Speed
        <select id="speedSelect">
          <option value="1">1x</option>
          <option value="2">2x</option>
          <option value="5" selected>5x</option>
          <option value="10">10x</option>
          <option value="20">20x</option>
        </select>
      </label>
      <span id="timeLabel"></span>
    </div>
    <div class="topbar" style="margin-top: 10px;">
      <input id="timeSlider" type="range" min="0" max="1" step="1" value="0">
    </div>
    <div id="summary" style="margin-top: 10px;"></div>
    <div id="tardinessWarning"></div>
    <div class="legend" style="margin-top: 10px;">
      <span class="chip"><span class="swatch" style="background:#d1fae5"></span>P=0</span>
      <span class="chip"><span class="swatch" style="background:#fef3c7"></span>P≤10</span>
      <span class="chip"><span class="swatch" style="background:#fcd34d"></span>P≤30</span>
      <span class="chip"><span class="swatch" style="background:#fb923c"></span>P≤60</span>
      <span class="chip"><span class="swatch" style="background:#f87171"></span>P&gt;60</span>
      <span class="chip"><span class="swatch" style="background:#fff;border:3px solid #dc2626"></span>tardiness &gt; 0</span>
    </div>
  </div>

  <div id="bays"></div>

  <div class="panel">
    <h3 style="margin: 0 0 8px 0; font-size: 16px;">Obj1 + Obj3 block penalty</h3>
    <div id="tardinessPanel"></div>
  </div>

  <div class="panel">
    <h3 style="margin: 0 0 8px 0; font-size: 16px;">Obj2: normalized workload imbalance</h3>
    <div id="obj2"></div>
  </div>

  <div class="panel">
    <h3 style="margin: 0 0 8px 0; font-size: 16px;">Block info</h3>
    <div id="blockInfo" class="muted">Hover a block.</div>
  </div>

<script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
<script>
'use strict';

const viewerData = JSON.parse(document.getElementById('viewer-data').textContent);
const caseSelect = document.getElementById('caseSelect');
const playButton = document.getElementById('playButton');
const speedSelect = document.getElementById('speedSelect');
const timeSlider = document.getElementById('timeSlider');
const timeLabel = document.getElementById('timeLabel');
const summary = document.getElementById('summary');
const tardinessWarning = document.getElementById('tardinessWarning');
const baysRoot = document.getElementById('bays');
const tardinessPanel = document.getElementById('tardinessPanel');
const obj2Root = document.getElementById('obj2');
const blockInfo = document.getElementById('blockInfo');

let currentCaseIndex = viewerData.initial_case_index || 0;
let currentTime = 0;
let currentTimeFloat = 0;
let playing = false;
let lastFrameTime = null;
let canvases = [];
const BASE_TIME_PER_SEC = 5;

function currentCase() {
  return viewerData.cases[currentCaseIndex];
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  const num = Number(value);
  return Number.isInteger(num) ? String(num) : num.toFixed(digits);
}

function penaltyColor(p) {
  if (p <= 0) return '#d1fae5';
  if (p <= 10) return '#fef3c7';
  if (p <= 30) return '#fcd34d';
  if (p <= 60) return '#fb923c';
  return '#f87171';
}

function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function setupCaseSelect() {
  caseSelect.innerHTML = '';
  viewerData.cases.forEach((caseData, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    const result = caseData.result || {};
    const tardiness = caseData.tardiness || { total: 0, count: 0, max: 0 };
    const obj = result.objective !== undefined && result.objective !== null ? ` obj=${result.objective}` : '';
    const tardy = Number(tardiness.total || 0) > 0
      ? `⚠ ${caseData.name} T=${tardiness.total} blocks=${tardiness.count} max=${tardiness.max}`
      : `✓ ${caseData.name} T=0`;
    option.textContent = `${tardy}${obj}`;
    caseSelect.appendChild(option);
  });
  caseSelect.value = String(currentCaseIndex);
}

function setCase(index) {
  currentCaseIndex = index;
  const data = currentCase();
  currentTime = data.t_min;
  currentTimeFloat = data.t_min;
  timeSlider.min = String(data.t_min);
  timeSlider.max = String(data.t_max);
  timeSlider.value = String(currentTime);
  renderBays();
  renderTardiness();
  renderObj2();
  updateSummary();
  drawAll();
}

function updateSummary() {
  const data = currentCase();
  const meta = data.meta || {};
  const result = data.result || {};
  const weights = data.weights || { w1: 1, w2: 1, w3: 1 };
  const w1 = Number(weights.w1 ?? 1);
  const w2 = Number(weights.w2 ?? 1);
  const w3 = Number(weights.w3 ?? 1);
  const tardiness = data.tardiness || { total: 0, count: 0, max: 0 };
  const obj13 = data.obj13_penalty || {};
  const obj1Weighted = result.obj1 !== undefined && result.obj1 !== null
    ? Number(result.obj1) * w1
    : Number(obj13.obj1 ?? 0);
  const obj2Weighted = result.obj2 !== undefined && result.obj2 !== null
    ? Number(result.obj2) * w2
    : Number(data.obj2.value || 0) * w2;
  const obj3Weighted = result.obj3 !== undefined && result.obj3 !== null
    ? Number(result.obj3) * w3
    : Number(obj13.obj3 ?? 0);
  const parts = [
    `version=${meta.version ?? '-'}`,
    `timelimit=${meta.timelimit ?? '-'}`,
    `case=${data.name}`,
    `feasible=${result.feasible ?? meta.feasible ?? '-'}`,
    `objective=${result.objective ?? meta.objective ?? '-'}`,
    `obj1=${formatNumber(obj1Weighted)}`,
    `obj2=${formatNumber(obj2Weighted)}`,
    `obj3=${formatNumber(obj3Weighted)}`,
    `weights=(w1=${formatNumber(w1)} w2=${formatNumber(w2)} w3=${formatNumber(w3)})`,
    `run=${data.run_dir}`,
  ];
  summary.textContent = parts.join('  ');
  if (Number(tardiness.total || 0) > 0) {
    tardinessWarning.className = 'tardy-warning';
    tardinessWarning.textContent = `⚠ TARDINESS total=${tardiness.total} blocks=${tardiness.count} max=${tardiness.max}`;
  } else {
    tardinessWarning.className = 'tardy-ok';
    tardinessWarning.textContent = '✓ TARDINESS none';
  }
  timeLabel.textContent = `t=${currentTime} / ${data.t_max}`;
}

function renderBays() {
  const data = currentCase();
  baysRoot.innerHTML = '';
  canvases = [];
  const maxBayWidth = Math.max(...data.bays.map(bay => Number(bay.width || 1)));
  const maxBayHeight = Math.max(...data.bays.map(bay => Number(bay.height || 1)));
  const commonAspect = maxBayHeight / Math.max(1, maxBayWidth);
  const commonCanvasHeight = Math.max(180, Math.min(420, Math.round(520 * commonAspect + 70)));
  data.bays.forEach((bay, bayId) => {
    const card = document.createElement('div');
    card.className = 'bay-card';

    const title = document.createElement('div');
    title.className = 'bay-title';
    title.innerHTML = `<span>Bay ${bayId}</span><span class="muted">${bay.width} × ${bay.height}</span>`;
    card.appendChild(title);

    const canvas = document.createElement('canvas');
    canvas.dataset.bayId = String(bayId);
    canvas.style.height = `${commonCanvasHeight}px`; 
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('mouseleave', () => {
      blockInfo.textContent = 'Hover a block.';
      blockInfo.classList.add('muted');
    });
    card.appendChild(canvas);
    baysRoot.appendChild(card);
    canvases.push(canvas);
  });
}

function jumpToTime(t) {
  const data = currentCase();
  const nextTime = Math.max(Number(data.t_min || 0), Math.min(Number(data.t_max || 0), Number(t)));
  currentTime = Math.floor(nextTime);
  currentTimeFloat = currentTime;
  timeSlider.value = String(currentTime);
  drawAll();
}

function renderTardiness() {
  const data = currentCase();
  const penalty = data.obj13_penalty || { total: 0, obj1: 0, obj3: 0, count: 0, blocks: [] };
  const blocks = penalty.blocks || [];
  tardinessPanel.innerHTML = '';

  const header = document.createElement('div');
  header.style.cssText = 'margin-bottom:8px;font-family:Menlo,Consolas,monospace;';
  header.textContent = `total=${formatNumber(penalty.total || 0)}  obj1=${formatNumber(penalty.obj1 || 0)}  obj3=${formatNumber(penalty.obj3 || 0)}  blocks=${penalty.count || 0}`;
  tardinessPanel.appendChild(header);

  if (!blocks.length) {
    const none = document.createElement('div');
    none.className = 'muted';
    none.textContent = 'none';
    tardinessPanel.appendChild(none);
    return;
  }

  const list = document.createElement('div');
  list.className = 'penalty-list';
  tardinessPanel.appendChild(list);

  blocks.forEach((block, index) => {
    const row = document.createElement('div');
    row.className = 'tardy-row';

    const rank = document.createElement('div');
    rank.className = 'tardy-rank';
    rank.textContent = `${index + 1}. B${block.block_id}`;
    row.appendChild(rank);

    const detail = document.createElement('div');
    const bestBays = (block.best_bays || []).map(bay => `Bay${bay}`).join('/');
    detail.textContent = `total=${formatNumber(block.total)}  obj1=${formatNumber(block.obj1_penalty)}(T=${block.tardiness})  obj3=${formatNumber(block.obj3_penalty)}(P=${block.preference_penalty})  bay=${block.bay_id}  best=${bestBays || '-'}  release=${block.release}  due=${block.due}  entry=${block.entry}  exit=${block.exit}  proc=${block.processing}`;
    row.appendChild(detail);

    const jumps = document.createElement('div');
    jumps.className = 'jump-buttons';
    [
      ['entry', block.entry],
      ['due', block.due],
      ['exit-1', Number(block.exit) - 1],
    ].forEach(([label, time]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.addEventListener('click', () => jumpToTime(time));
      jumps.appendChild(button);
    });
    row.appendChild(jumps);

    list.appendChild(row);
  });
}

function renderObj2() {
  const data = currentCase();
  const obj2 = data.obj2 || { value: 0, per_bay: [] };
  const maxNorm = Math.max(1, ...obj2.per_bay.map(row => Number(row.normalized || 0)));
  obj2Root.innerHTML = `<div style="margin-bottom:8px;font-family:Menlo,Consolas,monospace;">obj2=${obj2.value}  max-min=${formatNumber((obj2.max || 0) - (obj2.min || 0))}</div>`;
  obj2.per_bay.forEach(row => {
    const line = document.createElement('div');
    line.className = 'obj2-row';
    const width = Math.max(1, Number(row.normalized || 0) / maxNorm * 100);
    const cls = row.is_max ? 'bar max' : row.is_min ? 'bar min' : 'bar';
    const tag = row.is_max ? ' max' : row.is_min ? ' min' : '';
    line.innerHTML = `
      <div>Bay ${row.bay_id}</div>
      <div class="bar-bg"><div class="${cls}" style="width:${width.toFixed(2)}%"></div></div>
      <div class="muted">load=${formatNumber(row.load)} u=${formatNumber(row.u)} norm=${formatNumber(row.normalized)}${tag}</div>
    `;
    obj2Root.appendChild(line);
  });
}

function resizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function makeTransform(bay, bays, width, height) {
  const margin = 22;
  const maxBayWidth = Math.max(...bays.map(item => Number(item.width || 1)));
  const maxBayHeight = Math.max(...bays.map(item => Number(item.height || 1)));
  const scale = Math.min(
    (width - margin * 2) / maxBayWidth,
    (height - margin * 2) / maxBayHeight,
  );
  const originX = (width - Number(bay.width || 1) * scale) / 2;
  const originY = (height - Number(bay.height || 1) * scale) / 2;
  return {
    scale,
    x: worldX => originX + worldX * scale,
    y: worldY => originY + (Number(bay.height || 1) - worldY) * scale,
  };
}

function drawAll() {
  updateSummary();
  const data = currentCase();
  canvases.forEach(canvas => drawBay(canvas, data));
}

function drawBay(canvas, data) {
  const bayId = Number(canvas.dataset.bayId);
  const bay = data.bays[bayId];
  const { ctx, width, height } = resizeCanvas(canvas);
  const tr = makeTransform(bay, data.bays, width, height);
  canvas._hitboxes = [];

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  const x0 = tr.x(0);
  const y0 = tr.y(Number(bay.height || 0));
  const bw = Number(bay.width || 0) * tr.scale;
  const bh = Number(bay.height || 0) * tr.scale;
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, bw, bh);

  const active = data.assignments
    .filter(a => Number(a.bay_id) === bayId && Number(a.entry) <= currentTime && currentTime < Number(a.exit))
    .sort((a, b) => Number(a.block_id) - Number(b.block_id));

  for (const assignment of active) {
    drawBlock(ctx, tr, data, assignment, canvas._hitboxes);
  }

  ctx.fillStyle = '#64748b';
  ctx.font = '12px Menlo, Consolas, monospace';
  ctx.fillText(`active=${active.length}`, x0 + 6, y0 + 16);
}

function drawBlock(ctx, tr, data, assignment, hitboxes) {
  const block = data.blocks[assignment.block_id];
  if (!block) return;
  const orientations = block.shape || [];
  const orient = orientations[assignment.orient_idx] || orientations[0];
  if (!orient || !Array.isArray(orient.layers)) return;

  const fill = penaltyColor(Number(assignment.preference_penalty || 0));
  const stroke = Number(assignment.tardiness || 0) > 0 ? '#dc2626' : '#374151';
  const lineWidth = Number(assignment.tardiness || 0) > 0 ? 3 : 1;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  orient.layers.forEach((layer, layerIndex) => {
    if (!Array.isArray(layer) || layer.length < 3) return;
    ctx.beginPath();
    layer.forEach((point, index) => {
      const wx = Number(assignment.x || 0) + Number(point[0]);
      const wy = Number(assignment.y || 0) + Number(point[1]);
      minX = Math.min(minX, wx); maxX = Math.max(maxX, wx);
      minY = Math.min(minY, wy); maxY = Math.max(maxY, wy);
      const sx = tr.x(wx);
      const sy = tr.y(wy);
      if (index === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
    ctx.fillStyle = hexToRgba(fill, Math.max(0.35, 0.68 - layerIndex * 0.08));
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.fill();
    ctx.stroke();
  });

  if (!Number.isFinite(minX)) return;
  const sx0 = tr.x(minX), sx1 = tr.x(maxX);
  const sy0 = tr.y(maxY), sy1 = tr.y(minY);
  hitboxes.push({ x0: Math.min(sx0, sx1), y0: Math.min(sy0, sy1), x1: Math.max(sx0, sx1), y1: Math.max(sy0, sy1), assignment });

  const cx = tr.x((minX + maxX) / 2);
  const cy = tr.y((minY + maxY) / 2);
  const label = `B${assignment.block_id}`;
  const bestBayText = assignment.preference_penalty ? ` →${(assignment.best_bays || []).map(bay => 'Bay' + bay).join('/')}` : '';
  const sub = `${assignment.preference_penalty ? 'P+' + assignment.preference_penalty + bestBayText : ''}${assignment.tardiness ? ' T+' + assignment.tardiness : ''}`.trim();
  ctx.save();
  ctx.font = '11px Menlo, Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.fillStyle = '#111827';
  ctx.strokeText(label, cx, cy - (sub ? 6 : 0));
  ctx.fillText(label, cx, cy - (sub ? 6 : 0));
  if (sub) {
    ctx.font = '10px Menlo, Consolas, monospace';
    ctx.strokeText(sub, cx, cy + 7);
    ctx.fillText(sub, cx, cy + 7);
  }
  ctx.restore();
}

function onCanvasMouseMove(event) {
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hitboxes = canvas._hitboxes || [];
  let hit = null;
  for (let i = hitboxes.length - 1; i >= 0; i--) {
    const h = hitboxes[i];
    if (h.x0 <= x && x <= h.x1 && h.y0 <= y && y <= h.y1) {
      hit = h.assignment;
      break;
    }
  }
  if (!hit) {
    canvas.style.cursor = 'default';
    blockInfo.textContent = 'Hover a block.';
    blockInfo.classList.add('muted');
    return;
  }
  canvas.style.cursor = 'pointer';
  blockInfo.classList.remove('muted');
  const prefs = hit.bay_preferences || [];
  const prefText = prefs.map((s, bay) => `Bay${bay}:${s}`).join(' ');
  const bestText = (hit.best_bays || []).map(bay => `Bay${bay}`).join('/');
  blockInfo.textContent = [
    `B${hit.block_id}`,
    `bay=${hit.bay_id}`,
    `entry=${hit.entry}`,
    `exit=${hit.exit}`,
    `release=${hit.release}`,
    `due=${hit.due}`,
    `proc=${hit.processing}`,
    `workload=${hit.workload}`,
    `S=[${prefText}]`,
    `assignedS=${hit.assigned_pref}`,
    `maxS=${hit.max_pref}`,
    `obj3Penalty=${hit.preference_penalty}`,
    `obj3ZeroBay=${bestText || '-'}`,
    `T=${hit.tardiness}`,
    `orient=${hit.orient_idx}`,
    `x=${hit.x}`,
    `y=${hit.y}`,
  ].join('  ');
}

function animationFrame(timestamp) {
  if (!playing) return;
  if (lastFrameTime === null) lastFrameTime = timestamp;
  const dt = (timestamp - lastFrameTime) / 1000;
  lastFrameTime = timestamp;
  const data = currentCase();
  const speed = Number(speedSelect.value || 1);
  currentTimeFloat += dt * BASE_TIME_PER_SEC * speed;
  if (currentTimeFloat > data.t_max) currentTimeFloat = data.t_min;
  currentTime = Math.floor(currentTimeFloat);
  timeSlider.value = String(currentTime);
  drawAll();
  requestAnimationFrame(animationFrame);
}

caseSelect.addEventListener('change', () => setCase(Number(caseSelect.value)));
timeSlider.addEventListener('input', () => {
  currentTime = Number(timeSlider.value);
  currentTimeFloat = currentTime;
  drawAll();
});
playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  lastFrameTime = null;
  if (playing) requestAnimationFrame(animationFrame);
});
window.addEventListener('resize', drawAll);

setupCaseSelect();
setCase(currentCaseIndex);
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()

    run_dirs = collect_run_dirs(run_dir)
    if not run_dirs:
        print(
            f"error: no run directories found under {run_dir}; expected meta.json and solution.json",
            file=sys.stderr,
        )
        return 1

    cases = []
    initial_case_index = 0
    selected = run_dir if is_case_run_dir(run_dir) else run_dirs[0]
    for candidate in run_dirs:
        try:
            if candidate == selected:
                initial_case_index = len(cases)
            cases.append(load_view_case(root, candidate))
        except Exception as exc:
            if candidate == selected:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"warning: skipped {candidate}: {exc}", file=sys.stderr)

    if not cases:
        print("error: no viewable cases found", file=sys.stderr)
        return 1

    viewer_data = {
        "cases": cases,
        "initial_case_index": initial_case_index,
    }
    html = render_html(viewer_data)

    out_path = Path(args.out) if args.out else run_dir / "viewer.html"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
