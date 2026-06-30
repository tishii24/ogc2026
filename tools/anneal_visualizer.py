#!/usr/bin/env python3
"""Create an annealing-process HTML viewer from worker JSONL snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create annealing viewer HTML.")
    parser.add_argument("problem", help="Problem JSON path")
    parser.add_argument("visualize_dir", help="Directory containing worker_{id}.jsonl")
    parser.add_argument(
        "--worker-id", type=int, required=True, help="Worker id to visualize"
    )
    parser.add_argument(
        "--out", help="Output HTML path. default: {visualize_dir}/worker_{id}.html"
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    snapshots = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if item.get("type") == "snapshot":
                snapshots.append(item)
    snapshots.sort(key=lambda s: (float(s.get("elapsed", 0.0)), int(s.get("iter", 0))))
    return snapshots


def render_html(viewer_data: dict[str, Any]) -> str:
    data_json = json.dumps(viewer_data, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__VIEWER_DATA__", data_json)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OGC 2026 Annealing Viewer</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; padding: 16px; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }
  .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  .topbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  label { font-size: 13px; color: #374151; }
  select, button, input[type="range"] { font: inherit; }
  button { border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
  button:hover { background: #eef2ff; }
  #summary, #snapLabel, #blockInfo { font-family: Menlo, Consolas, monospace; font-size: 13px; }
  #snapSlider { min-width: 320px; flex: 1; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #374151; }
  .chip { display: inline-flex; align-items: center; gap: 4px; }
  .swatch { width: 18px; height: 12px; border: 1px solid #64748b; border-radius: 2px; display: inline-block; }
  #bays { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
  .bay-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }
  .bay-title { font-weight: 600; font-size: 14px; margin-bottom: 6px; display: flex; justify-content: space-between; }
  canvas { width: 100%; display: block; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; }
  .muted { color: #6b7280; }
</style>
</head>
<body>
  <div class="panel">
    <div class="topbar">
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
      <span id="snapLabel"></span>
    </div>
    <div class="topbar" style="margin-top:10px;">
      <input id="snapSlider" type="range" min="0" max="0" step="1" value="0">
    </div>
    <div id="summary" style="margin-top:10px;"></div>
    <div class="legend" style="margin-top:10px;">
      <span class="chip"><span class="swatch" style="background:#d1fae5"></span>T=0</span>
      <span class="chip"><span class="swatch" style="background:#fef3c7"></span>T≤10</span>
      <span class="chip"><span class="swatch" style="background:#fcd34d"></span>T≤30</span>
      <span class="chip"><span class="swatch" style="background:#fb923c"></span>T≤60</span>
      <span class="chip"><span class="swatch" style="background:#f87171"></span>T&gt;60</span>
    </div>
  </div>

  <div id="bays"></div>

  <div class="panel">
    <h3 style="margin:0 0 8px 0;font-size:16px;">Block info</h3>
    <div id="blockInfo" class="muted">Hover a block.</div>
  </div>

<script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
<script>
'use strict';

const viewerData = JSON.parse(document.getElementById('viewer-data').textContent);
const problem = viewerData.problem;
const snapshots = viewerData.snapshots || [];
const playButton = document.getElementById('playButton');
const speedSelect = document.getElementById('speedSelect');
const snapSlider = document.getElementById('snapSlider');
const snapLabel = document.getElementById('snapLabel');
const summary = document.getElementById('summary');
const baysRoot = document.getElementById('bays');
const blockInfo = document.getElementById('blockInfo');
let currentIndex = 0;
let currentIndexFloat = 0;
let playing = false;
let lastFrameTime = null;
let canvases = [];
const BASE_SNAPSHOTS_PER_SEC = 5;

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  const num = Number(value);
  return Number.isInteger(num) ? String(num) : num.toFixed(digits);
}

function tardinessColor(t) {
  if (t <= 0) return '#d1fae5';
  if (t <= 10) return '#fef3c7';
  if (t <= 30) return '#fcd34d';
  if (t <= 60) return '#fb923c';
  return '#f87171';
}

function currentSnapshot() { return snapshots[currentIndex] || { schedule: [] }; }

function setup() {
  snapSlider.max = String(Math.max(0, snapshots.length - 1));
  renderBays();
  updateAll();
}

function renderBays() {
  baysRoot.innerHTML = '';
  canvases = [];
  (problem.bays || []).forEach((bay, bayId) => {
    const card = document.createElement('div');
    card.className = 'bay-card';
    const title = document.createElement('div');
    title.className = 'bay-title';
    title.innerHTML = `<span>Bay ${bayId}</span><span class="muted">${bay.width} × ${bay.height}</span>`;
    card.appendChild(title);
    const canvas = document.createElement('canvas');
    canvas.dataset.bayId = String(bayId);
    const aspect = Number(bay.height || 1) / Math.max(1, Number(bay.width || 1));
    canvas.style.height = `${Math.max(180, Math.min(420, Math.round(520 * aspect + 70)))}px`;
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

function transformPoint(x, y, bay, width, height) {
  const pad = 18;
  const sx = (width - 2 * pad) / Math.max(1, Number(bay.width || 1));
  const sy = (height - 2 * pad) / Math.max(1, Number(bay.height || 1));
  const scale = Math.min(sx, sy);
  const ox = pad + (width - 2 * pad - Number(bay.width || 1) * scale) / 2;
  const oy = pad + (height - 2 * pad - Number(bay.height || 1) * scale) / 2;
  return [ox + x * scale, oy + (Number(bay.height || 1) - y) * scale];
}

function blockTardiness(s) {
  const block = problem.blocks[s.block_id] || {};
  return Math.max(0, Number(s.exit_time || 0) - Number(block.due_date || 0));
}

function drawAll() { canvases.forEach(drawBay); updateSummary(); }

function drawBay(canvas) {
  const bayId = Number(canvas.dataset.bayId);
  const bay = problem.bays[bayId];
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  const [x0, y0] = transformPoint(0, 0, bay, width, height);
  const [x1, y1] = transformPoint(Number(bay.width || 0), Number(bay.height || 0), bay, width, height);
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);

  const snap = currentSnapshot();
  const blocks = (snap.schedule || []).filter(s => Number(s.bay_id) === bayId);
  blocks.sort((a, b) => Number(a.block_id) - Number(b.block_id));
  for (const s of blocks) drawBlock(ctx, width, height, bay, s);
}

function drawBlock(ctx, width, height, bay, s) {
  const block = problem.blocks[s.block_id];
  if (!block) return;
  const orient = (block.shape || [])[s.orient_idx];
  if (!orient) return;
  const layers = orient.layers || [];
  const color = tardinessColor(blockTardiness(s));
  layers.forEach((poly, layerIndex) => {
    if (!poly.length) return;
    ctx.beginPath();
    poly.forEach((pt, index) => {
      const [px, py] = transformPoint(Number(s.x) + Number(pt[0]), Number(s.y) + Number(pt[1]), bay, width, height);
      if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = Math.max(0.18, 0.72 - layerIndex * 0.08);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = blockTardiness(s) > 0 ? '#dc2626' : '#334155';
    ctx.lineWidth = blockTardiness(s) > 0 ? 2 : 1;
    ctx.stroke();
  });

  const first = layers.find(poly => poly.length);
  if (first) {
    const xs = first.map(pt => Number(s.x) + Number(pt[0]));
    const ys = first.map(pt => Number(s.y) + Number(pt[1]));
    const [cx, cy] = transformPoint((Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2, bay, width, height);
    ctx.fillStyle = '#111827';
    ctx.font = '11px Menlo, Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(s.block_id), cx, cy);
  }
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    const intersect = ((yi > y) !== (yj > y)) && x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function canvasToBay(canvas, event) {
  const bayId = Number(canvas.dataset.bayId);
  const bay = problem.bays[bayId];
  const rect = canvas.getBoundingClientRect();
  const pad = 18;
  const sx = (rect.width - 2 * pad) / Math.max(1, Number(bay.width || 1));
  const sy = (rect.height - 2 * pad) / Math.max(1, Number(bay.height || 1));
  const scale = Math.min(sx, sy);
  const ox = pad + (rect.width - 2 * pad - Number(bay.width || 1) * scale) / 2;
  const oy = pad + (rect.height - 2 * pad - Number(bay.height || 1) * scale) / 2;
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  return [(px - ox) / scale, Number(bay.height || 1) - (py - oy) / scale];
}

function onCanvasMouseMove(event) {
  const canvas = event.currentTarget;
  const bayId = Number(canvas.dataset.bayId);
  const [x, y] = canvasToBay(canvas, event);
  const blocks = (currentSnapshot().schedule || []).filter(s => Number(s.bay_id) === bayId);
  for (let k = blocks.length - 1; k >= 0; k--) {
    const s = blocks[k];
    const block = problem.blocks[s.block_id];
    const orient = block && (block.shape || [])[s.orient_idx];
    if (!orient) continue;
    for (const poly of orient.layers || []) {
      const shifted = poly.map(pt => [Number(s.x) + Number(pt[0]), Number(s.y) + Number(pt[1])]);
      if (pointInPolygon(x, y, shifted)) {
        const tardy = blockTardiness(s);
        const prefs = block.bay_preferences || [];
        const pref = Number(prefs[s.bay_id] || 0);
        const maxPref = Math.max(pref, ...prefs.map(Number));
        blockInfo.classList.remove('muted');
        blockInfo.textContent = `B${s.block_id} bay=${s.bay_id} orient=${s.orient_idx} x=${s.x} y=${s.y} entry=${s.entry_time} exit=${s.exit_time} due=${block.due_date} tardiness=${tardy} pref_penalty=${maxPref - pref}`;
        return;
      }
    }
  }
  blockInfo.textContent = 'Hover a block.';
  blockInfo.classList.add('muted');
}

function updateSummary() {
  const snap = currentSnapshot();
  snapLabel.textContent = `snapshot ${currentIndex + 1} / ${snapshots.length}`;
  snapSlider.value = String(currentIndex);
  summary.textContent = [
    `worker=${snap.worker ?? viewerData.worker_id}`,
    `iter=${snap.iter ?? '-'}`,
    `elapsed=${formatNumber(snap.elapsed)}s`,
    `reason=${snap.reason ?? '-'}`,
    `neighbor=${snap.neighbor ?? '-'}`,
    `accepted=${snap.accepted ?? '-'}`,
    `improved_current=${snap.improved_current ?? '-'}`,
    `improved_best=${snap.improved_best ?? '-'}`,
    `score=${formatNumber(snap.score)}`,
    `current=${formatNumber(snap.current_score)}`,
    `best=${formatNumber(snap.best_score)}`,
    `delta=${formatNumber(snap.delta)}`,
  ].join('  ');
}

function updateAll() { drawAll(); }

function setSnapshot(index) {
  currentIndex = Math.max(0, Math.min(snapshots.length - 1, Math.floor(index)));
  currentIndexFloat = currentIndex;
  updateAll();
}

function animate(timestamp) {
  if (!playing) return;
  if (lastFrameTime === null) lastFrameTime = timestamp;
  const dt = Math.max(0, (timestamp - lastFrameTime) / 1000);
  lastFrameTime = timestamp;
  currentIndexFloat += dt * BASE_SNAPSHOTS_PER_SEC * Number(speedSelect.value || 1);
  if (currentIndexFloat >= snapshots.length - 1) {
    currentIndexFloat = snapshots.length - 1;
    playing = false;
    playButton.textContent = 'Play';
  }
  setSnapshot(currentIndexFloat);
  if (playing) requestAnimationFrame(animate);
}

playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  lastFrameTime = null;
  if (playing) requestAnimationFrame(animate);
});
snapSlider.addEventListener('input', () => setSnapshot(Number(snapSlider.value)));
window.addEventListener('resize', drawAll);

setup();
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    root = repo_root()
    problem_path = resolve_path(root, args.problem)
    visualize_dir = resolve_path(root, args.visualize_dir)
    worker_path = visualize_dir / f"worker_{args.worker_id}.jsonl"
    if not problem_path.is_file():
        print(f"error: problem not found: {problem_path}", file=sys.stderr)
        return 1
    if not worker_path.is_file():
        print(f"error: worker snapshot not found: {worker_path}", file=sys.stderr)
        return 1

    try:
        problem = load_json(problem_path)
        snapshots = load_snapshots(worker_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not snapshots:
        print(f"error: no snapshots in {worker_path}", file=sys.stderr)
        return 1

    viewer_data = {
        "problem": problem,
        "worker_id": args.worker_id,
        "snapshots": snapshots,
    }
    html = render_html(viewer_data)
    out_path = (
        Path(args.out) if args.out else visualize_dir / f"worker_{args.worker_id}.html"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
