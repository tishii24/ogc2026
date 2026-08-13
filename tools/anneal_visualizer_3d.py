# ruff: noqa
# type: ignore
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union


KEYFRAME_INTERVAL = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lightweight 3D annealing viewer from worker JSONL snapshots."
    )
    parser.add_argument("problem", help="Problem JSON path")
    parser.add_argument("visualize_dir", help="Directory containing worker_{id}.jsonl")
    parser.add_argument("--worker-id", type=int, required=True, help="Worker id to visualize")
    parser.add_argument(
        "--out", help="Output HTML path. default: {visualize_dir}/worker_{id}_3d.html"
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=1.0,
        help="Ratio of snapshots to render (0 < ratio <= 1, default: 1.0)",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    snapshots = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if item.get("type") == "snapshot":
                snapshots.append(item)
    snapshots.sort(key=lambda item: (float(item.get("elapsed", 0)), int(item.get("iter", 0))))
    return snapshots


def sample_snapshots(
    snapshots: list[dict[str, Any]], ratio: float
) -> list[dict[str, Any]]:
    if not 0 < ratio <= 1:
        raise ValueError("sampling ratio must satisfy 0 < ratio <= 1")
    if len(snapshots) <= 1 or ratio == 1.0:
        return snapshots
    target_count = min(len(snapshots), max(2, round(len(snapshots) * ratio)))
    indices = {
        round(i * (len(snapshots) - 1) / (target_count - 1))
        for i in range(target_count)
    }
    return [snapshots[index] for index in sorted(indices)]


def orientation_union(orientation: dict[str, Any]) -> Polygon | MultiPolygon:
    polygons = []
    for layer in orientation.get("layers", []):
        if len(layer) < 3:
            continue
        polygon = Polygon([(float(point[0]), float(point[1])) for point in layer])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    if not polygons:
        return Polygon()
    geometry = unary_union(polygons)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    parts = [part for part in geometry.geoms if isinstance(part, Polygon)]
    return unary_union(parts) if parts else Polygon()


def polygon_parts(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    return [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)


def triangle_normal(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    a, b, c = points
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def add_triangle(
    positions: list[float], normals: list[float], points: list[tuple[float, float, float]]
) -> None:
    normal = triangle_normal(points)
    for point in points:
        positions.extend(point)
        normals.extend(normal)


def add_extruded_polygon(
    positions: list[float], normals: list[float], polygon: Polygon
) -> None:
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        coordinates = list(triangle.exterior.coords)[:3]
        bottom = [(x, y, 0.0) for x, y in reversed(coordinates)]
        top = [(x, y, 1.0) for x, y in coordinates]
        add_triangle(positions, normals, bottom)
        add_triangle(positions, normals, top)

    for ring in [polygon.exterior, *polygon.interiors]:
        coordinates = list(ring.coords)
        for start, end in zip(coordinates, coordinates[1:]):
            x0, y0 = start
            x1, y1 = end
            add_triangle(
                positions,
                normals,
                [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, 1.0)],
            )
            add_triangle(
                positions,
                normals,
                [(x0, y0, 0.0), (x1, y1, 1.0), (x0, y0, 1.0)],
            )


def build_meshes(problem: dict[str, Any]) -> list[list[dict[str, list[float]]]]:
    meshes = []
    for block in problem.get("blocks", []):
        block_meshes = []
        for orientation in block.get("shape", []):
            positions: list[float] = []
            normals: list[float] = []
            for polygon in polygon_parts(orientation_union(orientation)):
                add_extruded_polygon(positions, normals, polygon)
            block_meshes.append({"p": positions, "n": normals})
        meshes.append(block_meshes)
    return meshes


def compact_scheduled(scheduled: dict[str, Any]) -> list[int]:
    return [
        int(scheduled["block_id"]),
        int(scheduled["bay_id"]),
        int(scheduled["orient_idx"]),
        int(scheduled["x"]),
        int(scheduled["y"]),
        int(scheduled["entry_time"]),
        int(scheduled["exit_time"]),
    ]


def build_frames(snapshots: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    frames = []
    previous: dict[int, list[int]] = {}
    time_max = 1.0
    for index, snapshot in enumerate(snapshots):
        current = {
            scheduled[0]: scheduled
            for scheduled in map(compact_scheduled, snapshot.get("schedule", []))
        }
        time_max = max(time_max, *(item[6] for item in current.values()))
        frame = {
            "iter": int(snapshot.get("iter", 0)),
            "elapsed": float(snapshot.get("elapsed", 0)),
            "reason": snapshot.get("reason") or "-",
            "neighbor": snapshot.get("neighbor") or "-",
            "accepted": bool(snapshot.get("accepted", False)),
            "improvedCurrent": bool(snapshot.get("improved_current", False)),
            "improvedBest": bool(snapshot.get("improved_best", False)),
            "score": float(snapshot.get("score", 0)),
            "currentScore": float(snapshot.get("current_score", 0)),
            "bestScore": float(snapshot.get("best_score", 0)),
            "delta": float(snapshot.get("delta", 0)),
            "changed": [int(value) for value in snapshot.get("changed_block_ids", [])],
            "selected": [int(value) for value in snapshot.get("selected_block_ids", [])],
            "improvements": snapshot.get("block_improvements", []),
        }
        if index % KEYFRAME_INTERVAL == 0:
            frame["full"] = list(current.values())
        else:
            frame["updates"] = [
                scheduled
                for block_id, scheduled in current.items()
                if previous.get(block_id) != scheduled
            ]
            removed = [block_id for block_id in previous if block_id not in current]
            if removed:
                frame["removed"] = removed
        frames.append(frame)
        previous = current
    return frames, time_max


def block_colors(count: int) -> list[list[float]]:
    colors = []
    for block_id in range(count):
        hue = (block_id * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.42, 0.82)
        colors.append([red, green, blue])
    return colors


def render_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__VIEWER_DATA__", data_json)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OGC 2026 Annealing 3D Viewer</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }
  .panel, .bay-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }
  .panel { padding: 12px; margin-bottom: 12px; }
  .topbar { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }
  label { font-size: 13px; color: #374151; display: inline-flex; gap: 6px; align-items: center; }
  button, select { font: inherit; border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 6px; padding: 5px 10px; }
  button { cursor: pointer; }
  button:hover { background: #eef2ff; }
  #frameSlider { min-width: 320px; flex: 1; }
  #frameLabel, #summary, #blockInfo { font-family: Menlo, Consolas, monospace; font-size: 12px; }
  #summary { margin-top: 10px; line-height: 1.7; }
  #bays { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; margin-bottom: 12px; }
  .bay-card { padding: 10px; min-width: 0; }
  .bay-title { display: flex; justify-content: space-between; align-items: center; font-size: 14px; font-weight: 600; margin-bottom: 6px; }
  .bay-title span { color: #6b7280; font-size: 12px; font-weight: 400; }
  canvas.view { width: 100%; height: 430px; display: block; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; cursor: grab; }
  canvas.view:active { cursor: grabbing; }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 12px; color: #4b5563; }
  .chip { display: inline-flex; gap: 5px; align-items: center; }
  .swatch { width: 17px; height: 11px; border: 1px solid #94a3b8; border-radius: 2px; }
  .muted { color: #6b7280; }
  .status-accepted { color: #047857; font-weight: 600; }
  .status-rejected { color: #b91c1c; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-family: Menlo, Consolas, monospace; font-size: 12px; }
  th, td { padding: 5px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; }
  th:first-child, td:first-child, th:nth-child(3), td:nth-child(3) { text-align: left; }
  @media (max-width: 600px) { #bays { grid-template-columns: 1fr; } canvas.view { height: 340px; } }
</style>
</head>
<body>
<div class="panel">
  <div class="topbar">
    <button id="playButton">Play</button>
    <label>Speed
      <select id="speedSelect">
        <option value="1">1x</option><option value="2">2x</option>
        <option value="5" selected>5x</option><option value="10">10x</option><option value="20">20x</option>
      </select>
    </label>
    <label>Neighbor <select id="neighborSelect"></select></label>
    <label><input id="improvedOnly" type="checkbox"> Improved current only</label>
    <button id="resetCamera">Reset camera</button>
    <span id="frameLabel"></span>
  </div>
  <div class="topbar" style="margin-top:10px"><input id="frameSlider" type="range" min="0" max="0" step="1" value="0"></div>
  <div id="summary"></div>
  <div class="legend">
    <span class="chip"><span class="swatch" style="background:#567d91"></span>changed</span>
    <span class="chip"><span class="swatch" style="background:#766b8f"></span>selected</span>
    <span class="muted">Drag: rotate · Wheel: zoom</span>
  </div>
</div>
<div id="bays"></div>
<div class="panel"><strong>Block information</strong><div id="blockInfo" class="muted" style="margin-top:8px">Click a block in the changed/selected list.</div></div>
<div class="panel"><strong>Improved blocks</strong><div id="improvements" class="muted" style="margin-top:8px">No block score improvements.</div></div>
<script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('viewer-data').textContent);
const frames = data.frames;
const bays = data.bays;
const meshes = data.meshes;
const colors = data.colors;
const frameSlider = document.getElementById('frameSlider');
const playButton = document.getElementById('playButton');
const speedSelect = document.getElementById('speedSelect');
const neighborSelect = document.getElementById('neighborSelect');
const improvedOnly = document.getElementById('improvedOnly');
const frameLabel = document.getElementById('frameLabel');
const summary = document.getElementById('summary');
const blockInfo = document.getElementById('blockInfo');
const improvements = document.getElementById('improvements');
const baysRoot = document.getElementById('bays');
const changedColor = [0.337, 0.49, 0.569];
const selectedColor = [0.463, 0.42, 0.561];
let visibleFrames = frames.map((_, index) => index);
let visibleIndex = 0;
let playing = false;
let lastPlayTime = 0;
let currentState = new Map();
let currentFrameIndex = -1;
let camera = { yaw: 0.72, pitch: 0.48, distance: 2.55 };
const views = [];

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
function program(gl, vertex, fragment) {
  const result = gl.createProgram();
  gl.attachShader(result, compile(gl, gl.VERTEX_SHADER, vertex));
  gl.attachShader(result, compile(gl, gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(result);
  if (!gl.getProgramParameter(result, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(result));
  return result;
}
const VERTEX_SHADER = `#version 300 es
in vec3 aPosition; in vec3 aNormal;
uniform mat4 uViewProjection; uniform vec3 uOffset; uniform vec3 uNorm; uniform float uHeight;
out vec3 vNormal;
void main() {
  vec3 world = vec3((aPosition.xy + uOffset.xy) * uNorm.xy, (uOffset.z + aPosition.z * uHeight) * uNorm.z);
  gl_Position = uViewProjection * vec4(world, 1.0);
  vNormal = normalize(vec3(aNormal.xy / uNorm.xy, aNormal.z / max(uNorm.z / max(uHeight, 0.0001), 0.0001)));
}`;
const FRAGMENT_SHADER = `#version 300 es
precision highp float; in vec3 vNormal; uniform vec3 uColor; out vec4 outColor;
void main() {
  vec3 light = normalize(vec3(0.5, 0.7, 1.0));
  float shade = 0.68 + 0.32 * abs(dot(normalize(vNormal), light));
  outColor = vec4(uColor * shade, 1.0);
}`;
const LINE_VERTEX_SHADER = `#version 300 es
in vec3 aPosition; uniform mat4 uViewProjection; void main(){ gl_Position = uViewProjection * vec4(aPosition,1.0); }`;
const LINE_FRAGMENT_SHADER = `#version 300 es
precision highp float; out vec4 outColor; void main(){ outColor=vec4(0.55,0.58,0.63,1.0); }`;

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) {
    let value=0; for (let k=0;k<4;k++) value += a[k*4+r]*b[c*4+k]; out[c*4+r]=value;
  }
  return out;
}
function perspective(fovy, aspect, near, far) {
  const f=1/Math.tan(fovy/2), nf=1/(near-far), out=new Float32Array(16);
  out[0]=f/aspect; out[5]=f; out[10]=(far+near)*nf; out[11]=-1; out[14]=2*far*near*nf; return out;
}
function normalize(v) { const l=Math.hypot(...v)||1; return v.map(x=>x/l); }
function cross(a,b) { return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]; }
function lookAt(eye, center, up) {
  const z=normalize(eye.map((v,i)=>v-center[i])), x=normalize(cross(up,z)), y=cross(z,x), out=new Float32Array(16);
  out[0]=x[0];out[1]=y[0];out[2]=z[0];out[4]=x[1];out[5]=y[1];out[6]=z[1];out[8]=x[2];out[9]=y[2];out[10]=z[2];out[15]=1;
  out[12]=-x.reduce((s,v,i)=>s+v*eye[i],0);out[13]=-y.reduce((s,v,i)=>s+v*eye[i],0);out[14]=-z.reduce((s,v,i)=>s+v*eye[i],0); return out;
}
function viewProjection(width, height) {
  const cp=Math.cos(camera.pitch), target=[0.5,0.35,0.55];
  const eye=[target[0]+camera.distance*cp*Math.cos(camera.yaw),target[1]+camera.distance*cp*Math.sin(camera.yaw),target[2]+camera.distance*Math.sin(camera.pitch)];
  return mat4Multiply(perspective(Math.PI/4, width/Math.max(1,height), 0.05, 20), lookAt(eye,target,[0,0,1]));
}
function boxLines(bay) {
  const x=Number(bay.width)/data.maxWidth, y=Number(bay.height)/data.maxWidth, z=data.zScale;
  const p=[[0,0,0],[x,0,0],[x,y,0],[0,y,0],[0,0,z],[x,0,z],[x,y,z],[0,y,z]];
  const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  return new Float32Array(edges.flatMap(([a,b])=>[...p[a],...p[b]]));
}
function createView(canvas, bayId) {
  const gl=canvas.getContext('webgl2',{antialias:true}); if(!gl) throw new Error('WebGL2 is not supported');
  const meshProgram=program(gl,VERTEX_SHADER,FRAGMENT_SHADER), lineProgram=program(gl,LINE_VERTEX_SHADER,LINE_FRAGMENT_SHADER);
  const lineBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,lineBuffer); gl.bufferData(gl.ARRAY_BUFFER,boxLines(bays[bayId]),gl.STATIC_DRAW);
  const view={canvas,gl,bayId,meshProgram,lineProgram,lineBuffer,buffers:new Map()};
  let dragging=false,lastX=0,lastY=0;
  canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!dragging)return;camera.yaw-=(e.clientX-lastX)*0.008;camera.pitch=Math.max(-0.05,Math.min(1.35,camera.pitch+(e.clientY-lastY)*0.008));lastX=e.clientX;lastY=e.clientY;drawAll();});
  canvas.addEventListener('pointerup',()=>dragging=false);
  canvas.addEventListener('wheel',e=>{e.preventDefault();camera.distance=Math.max(1.1,Math.min(6,camera.distance*Math.exp(e.deltaY*0.001)));drawAll();},{passive:false});
  return view;
}
function meshBuffer(view, blockId, orientId) {
  const key=`${blockId}:${orientId}`; if(view.buffers.has(key)) return view.buffers.get(key);
  const source=meshes[blockId]?.[orientId] || {p:[],n:[]}, gl=view.gl;
  const position=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,position); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(source.p),gl.STATIC_DRAW);
  const normal=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,normal); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(source.n),gl.STATIC_DRAW);
  const result={position,normal,count:source.p.length/3}; view.buffers.set(key,result); return result;
}
function resize(view) {
  const dpr=window.devicePixelRatio||1, rect=view.canvas.getBoundingClientRect(), w=Math.max(1,Math.round(rect.width*dpr)), h=Math.max(1,Math.round(rect.height*dpr));
  if(view.canvas.width!==w||view.canvas.height!==h){view.canvas.width=w;view.canvas.height=h;} return [w,h];
}
function drawView(view) {
  const gl=view.gl,[width,height]=resize(view),vp=viewProjection(width,height); gl.viewport(0,0,width,height); gl.clearColor(1,1,1,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
  gl.useProgram(view.lineProgram); gl.bindBuffer(gl.ARRAY_BUFFER,view.lineBuffer); const lp=gl.getAttribLocation(view.lineProgram,'aPosition'); gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp,3,gl.FLOAT,false,0,0); gl.uniformMatrix4fv(gl.getUniformLocation(view.lineProgram,'uViewProjection'),false,vp); gl.drawArrays(gl.LINES,0,24);
  const frame=frames[currentFrameIndex]||{},changed=new Set(frame.changed||[]),selected=new Set(frame.selected||[]); gl.useProgram(view.meshProgram); gl.uniformMatrix4fv(gl.getUniformLocation(view.meshProgram,'uViewProjection'),false,vp); gl.uniform3f(gl.getUniformLocation(view.meshProgram,'uNorm'),1/data.maxWidth,1/data.maxWidth,data.zScale/data.timeMax);
  for(const scheduled of currentState.values()) {
    const [blockId,bayId,orientId,x,y,entry,exit]=scheduled; if(bayId!==view.bayId) continue; const buffer=meshBuffer(view,blockId,orientId); if(!buffer.count) continue;
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer.position); const ap=gl.getAttribLocation(view.meshProgram,'aPosition'); gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer.normal); const an=gl.getAttribLocation(view.meshProgram,'aNormal'); gl.enableVertexAttribArray(an); gl.vertexAttribPointer(an,3,gl.FLOAT,false,0,0);
    gl.uniform3f(gl.getUniformLocation(view.meshProgram,'uOffset'),x,y,entry); gl.uniform1f(gl.getUniformLocation(view.meshProgram,'uHeight'),Math.max(0.001,exit-entry)); const color=selected.has(blockId)?selectedColor:(changed.has(blockId)?changedColor:colors[blockId]); gl.uniform3fv(gl.getUniformLocation(view.meshProgram,'uColor'),color); gl.drawArrays(gl.TRIANGLES,0,buffer.count);
  }
}
function drawAll(){for(const view of views)drawView(view);}
function applyFrame(state, frame) { if(frame.full){state.clear();for(const item of frame.full)state.set(item[0],item);} else {for(const id of frame.removed||[])state.delete(id);for(const item of frame.updates||[])state.set(item[0],item);} }
function stateAt(index) {
  if(index===currentFrameIndex+1){applyFrame(currentState,frames[index]);currentFrameIndex=index;return;}
  const start=Math.floor(index/data.keyframeInterval)*data.keyframeInterval; currentState=new Map(); for(let i=start;i<=index;i++)applyFrame(currentState,frames[i]); currentFrameIndex=index;
}
function showBlock(blockId) {
  const item=currentState.get(Number(blockId)); if(!item){blockInfo.textContent=`B${blockId}: not in this schedule`;return;} const [id,bay,orient,x,y,entry,exit]=item; blockInfo.textContent=`B${id}  bay=${bay}  orient=${orient}  x=${x}  y=${y}  entry=${entry}  exit=${exit}`;
}
function idLinks(ids){return ids.length?ids.map(id=>`<button
 data-block="${id}" style="padding:2px 6px">B${id}</button>`).join(' '):'<span class="muted">none</span>';}
function updateInfo() {
  const frame=frames[currentFrameIndex],status=frame.accepted?'accepted':'rejected'; frameLabel.textContent=`${visibleIndex+1} / ${visibleFrames.length}`;
  summary.innerHTML=`<span class="status-${status}">${status}</span> · elapsed=${frame.elapsed.toFixed(4)}s · iter=${frame.iter} · neighbor=${frame.neighbor} · reason=${frame.reason}<br>score=${frame.score.toFixed(3)} · current=${frame.currentScore.toFixed(3)} · best=${frame.bestScore.toFixed(3)} · delta=${frame.delta.toFixed(3)}<br>changed: ${idLinks(frame.changed||[])} &nbsp; selected: ${idLinks(frame.selected||[])}`;
  summary.querySelectorAll('[data-block]').forEach(button=>button.addEventListener('click',()=>showBlock(button.dataset.block)));
  if(!(frame.improvements||[]).length){improvements.innerHTML='<span class="muted">No block score improvements.</span>';} else {improvements.innerHTML='<table><thead><tr><th>Block</th><th>Improvement</th><th>Reasons</th><th>Tardiness</th><th>Preference</th></tr></thead><tbody>'+frame.improvements.map(item=>`<tr><td><button data-block="${item.block_id}" style="padding:2px 6px">B${item.block_id}</button></td><td>${Number(item.score_improvement).toFixed(3)}</td><td>${(item.reasons||[]).join(', ')}</td><td>${item.tardiness.before} → ${item.tardiness.after}</td><td>${item.preference_penalty.before} → ${item.preference_penalty.after}</td></tr>`).join('')+'</tbody></table>'; improvements.querySelectorAll('[data-block]').forEach(button=>button.addEventListener('click',()=>showBlock(button.dataset.block)));}
}
function showVisible(index) { if(!visibleFrames.length)return; visibleIndex=Math.max(0,Math.min(index,visibleFrames.length-1)); frameSlider.value=String(visibleIndex); stateAt(visibleFrames[visibleIndex]); updateInfo(); drawAll(); }
function rebuildFilter() { const neighbor=neighborSelect.value; visibleFrames=frames.map((f,i)=>({f,i})).filter(({f})=>(neighbor==='*'||f.neighbor===neighbor)&&(!improvedOnly.checked||f.improvedCurrent)).map(({i})=>i); frameSlider.max=String(Math.max(0,visibleFrames.length-1)); frameSlider.disabled=!visibleFrames.length; showVisible(0); }
function animate(now) { if(!playing)return; const interval=1000/(5*Number(speedSelect.value)); if(now-lastPlay
Time>=interval){lastPlayTime=now;if(visibleIndex+1>=visibleFrames.length){playing=false;playButton.textContent='Play';return;}showVisible(visibleIndex+1);}requestAnimationFrame(animate);}

for(const [bayId,bay] of bays.entries()) { const card=document.createElement('div');card.className='bay-card';card.innerHTML=`<div class="bay-title">Bay ${bayId}<span>${bay.width} × ${bay.height}</span></div>`;const canvas=document.createElement('canvas');canvas.className='view';card.appendChild(canvas);baysRoot.appendChild(card);views.push(createView(canvas,bayId)); }
const neighbors=[...new Set(frames.map(frame=>frame.neighbor))].sort(); neighborSelect.innerHTML='<option value="*">All</option>'+neighbors.map(value=>`<option value="${value}">${value}</option>`).join('');
frameSlider.max=String(Math.max(0,frames.length-1)); frameSlider.addEventListener('input',()=>showVisible(Number(frameSlider.value))); neighborSelect.addEventListener('change',rebuildFilter); improvedOnly.addEventListener('change',rebuildFilter);
playButton.addEventListener('click',()=>{playing=!playing;playButton.textContent=playing?'Pause':'Play';lastPlayTime=0;if(playing)requestAnimationFrame(animate);});
document.getElementById('resetCamera').addEventListener('click',()=>{camera={yaw:0.72,pitch:0.48,distance:2.55};drawAll();}); window.addEventListener('resize',drawAll);
showVisible(0);
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
        if not snapshots:
            raise ValueError(f"no snapshots in {worker_path}")
        original_count = len(snapshots)
        snapshots = sample_snapshots(snapshots, args.sampling_ratio)
        print(
            f"snapshots: {len(snapshots)} / {original_count}; building reusable meshes...",
            file=sys.stderr,
            flush=True,
        )
        meshes = build_meshes(problem)
        print("compressing snapshot schedules...", file=sys.stderr, flush=True)
        frames, time_max = build_frames(snapshots)
        bays = problem.get("bays", [])
        if not bays:
            raise ValueError("problem has no bays")
        data = {
            "bays": bays,
            "meshes": meshes,
            "frames": frames,
            "colors": block_colors(len(problem.get("blocks", []))),
            "maxWidth": max(float(bay.get("width", 1)) for bay in bays),
            "timeMax": time_max,
            "zScale": 1.35,
            "keyframeInterval": KEYFRAME_INTERVAL,
        }
        html = render_html(data)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = (
        Path(args.out)
        if args.out
        else visualize_dir / f"worker_{args.worker_id}_3d.html"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
