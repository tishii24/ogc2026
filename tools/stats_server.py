# type: ignore
#!/usr/bin/env python3
"""Serve a small browser UI that runs tools/stats.py on each request."""

from __future__ import annotations

import argparse
import html
import json
import shlex
import subprocess
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the stats.py browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def query_value(query: dict[str, list[str]], name: str) -> str:
    return query.get(name, [""])[0].strip()


def list_suites(root: Path) -> list[str]:
    return [
        str(path.relative_to(root))
        for path in sorted((root / "suites").glob("*.json"))
    ]


class StatsHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: object, root: Path, **kwargs: object) -> None:
        self.root = root
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path != "/":
            self.send_error(404)
            return

        query = parse_qs(url.query, keep_blank_values=True)
        suite = query_value(query, "suite")
        suite_options = list_suites(self.root)
        timelimit = query_value(query, "tl")
        last_versions = query_value(query, "n")
        matrix = query_value(query, "m") == "1"
        include_tune = query_value(query, "include_tune") == "1"
        all_feasible = query_value(query, "all_feasible") == "1"

        stats_args: list[str] = []
        if suite:
            stats_args.extend(["--suite", suite])
        if timelimit:
            stats_args.extend(["--tl", timelimit])
        if last_versions:
            stats_args.extend(["--last-versions", last_versions])
        if matrix:
            stats_args.append("--matrix")
        if include_tune:
            stats_args.append("--include-tune")
        if all_feasible:
            stats_args.append("--all-feasible")
        stats_args.append("--json")

        command = [sys.executable, str(self.root / "tools" / "stats.py"), *stats_args]
        result = subprocess.run(
            command,
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode:
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            content = f'<pre class="error">{html.escape(output)}</pre>'
        else:
            data = json.loads(result.stdout)
            content = (
                self.render_matrix_table(data)
                if matrix
                else self.render_summary_table(data)
            )

        display_command = " ".join(
            shlex.quote(part)
            for part in [sys.executable, "tools/stats.py", *stats_args]
        )
        page = self.render_page(
            suite,
            suite_options,
            timelimit,
            last_versions,
            matrix,
            include_tune,
            all_feasible,
            display_command,
            content,
            result.returncode,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    @staticmethod
    def render_summary_table(summaries: list[dict[str, Any]]) -> str:
        headers = [
            "version",
            "tl",
            "cases",
            "feasible",
            "failed",
            "best",
            "rank_score",
            "relative_score",
            "total_objective",
            "total_obj1",
            "total_obj2",
            "total_obj3",
            "total_elapsed",
        ]
        rows = [
            [
                str(item["version"]),
                str(round(item["timelimit"])),
                str(item["cases"]),
                str(item["feasible"]),
                str(item["failed"]),
                str(item["best"]),
                str(item["rank_score"]),
                f'{item["relative_score"]:.3f}',
                str(round(item["total_objective"])),
                str(round(item["total_obj1"])),
                str(round(item["total_obj2"])),
                str(round(item["total_obj3"])),
                f'{item["total_elapsed"]:.3f}',
            ]
            for item in summaries
        ]
        return StatsHandler.render_table(headers, rows, matrix=False)

    @staticmethod
    def render_matrix_table(matrix_data: dict[str, Any]) -> str:
        headers = [str(value) for value in matrix_data["headers"]]
        rows = [[str(value) for value in row] for row in matrix_data["rows"]]
        return StatsHandler.render_table(headers, rows, matrix=True)

    @staticmethod
    def render_table(
        headers: list[str], rows: list[list[str]], matrix: bool
    ) -> str:
        best_row = next((row for row in rows if row and row[0] == "best"), None)
        header_html = "".join(
            f'<th class="{"numeric" if index else ""}">{html.escape(header)}</th>'
            for index, header in enumerate(headers)
        )
        body_rows = []
        for row in rows:
            row_class = "best-row" if row and row[0] == "best" else ""
            cells = []
            for index, value in enumerate(row):
                classes = []
                if index:
                    classes.append("numeric")
                if value == "NG":
                    classes.append("ng")
                elif value == "-":
                    classes.append("muted")
                if (
                    matrix
                    and best_row is not None
                    and row is not best_row
                    and headers[index] not in {"version", "relative_score", "tl"}
                    and value not in {"NG", "-"}
                    and value == best_row[index]
                ):
                    classes.append("best-cell")
                if (
                    not matrix
                    and headers[index] == "failed"
                    and value != "0"
                ):
                    classes.append("ng")
                class_attr = f' class="{" ".join(classes)}"' if classes else ""
                cells.append(f"<td{class_attr}>{html.escape(value)}</td>")
            body_rows.append(
                f'<tr class="{row_class}">' + "".join(cells) + "</tr>"
            )
        table_class = "matrix" if matrix else "summary"
        return (
            '<div class="table-wrap">'
            f'<table class="{table_class}"><thead><tr>{header_html}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table></div>'
        )

    @staticmethod
    def render_page(
        suite: str,
        suite_options: list[str],
        timelimit: str,
        last_versions: str,
        matrix: bool,
        include_tune: bool,
        all_feasible: bool,
        command: str,
        content: str,
        returncode: int,
    ) -> str:
        matrix_checked = " checked" if matrix else ""
        tune_checked = " checked" if include_tune else ""
        feasible_checked = " checked" if all_feasible else ""
        status_class = "error" if returncode else ""
        suite_option_html = ['<option value="">all cases</option>']
        for option in suite_options:
            selected = " selected" if option == suite else ""
            escaped = html.escape(option, quote=True)
            suite_option_html.append(
                f'<option value="{escaped}"{selected}>{escaped}</option>'
            )
        return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OGC 2026 Stats</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2937; background: #f5f7fa; }}
h1 {{ margin: 0 0 16px; }}
form {{ display: flex; flex-wrap: wrap; align-items: end; gap: 12px; margin-bottom: 16px; padding: 14px; border: 1px solid #d8dee8; border-radius: 8px; background: white; }}
label {{ display: grid; gap: 4px; font-size: 13px; font-weight: 600; }}
input[type="text"], input[type="number"], select {{ padding: 6px 8px; border: 1px solid #b8c1cf; border-radius: 4px; }}
select[name="suite"] {{ width: 220px; }}
.checkbox {{ display: flex; align-items: center; gap: 5px; padding-bottom: 6px; }}
button {{ padding: 7px 16px; border: 0; border-radius: 4px; color: white; background: #2563eb; cursor: pointer; }}
button:hover {{ background: #1d4ed8; }}
.command {{ color: #64748b; font: 12px monospace; margin-bottom: 8px; }}
.table-wrap {{ max-height: calc(100vh - 220px); overflow: auto; border: 1px solid #d8dee8; border-radius: 8px; background: white; }}
table {{ width: 100%; border-spacing: 0; font-size: 13px; white-space: nowrap; }}
table.matrix {{ width: max-content; min-width: 100%; }}
th, td {{ padding: 8px 10px; border-right: 1px solid #e5e9f0; border-bottom: 1px solid #e5e9f0; }}
th:last-child, td:last-child {{ border-right: 0; }}
th {{ position: sticky; top: 0; z-index: 1; text-align: left; color: #475569; background: #eef2f7; }}
td.numeric, th.numeric {{ text-align: right; font-variant-numeric: tabular-nums; }}
tbody tr:nth-child(even):not(.best-row) {{ background: #f8fafc; }}
tbody tr:hover:not(.best-row) {{ background: #eef6ff; }}
.best-row {{ font-weight: 700; background: #dcfce7; }}
.best-cell {{ font-weight: 700; color: #166534; background: #ecfdf3; }}
.ng {{ font-weight: 600; color: #b91c1c; background: #fef2f2; }}
.muted {{ color: #94a3b8; }}
pre {{ margin: 0; padding: 12px; border: 1px solid #fecaca; border-radius: 8px; background: #fff7f7; overflow: auto; }}
.error {{ color: #b00020; }}
</style>
</head>
<body>
<h1>Stats</h1>
<form method="get" action="/">
<label>suite
<select name="suite">{"".join(suite_option_html)}</select>
</label>
<label>tl
<input type="number" name="tl" value="{html.escape(timelimit, quote=True)}" step="any">
</label>
<label>last versions (-n)
<input type="number" name="n" value="{html.escape(last_versions, quote=True)}" min="1" step="1">
</label>
<label class="checkbox"><input type="checkbox" name="m" value="1"{matrix_checked}>matrix (-m)</label>
<label class="checkbox"><input type="checkbox" name="include_tune" value="1"{tune_checked}>include tune</label>
<label class="checkbox"><input type="checkbox" name="all_feasible" value="1"{feasible_checked}>all feasible</label>
<button type="submit">表示</button>
</form>
<div class="command {status_class}">$ {html.escape(command)}</div>
{content}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    handler = partial(StatsHandler, root=root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
