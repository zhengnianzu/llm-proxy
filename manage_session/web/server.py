#!/usr/bin/env python3
"""
web/server.py — FastAPI Web UI for manage_session

启动方式:
    cd manage_session
    python web/server.py --port 8081

访问: http://localhost:8081
"""

import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 保证 core 包可 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import manifest as manifest_mod
from core import views    as views_mod
from core import config   as config_mod

# ── setup ─────────────────────────────────────────────────────────────────────

TEMPLATES_DIR = config_mod.web_templates_dir()
STATIC_DIR    = config_mod.web_static_dir()

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Session Manager")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """全局汇总页面"""
    summary = views_mod.update_global_summary()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": summary},
    )


@app.get("/batch/{batch_id:path}", response_class=HTMLResponse)
def batch_detail(request: Request, batch_id: str):
    """批次详情页面"""
    safe_id = batch_id.replace("/", "_").replace("\\", "_")
    bv_path = config_mod.views_dir() / "batch_status" / f"{safe_id}.json"
    if not bv_path.exists():
        raise HTTPException(status_code=404, detail="Batch not found")
    with open(bv_path, encoding="utf-8") as f:
        bv = json.load(f)
    return templates.TemplateResponse(
        "batch_detail.html",
        {"request": request, "bv": bv},
    )


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/summary")
def api_summary():
    return views_mod.update_global_summary()


@app.get("/api/batch/{batch_id:path}")
def api_batch(batch_id: str):
    safe_id = batch_id.replace("/", "_").replace("\\", "_")
    bv_path = config_mod.views_dir() / "batch_status" / f"{safe_id}.json"
    if not bv_path.exists():
        raise HTTPException(status_code=404, detail="Batch not found")
    with open(bv_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/report-html/{index_id:path}", response_class=HTMLResponse)
def report_html(index_id: str):
    """直接返回 xlsx index 同目录下的 session_report.html"""
    idx = manifest_mod.get_index(index_id)
    if not idx:
        raise HTTPException(status_code=404, detail="Index not found")
    if idx.get("format") != "xlsx":
        raise HTTPException(status_code=400, detail="Not an xlsx index")
    html_path = Path(idx["abs_path"]).parent / "session_report.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="session_report.html not found")
    return FileResponse(str(html_path), media_type="text/html")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=config_mod.web_port())
    parser.add_argument("--host", default=config_mod.web_host())
    args = parser.parse_args()

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port)
