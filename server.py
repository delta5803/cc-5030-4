"""Multi-end production panel server.

The UI never touches media code directly: every control goes through the
multi-end recording kernel (Coordinator), and kernel results (including
failures) are fed back to the panel.
"""
import copy
import json
import os
import threading
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from multirec.coordinator import Coordinator, DEFAULT_NODES
from multirec import verify as verify_mod

BASE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(BASE, "recordings")
STATIC = os.path.join(BASE, "static")

kernel = Coordinator(recordings_dir=RECORDINGS)
for cfg in DEFAULT_NODES:
    kernel.add_node(cfg)

app = FastAPI(title="Multi-End Recording Panel")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/files", StaticFiles(directory=RECORDINGS), name="files")


class SessionRequest(BaseModel):
    nodes: list[str]
    duration: float = 2.0
    policy: str = "cancel"          # cancel | partial
    prepare_timeout: float = 3.0
    fail: dict[str, str] = {}       # node_id -> "prepare" | "start"
    loss: list[str] = []            # node_ids losing ack/first-sample replies


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/nodes")
def nodes():
    return {"nodes": kernel.node_states(),
            "session_running": kernel._session_lock.locked()}


@app.post("/api/sessions")
def start_session(req: SessionRequest):
    if kernel._session_lock.locked():
        return {"error": "another session is running"}, 409
    if req.policy not in ("cancel", "partial"):
        return {"error": "policy must be cancel|partial"}, 400
    unknown = [n for n in req.nodes if n not in kernel.nodes]
    if unknown:
        return {"error": f"unknown nodes: {unknown}"}, 400
    if not req.nodes:
        return {"error": "no nodes selected"}, 400

    # apply fault injection for this run
    for nid, node in kernel.nodes.items():
        node.fail_at = req.fail.get(nid)
        kernel.links_down[nid].loss = 0.35 if nid in req.loss else 0.0

    holder = {}

    def run():
        try:
            holder["rec"] = kernel.run_session(
                req.nodes, duration=req.duration, policy=req.policy,
                prepare_timeout=req.prepare_timeout)
        finally:
            for nid, node in kernel.nodes.items():
                node.fail_at = None
                kernel.links_down[nid].loss = 0.0

    # pre-register so the panel can poll immediately
    threading.Thread(target=run, daemon=True).start()
    # wait until the session record exists (a few ms)
    sid = None
    for _ in range(100):
        running = [s for s, r in kernel.sessions.items()
                   if r.get("result") == "running"]
        if running:
            sid = running[-1]
            break
        time.sleep(0.01)
    return {"session_id": sid}


def _disk_sessions():
    """Sessions persisted on disk (incl. committed regression samples),
    so they stay openable from the panel after a restart."""
    out = []
    for d in sorted(os.listdir(RECORDINGS)):
        p = os.path.join(RECORDINGS, d, "record.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as fh:
                r = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"session_id": d, "result": r.get("result"),
                    "created_wall": r.get("created_wall"),
                    "nodes": list(r.get("nodes", {}).keys())})
    return out


@app.get("/api/sessions")
def list_sessions():
    out = []
    seen = set()
    for sid, rec in kernel.sessions.items():
        seen.add(sid)
        out.append({"session_id": sid, "result": rec.get("result"),
                    "created_wall": rec.get("created_wall"),
                    "nodes": list(rec.get("nodes", {}).keys())})
    for s in _disk_sessions():
        if s["session_id"] not in seen:
            out.append(s)
    return {"sessions": out}


def _load_record(sid):
    rec = kernel.sessions.get(sid)
    if rec is not None:
        return rec
    path = os.path.join(RECORDINGS, sid, "record.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


@app.get("/api/sessions/{sid}/verify")
def verify_session(sid):
    """Actual computation over the real clips: decode every file, re-derive
    each end's real start from the ground-truth clock, and report per-node
    failures for the panel to display."""
    rec = _load_record(sid)
    if rec is None:
        return {"error": "not found"}, 404
    if rec.get("result") == "running":
        return {"error": "session still running"}, 409
    cfg = {nid: {"color": n.color, "tone": n.tone,
                 "true_offset": n.true_offset}
           for nid, n in kernel.nodes.items()}
    try:
        return verify_mod.verify_session(
            rec, nodes_cfg=cfg, base_dir=os.path.join(RECORDINGS, sid))
    except Exception as exc:  # surfaced to the panel, never swallowed
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    rec = kernel.sessions.get(sid)
    if rec is None:
        path = os.path.join(RECORDINGS, sid, "record.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        return {"error": "not found"}, 404
    for _ in range(3):  # deepcopy may race with the running session
        try:
            return copy.deepcopy(rec)
        except RuntimeError:
            time.sleep(0.005)
    return {"error": "busy"}, 503
