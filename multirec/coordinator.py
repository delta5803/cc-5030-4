"""Coordinator: readiness barrier, reproducible clock-skew estimation,
common start boundary, idempotent sessions, cancel/partial policies.

Three events are kept strictly distinct per node per session:
  start_command_sent  - coordinator dispatched the start command
  start_ack           - the device confirmed it received the command
  first_sample        - the first real media sample hit the file
The recording start point is ALWAYS the first_sample event, never the
command dispatch or ack time.
"""
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone

from .link import Link
from .node import Node

PING_COUNT = 6
DEFAULT_NODES = [
    {"id": "cam1", "kind": "camera", "true_offset": -0.090,
     "startup_delay": 0.30, "color": (35, 60, 150), "seed": 101},
    {"id": "cam2", "kind": "camera", "true_offset": 0.060,
     "startup_delay": 0.10, "color": (150, 45, 40), "seed": 202},
    {"id": "mic1", "kind": "mic", "true_offset": 0.140,
     "startup_delay": 0.20, "tone": 880.0, "seed": 303},
    {"id": "mic2", "kind": "mic", "true_offset": -0.040,
     "startup_delay": 0.45, "tone": 640.0, "seed": 404},
]


class _Pending:
    __slots__ = ("event", "msg")

    def __init__(self):
        self.event = threading.Event()
        self.msg = None


class Coordinator:
    def __init__(self, recordings_dir="recordings"):
        self.recordings_dir = os.path.abspath(recordings_dir)
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.nodes = {}
        self.links_up = {}
        self.links_down = {}
        self._pending = {}
        self._pending_lock = threading.Lock()
        self._nonce = 0
        self._epoch = 0
        self.stale = []            # messages that matched no live barrier
        self.sessions = {}         # session_id -> live record dict
        self._session_lock = threading.Lock()
        self._active_queue = None
        self._active_session = None

    # ---------------- wiring ----------------

    def add_node(self, cfg):
        nid = cfg["id"]
        up = Link(seed=cfg["seed"], name=f"up-{nid}")
        down = Link(seed=cfg["seed"] + 1, name=f"down-{nid}")
        node = Node(nid, cfg["kind"], cfg["true_offset"], cfg["startup_delay"],
                    link_up=up, link_down=down, coord_deliver=self._dispatch,
                    color=cfg.get("color", (40, 60, 120)),
                    tone=cfg.get("tone", 880.0))
        self.nodes[nid] = node
        self.links_up[nid] = up
        self.links_down[nid] = down
        return node

    def node_states(self):
        return [{"id": n.id, "kind": n.kind, "state": n.state,
                 "true_offset": n.true_offset,
                 "startup_delay": n.startup_delay}
                for n in self.nodes.values()]

    # ---------------- message plumbing ----------------

    def _next_nonce(self):
        with self._pending_lock:
            self._nonce += 1
            return self._nonce

    # message types a node emits without being asked (no nonce expected)
    UNSOLICITED = {"first_sample", "done", "cancelled", "error"}

    def _dispatch(self, msg):
        """Entry point for every node->coordinator message."""
        nonce = msg.get("nonce")
        pend = None
        if nonce is not None:
            with self._pending_lock:
                pend = self._pending.pop(nonce, None)
        if pend is not None:
            pend.msg = msg
            pend.event.set()
            return
        if (msg.get("type") in self.UNSOLICITED
                and self._active_queue is not None
                and msg.get("session") == self._active_session):
            self._active_queue.put(msg)
            return
        # No live barrier matches: stale acks from old sessions land here and
        # are recorded, never honoured.
        self.stale.append({"t": time.monotonic(), "msg": dict(msg)})

    def _roundtrip(self, nid, msg, timeout):
        nonce = self._next_nonce()
        msg = dict(msg, nonce=nonce)
        pend = _Pending()
        with self._pending_lock:
            self._pending[nonce] = pend
        self.links_up[nid].send(msg, self.nodes[nid].deliver)
        if pend.event.wait(timeout):
            return pend.msg
        with self._pending_lock:
            self._pending.pop(nonce, None)
        return None

    def _broadcast(self, nids, msg_fn, timeout):
        """Send to all nodes at once, collect replies until deadline.
        On timeout the pending entry is removed, so a late reply becomes a
        recorded stale message instead of being silently consumed."""
        pends = {}
        nonces = {}
        sent_at = {}
        for nid in nids:
            nonce = self._next_nonce()
            pend = _Pending()
            with self._pending_lock:
                self._pending[nonce] = pend
            pends[nid] = pend
            nonces[nid] = nonce
            msg = dict(msg_fn(nid, nonce), nonce=nonce)
            sent_at[nid] = time.monotonic()
            self.links_up[nid].send(msg, self.nodes[nid].deliver)
        deadline = time.monotonic() + timeout
        out = {}
        for nid, pend in pends.items():
            remain = deadline - time.monotonic()
            if remain > 0:
                pend.event.wait(remain)
            # a reply that already arrived is honoured even when another
            # node's loss consumed the rest of the shared deadline
            if pend.event.is_set():
                out[nid] = pend.msg
            else:
                out[nid] = None
                with self._pending_lock:
                    self._pending.pop(nonces[nid], None)
        return out, sent_at

    # ---------------- clock skew estimation ----------------

    def estimate_skew(self, nid, k=PING_COUNT, timeout=0.5):
        """NTP-style exchange. Reproducible: link delays come from the node's
        seeded Link. Returns estimate with its retained error bound."""
        samples = []
        for _ in range(k):
            t0 = time.monotonic()
            pong = self._roundtrip(nid, {"type": "ping"}, timeout)
            t3 = time.monotonic()
            if pong is None:
                samples.append({"lost": True})
                continue
            t1, t2 = pong["t1"], pong["t2"]
            rtt = (t3 - t0) - (t2 - t1)
            theta = ((t1 - t0) + (t2 - t3)) / 2.0
            samples.append({"rtt": rtt, "theta": theta,
                            "t0": t0, "t1": t1, "t2": t2, "t3": t3})
        good = [s for s in samples if "rtt" in s]
        if not good:
            return None
        best = min(good, key=lambda s: s["rtt"])
        return {"theta": best["theta"], "error": best["rtt"] / 2.0,
                "samples": samples}

    # ---------------- session orchestration ----------------

    def run_session(self, node_ids, duration=2.0, policy="cancel",
                    prepare_timeout=3.0, lead=0.6, ack_timeout=0.6,
                    ack_retries=2):
        """Run one coordinated recording. Only one session at a time."""
        with self._session_lock:
            return self._run_session(node_ids, duration, policy,
                                     prepare_timeout, lead, ack_timeout,
                                     ack_retries)

    def _run_session(self, node_ids, duration, policy, prepare_timeout,
                     lead, ack_timeout, ack_retries):
        assert policy in ("cancel", "partial")
        # the start boundary must leave room for the whole ack retry budget
        lead = max(lead, ack_timeout * (1 + ack_retries) + 0.3)
        self._epoch += 1
        sid = uuid.uuid4().hex[:12]
        epoch = self._epoch
        outdir = os.path.join(self.recordings_dir, sid)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.monotonic()
        stale_mark = len(self.stale)

        record = {
            "session_id": sid, "epoch": epoch, "policy": policy,
            "duration": duration,
            "created_wall": datetime.now(timezone.utc).isoformat(),
            "params": {"prepare_timeout": prepare_timeout, "lead": lead},
            "t_start_coord": None, "result": "running",
            "nodes": {}, "events": [],
        }
        self.sessions[sid] = record
        self._active_session = sid
        self._active_queue = queue.Queue()

        def ev(etype, node=None, **detail):
            record["events"].append({
                "t": time.monotonic(), "dt": time.monotonic() - t0,
                "type": etype, "node": node, **detail})

        def node_rec(nid):
            return record["nodes"].setdefault(nid, {
                "kind": self.nodes[nid].kind,
                "true_offset": self.nodes[nid].true_offset,
                "events": [], "file": None, "stats": None, "error": None})

        selected = [n for n in node_ids if n in self.nodes]
        ev("session_created", nodes=selected, policy=policy)
        try:
            # ---- phase 1: reproducible skew estimation -------------------
            skew = {}
            for nid in selected:
                est = self.estimate_skew(nid)
                if est is None:
                    node_rec(nid)["error"] = "unreachable during skew estimation"
                    ev("skew_failed", node=nid)
                else:
                    skew[nid] = est
                    node_rec(nid)["skew"] = est
                    ev("skew_estimated", node=nid,
                       theta=est["theta"], error=est["error"])
            reachable = [n for n in selected if n in skew]
            unreachable = [n for n in selected if n not in skew]
            if unreachable and policy == "cancel":
                ev("policy_cancel", failed=unreachable)
                return self._finish(record, sid, "cancelled", t0, stale_mark)
            if unreachable:
                ev("policy_partial_drop", dropped=unreachable)
            if not reachable:
                ev("no_ready_nodes")
                return self._finish(record, sid, "failed", t0, stale_mark)

            # ---- phase 2: readiness barrier ------------------------------
            def _prepare_msg(nid, nonce):
                ev("prepare_sent", node=nid)
                return {"type": "prepare", "session": sid,
                        "epoch": epoch, "nonce": nonce}

            reqs, _ = self._broadcast(reachable, _prepare_msg, prepare_timeout)
            missing = [n for n in reachable if reqs.get(n) is None]
            if missing:  # one retry round for lost ready replies
                for nid in missing:
                    ev("prepare_retry", node=nid)
                retry, _ = self._broadcast(missing, _prepare_msg,
                                           prepare_timeout)
                reqs.update(retry)
            ready = []
            failed = []
            for nid, reply in reqs.items():
                nr = node_rec(nid)
                if reply is None:
                    nr["error"] = "prepare timeout"
                    ev("prepare_timeout", node=nid)
                    failed.append(nid)
                elif reply.get("type") == "ready":
                    ev("ready", node=nid, t_local=reply.get("t_local"))
                    ready.append(nid)
                else:
                    nr["error"] = reply.get("reason", "prepare failed")
                    ev("node_error", node=nid, stage="prepare",
                       reason=nr["error"])
                    failed.append(nid)
            if failed and policy == "cancel":
                ev("policy_cancel", failed=failed)
                return self._finish(record, sid, "cancelled", t0, stale_mark)
            if failed:
                ev("policy_partial_drop", dropped=failed)
            if not ready:
                ev("no_ready_nodes")
                return self._finish(record, sid, "failed", t0, stale_mark)

            # ---- phase 3: common start boundary --------------------------
            t_start = time.monotonic() + lead
            record["t_start_coord"] = t_start
            record["t_start_dt"] = t_start - t0

            def _start_msg_factory():
                def mk(nid, nonce):
                    est = skew[nid]
                    out = os.path.join(
                        outdir, f"{nid}.{'mp4' if self.nodes[nid].kind == 'camera' else 'wav'}")
                    node_rec(nid)["file"] = out
                    ev("start_command_sent", node=nid,
                       start_at_node=t_start + est["theta"])
                    return {"type": "start", "session": sid, "epoch": epoch,
                            "start_at": t_start + est["theta"],
                            "duration": duration, "out": out}
                return mk

            acks, _ = self._broadcast(ready, _start_msg_factory(),
                                      min(ack_timeout, lead))
            # retry lost acks: same session id, so the node stays idempotent
            for _attempt in range(ack_retries):
                missing = [n for n in ready if acks.get(n) is None]
                if not missing:
                    break
                for nid in missing:
                    ev("start_ack_lost_retry", node=nid)
                retry, _ = self._broadcast(missing, _start_msg_factory(),
                                           ack_timeout)
                acks.update(retry)

            active = []
            for nid in ready:
                ack = acks.get(nid)
                if ack is None:
                    node_rec(nid)["error"] = "start ack timeout"
                    ev("node_error", node=nid, stage="start",
                       reason="start ack timeout")
                elif ack.get("type") == "error":
                    node_rec(nid)["error"] = ack.get("reason", "start failed")
                    ev("node_error", node=nid, stage="start",
                       reason=node_rec(nid)["error"])
                else:
                    ev("start_ack", node=nid,
                       already=ack.get("already", False),
                       t_local=ack.get("t_local"))
                    active.append(nid)
            dropped = [n for n in ready if n not in active]
            if dropped and policy == "cancel":
                for nid in active:
                    self.links_up[nid].send({"type": "cancel", "session": sid},
                                            self.nodes[nid].deliver)
                ev("policy_cancel", failed=dropped)
                return self._finish(record, sid, "cancelled", t0, stale_mark)
            if dropped:
                ev("policy_partial_drop", dropped=dropped)
            if not active:
                return self._finish(record, sid, "failed", t0, stale_mark)

            # ---- phase 4+5: first real samples, then completion ----------
            max_err = max(skew[n]["error"] for n in active)
            first_deadline = t_start + max_err + 1.5
            done_deadline = t_start + duration + 4.0
            pending_first = set(active)
            pending_done = set(active)
            first_seen = set()
            while pending_first or pending_done:
                now = time.monotonic()
                if pending_first:
                    remain = first_deadline - now
                    if remain <= 0:
                        for nid in sorted(pending_first):
                            node_rec(nid)["error"] = \
                                "no first sample before deadline"
                            ev("first_sample_timeout", node=nid)
                            pending_first.discard(nid)
                            pending_done.discard(nid)
                        continue
                else:
                    remain = done_deadline - now
                    if remain <= 0:
                        break
                try:
                    msg = self._active_queue.get(timeout=remain)
                except queue.Empty:
                    continue  # re-evaluate which deadline applies
                nid = msg.get("node")
                mtype = msg.get("type")
                if msg.get("nonce") is not None and nid in self.nodes:
                    # reliable-notification handshake: let the node stop
                    # retransmitting; duplicates are deduped by the sets below
                    self.links_up[nid].send(
                        {"type": "notif_ack", "nonce": msg["nonce"],
                         "session": sid}, self.nodes[nid].deliver)
                if mtype == "first_sample" and nid in pending_first:
                    est = skew[nid]
                    t_coord = msg["t_local"] - est["theta"]
                    node_rec(nid)["first_sample"] = {
                        "t_local": msg["t_local"], "t_coord": t_coord,
                        "err": est["error"],
                        "offset_from_t_start": t_coord - t_start}
                    ev("first_sample", node=nid, t_coord=t_coord,
                       err=est["error"],
                       offset_from_t_start=t_coord - t_start)
                    pending_first.discard(nid)
                    first_seen.add(nid)
                elif mtype == "done":
                    node_rec(nid)["stats"] = msg.get("stats")
                    ev("done", node=nid, stats=msg.get("stats"))
                    pending_done.discard(nid)
                elif mtype == "error":
                    node_rec(nid)["error"] = msg.get("reason")
                    ev("node_error", node=nid, stage=msg.get("stage"),
                       reason=msg.get("reason"))
                    pending_first.discard(nid)
                    pending_done.discard(nid)

            ok_nodes = [n for n in active
                        if n in first_seen and not node_rec(n)["error"]]
            if len(ok_nodes) == len(selected):
                result = "ok"
            elif ok_nodes:
                result = "partial"
            else:
                result = "failed"
            return self._finish(record, sid, result, t0, stale_mark)
        finally:
            self._active_queue = None
            self._active_session = None

    def _finish(self, record, sid, result, t0, stale_mark):
        record["result"] = result
        record["finished_dt"] = time.monotonic() - t0
        record["stale_ignored"] = self.stale[stale_mark:]
        for nid, nr in record["nodes"].items():
            f = nr.get("file")
            nr["file_exists"] = bool(f and os.path.exists(f))
            if f:
                nr["file_rel"] = os.path.relpath(f, self.recordings_dir)
        path = os.path.join(self.recordings_dir, sid, "record.json")
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)
        record["record_path"] = path
        return record
