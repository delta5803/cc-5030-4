"""End-to-end tests of the multi-end recording kernel.

Scenarios required by the spec: varied startup delays, one node failing,
response loss. Generated media files are decoded and checked: video frame
numbers (binary strip burned into every frame) and audio pulse onsets.
"""
import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multirec.coordinator import Coordinator, DEFAULT_NODES
from multirec import media

ALL_IDS = [c["id"] for c in DEFAULT_NODES]
CAMS = [c["id"] for c in DEFAULT_NODES if c["kind"] == "camera"]
MICS = [c["id"] for c in DEFAULT_NODES if c["kind"] == "mic"]
SCHED_SLACK = 0.08  # timer/scheduling slack beyond the retained error bound


def make_kernel(tmp_path):
    k = Coordinator(recordings_dir=str(tmp_path / "rec"))
    for cfg in DEFAULT_NODES:
        k.add_node(cfg)
    return k


def evs(rec, nid, etype):
    return [e for e in rec["events"] if e["node"] == nid and e["type"] == etype]


def session_media_files(rec):
    d = os.path.dirname(rec["record_path"])
    return sorted(glob.glob(os.path.join(d, "*.mp4"))
                  + glob.glob(os.path.join(d, "*.wav")))


# ---------------------------------------------------------------- aligned start

def test_aligned_start_with_varied_startup_delays(tmp_path):
    """Nodes with startup delays 0.10-0.45s must all become ready BEFORE the
    common start boundary; first real samples land at T_start within the
    retained skew-estimation error."""
    k = make_kernel(tmp_path)
    rec = k.run_session(ALL_IDS, duration=1.5, policy="cancel")
    assert rec["result"] == "ok", rec["result"]

    t_start = rec["t_start_coord"]
    for nid in ALL_IDS:
        nr = rec["nodes"][nid]
        # three distinct events, in causal order
        sent = evs(rec, nid, "start_command_sent")[0]
        ack = evs(rec, nid, "start_ack")[0]
        first = evs(rec, nid, "first_sample")[0]
        assert sent["t"] < ack["t"] < first["t"]
        # command/ack happen BEFORE the boundary; first sample AT it
        assert sent["t"] < t_start and ack["t"] < t_start
        # skew estimate is honest: true offset inside the retained bound
        sk = nr["skew"]
        assert abs(sk["theta"] - nr["true_offset"]) <= sk["error"] + 0.01
        # actual first sample at T_start within retained error + slack
        assert abs(first["t_coord"] - t_start) <= sk["error"] + SCHED_SLACK

    # every node produced exactly one real media file
    files = session_media_files(rec)
    assert len(files) == len(ALL_IDS)

    # video: decode and verify burned-in frame numbers 0..n-1 contiguous
    for nid in CAMS:
        path = os.path.join(os.path.dirname(rec["record_path"]), nid + ".mp4")
        idxs = media.read_video_frame_indices(path)
        assert idxs == list(range(len(idxs))), nid
        assert len(idxs) >= int(1.5 * 24) - 2

    # audio: first pulse at sample 0 of the recording, then every 0.5s
    for nid in MICS:
        path = os.path.join(os.path.dirname(rec["record_path"]), nid + ".wav")
        onsets, rate = media.detect_pulse_onsets(path)
        assert len(onsets) >= 3, (nid, onsets)
        assert onsets[0] < 0.02
        for a, b in zip(onsets, onsets[1:]):
            assert abs((b - a) - media.PULSE_EVERY) < 0.02


# ---------------------------------------------------------------- node failure

def test_node_failure_cancel_policy(tmp_path):
    k = make_kernel(tmp_path)
    k.nodes["cam2"].fail_at = "prepare"
    rec = k.run_session(ALL_IDS, duration=1.0, policy="cancel")
    assert rec["result"] == "cancelled"
    assert "resource unavailable" in rec["nodes"]["cam2"]["error"]
    # explicit cancel: no media files at all
    assert session_media_files(rec) == []
    assert evs(rec, None, "policy_cancel")


def test_node_failure_partial_policy(tmp_path):
    k = make_kernel(tmp_path)
    k.nodes["cam2"].fail_at = "prepare"
    rec = k.run_session(ALL_IDS, duration=1.0, policy="partial")
    assert rec["result"] == "partial"
    assert rec["nodes"]["cam2"]["error"]
    files = {os.path.basename(f) for f in session_media_files(rec)}
    assert files == {"cam1.mp4", "mic1.wav", "mic2.wav"}
    for nid in ("cam1", "mic1", "mic2"):
        assert evs(rec, nid, "first_sample"), nid


def test_prepare_timeout(tmp_path):
    """A node still warming up when the prepare barrier expires is handled by
    policy; its late ready reply becomes a recorded stale message."""
    k = make_kernel(tmp_path)
    k.nodes["mic2"].startup_delay = 1.2
    rec = k.run_session(ALL_IDS, duration=1.0, policy="partial",
                        prepare_timeout=0.5)
    assert rec["result"] == "partial"
    assert rec["nodes"]["mic2"]["error"] == "prepare timeout"
    files = {os.path.basename(f) for f in session_media_files(rec)}
    assert "mic2.wav" not in files
    # the late ready from mic2 must show up as stale, not satisfy anything
    stale_types = [s["msg"].get("type") for s in rec["stale_ignored"]]
    assert "ready" in stale_types


# ------------------------------------------------------- loss / idempotency

def test_response_loss_never_opens_second_file(tmp_path):
    """Drop the first start_ack from mic1. The coordinator retries with the
    SAME session id; the node must confirm idempotently and open exactly one
    file."""
    k = make_kernel(tmp_path)
    dropped = {"n": 0}

    def drop_first_ack(msg):
        if msg.get("type") == "start_ack" and dropped["n"] == 0:
            dropped["n"] += 1
            return True
        return False

    k.links_down["mic1"].drop = drop_first_ack
    rec = k.run_session(ALL_IDS, duration=1.0, policy="cancel")
    assert rec["result"] == "ok"
    assert dropped["n"] == 1
    assert evs(rec, "mic1", "start_ack_lost_retry")
    ack = evs(rec, "mic1", "start_ack")[0]
    assert ack["already"] is True  # duplicate start did not re-arm the device
    d = os.path.dirname(rec["record_path"])
    assert len(glob.glob(os.path.join(d, "mic1*.wav"))) == 1
    assert len(session_media_files(rec)) == len(ALL_IDS)


def test_stale_ack_cannot_satisfy_new_barrier(tmp_path):
    """An ack from an old session carries an old nonce; injected while a new
    session's barrier is live, it is recorded as stale and cannot satisfy
    the new barrier."""
    import threading
    import time

    k = make_kernel(tmp_path)
    rec1 = k.run_session(ALL_IDS, duration=1.0, policy="cancel")
    assert rec1["result"] == "ok"
    max_nonce_s1 = k._nonce

    holder = {}
    t = threading.Thread(
        target=lambda: holder.update(
            rec=k.run_session(ALL_IDS, duration=1.0, policy="cancel")),
        daemon=True)
    t.start()
    # wait until session 2's prepare barrier is live
    sid2 = None
    while sid2 is None:
        for sid, r in k.sessions.items():
            if (r.get("result") == "running"
                    and any(e["type"] == "prepare_sent" for e in r["events"])):
                sid2 = sid
        time.sleep(0.005)
    # old acks arrive late, in the middle of the new barrier
    k._dispatch({"type": "ready", "session": rec1["session_id"],
                 "epoch": rec1["epoch"], "nonce": 3, "node": "cam1"})
    k._dispatch({"type": "start_ack", "session": rec1["session_id"],
                 "epoch": rec1["epoch"], "nonce": 7, "node": "cam1"})
    t.join()
    rec2 = holder["rec"]

    assert rec2["result"] == "ok"           # barrier satisfied by real replies
    assert k._nonce > max_nonce_s1          # fresh nonces for the new barrier
    stale_sessions = {s["msg"].get("session") for s in rec2["stale_ignored"]}
    assert rec1["session_id"] in stale_sessions
    for nid in ALL_IDS:
        assert evs(rec2, nid, "start_ack")
