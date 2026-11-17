"""Independent verification of a finished session record against the media
files on disk.

Every expectation is recomputed here from the files themselves and from the
node ground truth (configured colour/tone/true clock offset). The record
only claims; this module measures. The same computation backs the barrier
acceptance tests and the production panel's verify endpoint, so a bad
result cannot hide behind defaults in either place.
"""
import os

import numpy as np

from . import media

START_SLACK = 0.08   # scheduling slack beyond the retained skew error bound
EST_SLACK = 0.02     # numeric slack vs the ground-truth clock reference
COLOR_TOL = 25       # per-channel tolerance after the lossy codec round-trip
FREQ_TOL = 30.0      # Hz; a 40 ms pulse gives 25 Hz FFT bins


def dominant_freq(samples, rate):
    n = len(samples)
    if n == 0:
        return 0.0
    sp = np.abs(np.fft.rfft(samples * np.hanning(n)))
    return float(np.argmax(sp)) * rate / n


def _check(out, name, ok, detail):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        out["errors"].append(f"{name}: {detail}")
    return ok


def _resolve_path(record_file, base_dir):
    """The record stores absolute paths; a committed sample may have moved
    with the repo, so fall back to the record's own directory."""
    if record_file and os.path.exists(record_file):
        return record_file
    if base_dir and record_file:
        cand = os.path.join(base_dir, os.path.basename(record_file))
        if os.path.exists(cand):
            return cand
    return None


def _verify_video(out, path, nr, cfg):
    try:
        frames = media.decode_frames(path)
    except Exception as exc:
        _check(out, "decode", False, f"{type(exc).__name__}: {exc}")
        return
    _check(out, "decode", True, f"{len(frames)} frames decoded")
    try:
        idx = [media.read_strip_index(f) for f in frames]
    except Exception as exc:
        _check(out, "frame_identity", False,
               f"strip unreadable: {type(exc).__name__}: {exc}")
        return
    contiguous = idx == list(range(len(idx)))
    _check(out, "frame_identity",
           contiguous and bool(idx) and idx[0] == 0,
           f"first frame #{idx[0] if idx else '-'}, {len(idx)} frames, "
           f"{'contiguous 0..n-1' if contiguous else 'GAPS'}")
    stats = nr.get("stats") or {}
    if stats.get("frames") is not None:
        _check(out, "frame_count", len(idx) == stats["frames"],
               f"{len(idx)} decoded vs stats {stats['frames']}")
    if cfg.get("color") is not None and len(frames):
        med = np.median(frames[0].reshape(-1, 3), axis=0)
        exp = np.asarray(cfg["color"], dtype=float)
        dev = float(np.abs(med - exp).max())
        _check(out, "color_identity", dev <= COLOR_TOL,
               f"median RGB {med.astype(int).tolist()} vs configured "
               f"{[int(v) for v in exp]} (dev {dev:.0f})")


def _verify_audio(out, path, nr, cfg):
    try:
        samples, rate = media.read_wav(path)
        onsets, _ = media.detect_pulse_onsets(path)
    except Exception as exc:
        _check(out, "decode", False, f"{type(exc).__name__}: {exc}")
        return
    _check(out, "decode", True, f"{len(samples)} samples @ {rate}Hz")
    _check(out, "first_pulse_at_zero",
           bool(onsets) and onsets[0] < 0.02,
           (f"first pulse onset {onsets[0] * 1000:.1f}ms"
            if onsets else "no pulses found"))
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]
    spacing_ok = len(onsets) >= 2 and all(
        abs(g - media.PULSE_EVERY) < 0.02 for g in gaps)
    _check(out, "pulse_spacing", spacing_ok,
           f"{len(onsets)} pulses, gaps {[round(g, 3) for g in gaps]}")
    stats = nr.get("stats") or {}
    if stats.get("samples") is not None:
        _check(out, "sample_count", len(samples) == stats["samples"],
               f"{len(samples)} vs stats {stats['samples']}")
    if cfg.get("tone") is not None and onsets:
        plen = int(media.PULSE_LEN * rate)
        f0 = dominant_freq(samples[:plen], rate)
        _check(out, "tone_identity", abs(f0 - cfg["tone"]) <= FREQ_TOL,
               f"first pulse {f0:.1f}Hz vs configured {cfg['tone']}Hz")


def verify_session(record, nodes_cfg=None, base_dir=None):
    """Recompute every checkable claim of a session record.

    record: parsed record.json dict. nodes_cfg: optional
    {nid: {"color": (r,g,b), "tone": hz, "true_offset": s}} ground truth.
    Content problems are reported, never raised.
    """
    nodes_cfg = nodes_cfg or {}
    result = {"ok": True, "session_id": record.get("session_id"),
              "result": record.get("result"), "members": [],
              "stray_files": [], "nodes": {}}
    t_start = record.get("t_start_coord")
    for nid, nr in record.get("nodes", {}).items():
        out = {"checks": {}, "errors": []}
        cfg = nodes_cfg.get(nid, {})
        true_offset = cfg.get("true_offset", nr.get("true_offset"))
        fs = nr.get("first_sample")
        err = (fs or {}).get("err", 0.0)
        if nr.get("error"):
            out["node_error"] = nr["error"]
            if nr.get("file") and _resolve_path(nr["file"], base_dir):
                # a file on disk does NOT make the node a member
                result["stray_files"].append(nid)
        # real recording start, derived from the ground-truth clock
        if fs is not None and true_offset is not None:
            indep = fs["t_local"] - true_offset
            dev = abs(fs["t_coord"] - indep)
            _check(out, "start_from_ground_truth",
                   dev <= err + EST_SLACK,
                   f"|t_coord-(t_local-true_offset)|={dev * 1000:.1f}ms "
                   f"(bound {(err + EST_SLACK) * 1000:.1f}ms)")
            if t_start is not None:
                off = abs(fs["t_coord"] - t_start)
                _check(out, "aligned_to_barrier",
                       off <= err + START_SLACK,
                       f"first sample {off * 1000:.1f}ms from T_start "
                       f"(bound {(err + START_SLACK) * 1000:.1f}ms)")
        # real clip: decode and measure the actual file
        if nr.get("file") and not nr.get("error"):
            path = _resolve_path(nr["file"], base_dir)
            if path is None:
                _check(out, "file_present", False, f"missing: {nr['file']}")
            elif nr.get("kind") == "camera":
                _verify_video(out, path, nr, cfg)
            elif nr.get("kind") == "mic":
                _verify_audio(out, path, nr, cfg)
        out["ok"] = not out["errors"]
        if not out["ok"]:
            result["ok"] = False
        # membership comes from recorded evidence only: a first real sample
        # and no error. File presence alone never grants membership.
        if fs is not None and not nr.get("error"):
            result["members"].append(nid)
        result["nodes"][nid] = out
    result["members"].sort()
    return result
