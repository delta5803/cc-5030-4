"""Virtual acquisition node (camera or mic).

Runs its own thread and its own clock (monotonic base + hidden true offset).
Writes real media files. Start commands are idempotent per session id: a
duplicate start for the active session is acked as `already` and never opens
a second file.
"""
import queue
import threading
import time

from . import media


class Node:
    def __init__(self, node_id, kind, true_offset, startup_delay,
                 link_up, link_down, coord_deliver, fail_at=None,
                 color=(40, 60, 120), tone=880.0, fps=24, size=(320, 240)):
        assert kind in ("camera", "mic")
        self.id = node_id
        self.kind = kind
        self.true_offset = true_offset
        self.startup_delay = startup_delay
        self.link_up = link_up      # coordinator -> node
        self.link_down = link_down  # node -> coordinator
        self.coord_deliver = coord_deliver
        self.fail_at = fail_at      # None | "prepare" | "start"
        self.color = color
        self.tone = tone
        self.fps = fps
        self.size = size

        self.state = "idle"
        self.active_session = None
        self._done_sessions = set()  # completed: duplicate starts stay no-ops
        self._pending = {}
        self._nonce = 0
        self._inbox = queue.Queue()
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"node-{node_id}")
        self._thread.start()

    # node-local clock: this is the only clock the node can read
    def now(self):
        return time.monotonic() + self.true_offset

    def deliver(self, msg):
        self._inbox.put(msg)

    def _reply(self, msg):
        msg["node"] = self.id
        self.link_down.send(msg, self.coord_deliver)

    def _notify(self, msg, tries=3, timeout=0.4):
        """Reliable unsolicited notification: retransmit until the
        coordinator's notif_ack arrives. Duplicates are harmless because
        the coordinator dedupes by message type per node."""
        self._nonce += 1
        nonce = self._nonce
        acked = threading.Event()
        self._pending[nonce] = acked
        m = dict(msg, nonce=nonce)
        try:
            for _ in range(tries):
                self._reply(dict(m))
                if acked.wait(timeout):
                    break
        finally:
            self._pending.pop(nonce, None)

    def _run(self):
        while True:
            msg = self._inbox.get()
            mtype = msg.get("type")
            if mtype == "ping":
                t1 = self.now()
                self._reply({"type": "pong", "nonce": msg["nonce"],
                             "t1": t1, "t2": self.now()})
            elif mtype == "prepare":
                self._on_prepare(msg)
            elif mtype == "start":
                self._on_start(msg)
            elif mtype == "cancel":
                if msg.get("session") == self.active_session:
                    self._cancel.set()
                self.state = "idle" if self.active_session is None else self.state
            elif mtype == "notif_ack":
                ev = self._pending.pop(msg.get("nonce"), None)
                if ev is not None:
                    ev.set()

    def _on_prepare(self, msg):
        sid = msg["session"]
        if self.fail_at == "prepare":
            self.state = "failed"
            self._reply({"type": "error", "session": sid, "epoch": msg["epoch"],
                         "nonce": msg["nonce"], "stage": "prepare",
                         "reason": "resource unavailable (injected)"})
            return
        # simulate device resource acquisition / warm-up
        time.sleep(self.startup_delay)
        self.state = "ready"
        self._reply({"type": "ready", "session": sid, "epoch": msg["epoch"],
                     "nonce": msg["nonce"], "t_local": self.now()})

    def _on_start(self, msg):
        sid = msg["session"]
        if self.active_session == sid or sid in self._done_sessions:
            # duplicate start (retry or stale repeat): never open a second file
            self._reply({"type": "start_ack", "session": sid,
                         "epoch": msg["epoch"], "nonce": msg["nonce"],
                         "already": True, "t_local": self.now()})
            return
        if self.active_session is not None:
            self._reply({"type": "error", "session": sid,
                         "epoch": msg["epoch"], "nonce": msg["nonce"],
                         "stage": "start", "reason": "node busy"})
            return
        if self.fail_at == "start":
            self.state = "failed"
            self._reply({"type": "error", "session": sid,
                         "epoch": msg["epoch"], "nonce": msg["nonce"],
                         "stage": "start",
                         "reason": "sensor failed to arm (injected)"})
            return
        self.active_session = sid
        self._cancel.clear()
        self.state = "starting"
        self._reply({"type": "start_ack", "session": sid,
                     "epoch": msg["epoch"], "nonce": msg["nonce"],
                     "already": False, "t_local": self.now()})
        worker = threading.Thread(target=self._record, args=(msg,),
                                  daemon=True, name=f"rec-{self.id}")
        worker.start()

    def _record(self, msg):
        sid = msg["session"]
        writer = None
        try:
            # arm the encoding pipeline BEFORE the barrier, like a real device
            duration = msg["duration"]
            if self.kind == "camera":
                w, h = self.size
                writer = media.Mp4Writer(msg["out"], w, h, self.fps)
                media.make_frame(0, w, h, self.color)  # warm the render path
            else:
                writer = media.WavWriter(msg["out"])
            # wait until the barrier instant on the node-local clock
            while True:
                remain = msg["start_at"] - self.now()
                if remain <= 0:
                    break
                if remain > 0.1:
                    if self._cancel.wait(timeout=0.05):
                        self._finish(sid, cancelled=True)
                        return
                else:
                    time.sleep(remain)  # precise final approach to T_start
                    break
            self.state = "recording"
            if self.kind == "camera":
                self._record_video(writer, msg, duration)
            else:
                self._record_audio(writer, msg, duration)
            if self._cancel.is_set():
                self._finish(sid, cancelled=True)
                return
            self.state = "done"
            self._done_sessions.add(sid)
            self._notify({"type": "done", "session": sid, "stats": self._stats})
        except Exception as exc:  # real device failure path
            self.state = "failed"
            self._reply({"type": "error", "session": sid, "stage": "record",
                         "reason": f"{type(exc).__name__}: {exc}"})
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            self.active_session = None

    def _record_video(self, writer, msg, duration):
        w, h = self.size
        n_frames = int(round(duration * self.fps))
        t0 = self.now()
        first_stamped = False
        i = -1
        for i in range(n_frames):
            if self._cancel.is_set():
                break
            frame = media.make_frame(i, w, h, self.color)
            writer.write(frame)
            if not first_stamped:
                first_stamped = True
                self._notify({"type": "first_sample", "session": msg["session"],
                             "t_local": self.now(), "path": msg["out"]})
            # pace frames like a real sensor clock
            delay = t0 + (i + 1) / self.fps - self.now()
            if delay > 0:
                self._cancel.wait(timeout=delay)
        self._stats = {"frames": i + 1, "fps": self.fps}

    def _record_audio(self, writer, msg, duration):
        track = media.pulse_track(duration, self.tone)
        t0 = self.now()
        chunk = int(0.25 * media.AUDIO_RATE)
        pos = 0
        while pos < len(track):
            if self._cancel.is_set():
                break
            writer.write(track[pos:pos + chunk])
            if pos == 0:
                # first real samples hit the file now
                self._notify({"type": "first_sample",
                             "session": msg["session"],
                             "t_local": self.now(), "path": msg["out"]})
            pos += chunk
            # pace writes like a real audio interface clock
            delay = t0 + pos / media.AUDIO_RATE - self.now()
            if delay > 0:
                self._cancel.wait(timeout=delay)
        self._stats = {"samples": int(duration * media.AUDIO_RATE),
                       "rate": media.AUDIO_RATE, "tone": self.tone}

    def _finish(self, sid, cancelled=False):
        self.state = "cancelled" if cancelled else self.state
        self.active_session = None
        if cancelled:
            self._notify({"type": "cancelled", "session": sid})

    def shutdown(self):
        self._cancel.set()
        self.link_up.cancel_pending()
        self.link_down.cancel_pending()
