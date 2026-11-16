"""Seeded simulated link: reproducible latency and message loss.

Delays are drawn from a seeded RNG so a whole session (including clock-skew
estimation) is reproducible for a given seed. Delivery happens on real timer
threads, so timestamps taken by sender/receiver are real monotonic readings.
"""
import random
import threading


class Link:
    def __init__(self, seed, min_latency=0.004, max_latency=0.025, loss=0.0,
                 drop=None, name="link"):
        self.rng = random.Random(seed)
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.loss = loss
        self.drop = drop  # optional callable(msg) -> bool, for scripted loss
        self.name = name
        self.sent = 0
        self.dropped = 0
        self._timers = []
        self._lock = threading.Lock()

    def send(self, msg, deliver):
        self.sent += 1
        if self.drop is not None and self.drop(msg):
            self.dropped += 1
            return
        if self.rng.random() < self.loss:
            self.dropped += 1
            return
        d = self.rng.uniform(self.min_latency, self.max_latency)
        t = threading.Timer(d, deliver, args=(dict(msg),))
        t.daemon = True
        with self._lock:
            self._timers.append(t)
        t.start()

    def cancel_pending(self):
        with self._lock:
            for t in self._timers:
                t.cancel()
            self._timers.clear()
