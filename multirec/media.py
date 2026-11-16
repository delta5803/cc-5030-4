"""Real media writers and verifiers.

Camera nodes write H.264 MP4 via the imageio-ffmpeg bundled binary; every
frame carries its index burned in twice: a machine-readable binary strip
(magic prefix + 32 bits) and large digits. Mic nodes write 16-bit PCM WAV
with a pulse train whose first pulse starts at sample 0 of the recording.
"""
import math
import shutil
import struct
import subprocess
import wave

import numpy as np

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# 3x5 digit font, rows top->bottom, 1 = lit
_FONT = {
    "0": (0b111, 0b101, 0b101, 0b101, 0b111),
    "1": (0b010, 0b110, 0b010, 0b010, 0b111),
    "2": (0b111, 0b001, 0b111, 0b100, 0b111),
    "3": (0b111, 0b001, 0b111, 0b001, 0b111),
    "4": (0b101, 0b101, 0b111, 0b001, 0b001),
    "5": (0b111, 0b100, 0b111, 0b001, 0b111),
    "6": (0b111, 0b100, 0b111, 0b101, 0b111),
    "7": (0b111, 0b001, 0b010, 0b010, 0b010),
    "8": (0b111, 0b101, 0b111, 0b101, 0b111),
    "9": (0b111, 0b101, 0b111, 0b001, 0b111),
}

STRIP_MAGIC = (1, 0, 1, 1, 0, 1, 0, 0)  # 8-cell prefix locating the strip
STRIP_BITS = 32
CELL = 8  # px per strip cell
STRIP_CELLS = len(STRIP_MAGIC) + STRIP_BITS


def _draw_digit(img, digit, x, y, scale, color=(255, 255, 255)):
    rows = _FONT[digit]
    for r in range(5):
        bits = rows[r]
        for c in range(3):
            if bits >> (2 - c) & 1:
                img[y + r * scale: y + (r + 1) * scale,
                    x + c * scale: x + (c + 1) * scale] = color


def draw_number(img, value, x, y, scale=6, color=(255, 255, 255)):
    for ch in str(value):
        _draw_digit(img, ch, x, y, scale, color)
        x += 4 * scale


def make_frame(index, width, height, bg_color):
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = bg_color
    # binary strip: magic prefix + frame index bits, MSB first
    bits = list(STRIP_MAGIC) + [(index >> (STRIP_BITS - 1 - i)) & 1
                                for i in range(STRIP_BITS)]
    for i, b in enumerate(bits):
        v = 255 if b else 0
        img[0:CELL, i * CELL:(i + 1) * CELL] = v
    # big frame number, human readable
    draw_number(img, index, 12, 28, scale=7)
    # moving block so motion is visible
    bx = (index * 7) % (width - 30)
    img[height - 40:height - 10, bx:bx + 30] = (255, 255, 255)
    return img


def read_strip_index(frame):
    bits = []
    q = CELL // 4  # sample cell centres: robust to codec edge blur
    for i in range(STRIP_CELLS):
        cell = frame[q:CELL - q, i * CELL + q:(i + 1) * CELL - q]
        bits.append(1 if cell.mean() > 127 else 0)
    if tuple(bits[:len(STRIP_MAGIC)]) != STRIP_MAGIC:
        raise ValueError("strip magic not found")
    idx = 0
    for b in bits[len(STRIP_MAGIC):]:
        idx = (idx << 1) | b
    return idx


class Mp4Writer:
    def __init__(self, path, width, height, fps):
        cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def decode_frames(path):
    """Yield RGB frames of an mp4 as numpy arrays."""
    probe = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path],
        capture_output=True, text=True).stderr
    w = h = None
    for line in probe.splitlines():
        if "Video:" in line:
            for tok in line.replace(",", " ").split():
                if "x" in tok and tok.split("x")[0].isdigit():
                    a, b = tok.split("x")[:2]
                    if b.isdigit():
                        w, h = int(a), int(b)
                        break
        if w:
            break
    if not w:
        raise RuntimeError(f"cannot probe size of {path}: {probe}")
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], dtype=np.uint8)
    return arr.reshape(n, h, w, 3)


def read_video_frame_indices(path):
    return [read_strip_index(f) for f in decode_frames(path)]


# ---------------- audio ----------------

AUDIO_RATE = 16000
PULSE_EVERY = 0.5
PULSE_LEN = 0.04


def pulse_track(duration, freq, rate=AUDIO_RATE):
    """PCM float samples: pulses of `freq` starting at t=0, every PULSE_EVERY."""
    n = int(duration * rate)
    out = np.zeros(n, dtype=np.float32)
    plen = int(PULSE_LEN * rate)
    t = np.arange(plen) / rate
    tone = 0.6 * np.sin(2 * math.pi * freq * t)
    edge = int(0.002 * rate)
    tone[:edge] *= np.linspace(0, 1, edge)
    tone[-edge:] *= np.linspace(1, 0, edge)
    pos = 0
    while pos + plen <= n:
        out[pos:pos + plen] += tone
        pos += int(PULSE_EVERY * rate)
    return out


class WavWriter:
    def __init__(self, path, rate=AUDIO_RATE):
        self._w = wave.open(path, "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(rate)

    def write(self, samples_float):
        pcm = (np.clip(samples_float, -1, 1) * 32767).astype(np.int16)
        self._w.writeframes(pcm.tobytes())

    def close(self):
        self._w.close()


def read_wav(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        data = w.readframes(n)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767
    return samples, rate


def detect_pulse_onsets(path, threshold=0.15):
    samples, rate = read_wav(path)
    env = np.abs(samples)
    win = max(1, int(0.005 * rate))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    active = env > threshold
    onsets = []
    i = 0
    while i < len(active):
        if active[i]:
            onsets.append(i / rate)
            i += int(0.2 * rate)  # skip past this pulse
        else:
            i += 1
    return onsets, rate
