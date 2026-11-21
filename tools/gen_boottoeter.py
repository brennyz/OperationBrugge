#!/usr/bin/env python3
"""Build audio/boottoeter.mp3: an original boat horn, a short fanfare, then "I'm on a boat".

The voice is espeak-ng (open source), not a recording of any song.
The fanfare is a plain rising major arpeggio, written for this button.
The script prints VOICE_AT: the second the spoken phrase starts. index.html
uses that number so the on-screen line lands with the voice.

Requires: python3, numpy, espeak-ng, ffmpeg with libmp3lame.
Run from anywhere: python3 tools/gen_boottoeter.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audio" / "boottoeter.mp3"
SR = 22050

def env_adsr(n: int, attack: float, release: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    dur = n / sr
    att = 1 - np.exp(-t / max(attack, 1e-4))
    rel = np.clip((dur - t) / max(release, 1e-4), 0, 1)
    rel = rel * rel
    return (att * rel).astype(np.float64)


def lowpass(x: np.ndarray, coeff: float) -> np.ndarray:
    y = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s += coeff * (float(v) - s)
        y[i] = s
    return y


def airhorn(dur: float, f_low: float, f_high: float, seed: int) -> np.ndarray:
    """Dual-tone marine air horn: pressure glide, harmonics, a bit of air."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    glide = 0.78 + 0.22 * (1 - np.exp(-t / 0.05))

    def stack(freq: float) -> np.ndarray:
        phase = 2 * math.pi * freq * np.cumsum(glide) / SR
        y = (
            np.sin(phase) * 1.0
            + np.sin(2 * phase) * 0.62
            + np.sin(3 * phase) * 0.34
            + np.sin(4 * phase) * 0.16
            + np.sin(5 * phase) * 0.08
        )
        return y

    y = stack(f_low) * 0.72 + stack(f_high) * 0.58
    y = np.tanh(y * 1.25)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n)
    band = lowpass(noise, 0.22) - lowpass(noise, 0.05)
    y = y * 0.84 + band * 0.22
    y *= env_adsr(n, 0.03, 0.11)
    return np.tanh(y * 1.15) * 0.9


def bell(freq: float, dur: float = 0.42) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    mod = np.sin(2 * math.pi * freq * 1.4 * t) * np.exp(-t * 9)
    y = np.sin(2 * math.pi * freq * t + 1.4 * mod)
    y += 0.35 * np.sin(2 * math.pi * freq * 2.01 * t) * np.exp(-t * 7)
    y *= np.exp(-t * 5.5)
    return y * 0.55


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"Unexpected wav format from espeak: {path}")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0


def synth_voice(tmp: Path) -> np.ndarray:
    # One phrase, no breaks: a break makes espeak say the letter "A"
    # instead of the article. Pitch still rises across the four words.
    ssml = (
        "<speak>"
        '<prosody pitch="-2st" rate="86%">I\'m</prosody> '
        '<prosody pitch="+3st" rate="88%">on</prosody> '
        '<prosody pitch="+6st" rate="92%">a</prosody> '
        '<prosody pitch="+9st" rate="70%">boat</prosody>'
        "</speak>"
    )
    phon = subprocess.run(
        ["espeak-ng", "-v", "en-us", "-m", "-q", "-x", ssml],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # a# is the article "a". The letter name would be eI, which is the wrong word.
    if "a#" not in phon.split() and "a#" not in phon:
        raise SystemExit(f"voice would not say the article 'a': {phon!r}")
    wav = tmp / "voice.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-m", "-a", "200", "-w", str(wav), ssml],
        check=True,
    )
    voice = read_wav(wav)
    # Trim espeak's leading and trailing silence so the phrase starts on time.
    amp = np.abs(voice)
    hot = np.where(amp > 0.02)[0]
    if len(hot):
        a = max(0, int(hot[0]) - int(0.02 * SR))
        b = min(len(voice), int(hot[-1]) + int(0.08 * SR))
        voice = voice[a:b]
    peak = np.max(np.abs(voice)) or 1
    return voice / peak * 0.95


def place(mix: np.ndarray, clip: np.ndarray, at: float, gain: float = 1.0) -> None:
    start = int(round(at * SR))
    end = start + len(clip)
    if end > len(mix):
        raise SystemExit("mix buffer too short")
    mix[start:end] += clip * gain


def build() -> tuple[np.ndarray, float]:
    voice = synth_voice(Path("/tmp"))
    # Long blast, short blast, three-note fanfare, then the phrase.
    cursor = 0.0
    events: list[tuple[float, np.ndarray, float]] = []

    blast1 = airhorn(0.72, 311.13, 415.30, seed=7)
    events.append((cursor, blast1, 0.95))
    cursor += 0.72 + 0.09

    blast2 = airhorn(0.38, 329.63, 440.00, seed=11)
    events.append((cursor, blast2, 0.9))
    cursor += 0.38 + 0.14

    fanfare = [523.25, 659.25, 783.99, 1046.50]
    for i, freq in enumerate(fanfare):
        events.append((cursor, bell(freq, 0.48), 0.7 if i < 3 else 0.85))
        cursor += 0.13
    cursor += 0.16
    voice_at = cursor
    events.append((voice_at, voice, 1.0))
    # A quiet bell under "boat" so the phrase sits in the jingle.
    events.append((voice_at + 0.55, bell(1046.50, 0.7), 0.28))
    end = voice_at + len(voice) / SR + 0.18
    # Short answering toot, like the horn signing off.
    toot_at = voice_at + len(voice) / SR + 0.06
    toot = airhorn(0.28, 349.23, 466.16, seed=3)
    events.append((toot_at, toot, 0.72))
    end = max(end, toot_at + len(toot) / SR + 0.05)

    mix = np.zeros(int(end * SR) + 8, dtype=np.float64)
    for at, clip, gain in events:
        place(mix, clip, at, gain)
    peak = np.max(np.abs(mix)) or 1
    mix = mix / peak * 0.89
    return mix, voice_at


def write_mp3(samples: np.ndarray, path: Path) -> None:
    wav_path = path.with_suffix(".wav")
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "4",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
    )
    wav_path.unlink()


def main() -> None:
    if not shutil.which("espeak-ng") or not shutil.which("ffmpeg"):
        raise SystemExit("espeak-ng and ffmpeg are required")
    samples, voice_at = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_mp3(samples, OUT)
    seconds = len(samples) / SR
    print(f"wrote {OUT} ({seconds:.2f}s)")
    print(f"VOICE_AT={voice_at:.3f}")


if __name__ == "__main__":
    main()
