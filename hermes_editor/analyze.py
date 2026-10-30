"""Look at the footage before deciding anything.

For every clip: duration and frame size, where the speech is (ffmpeg silencedetect),
what is said and when (Whisper word timestamps), how loud and how much motion
there is over time (every frame, via ebur128 and signalstats), exposure and colour
of the frame, and whether a take restarts itself or repeats another clip.
The result is an inventory: plain JSON the brain reads and the renderer trusts.
"""
from __future__ import annotations

import difflib
import json
import re
import statistics
import subprocess
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
SILENCE_DB = -32
SILENCE_MIN = 0.35


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def list_clips(folder: Path) -> list[Path]:
    clips = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXT and p.is_file()]
    if not clips:
        raise SystemExit(f"no video clips in {folder}")
    return clips


def probe(path: Path) -> dict:
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    info = json.loads(r.stdout or "{}")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if v is None:
        raise SystemExit(f"{path.name}: no video stream")
    w, h = int(v["width"]), int(v["height"])
    rot = int((v.get("tags") or {}).get("rotate", 0) or 0)
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    tags = info.get("format", {}).get("tags", {}) or {}
    return {
        "duration": float(info["format"]["duration"]),
        "width": w, "height": h,
        "has_audio": a is not None,
        "created": tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate") or "",
    }


def audio_scan(path: Path) -> tuple[list[list[float]], list[list[float]]]:
    """Silences and a momentary-loudness curve (LUFS, 10 Hz) in one decode."""
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
             f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN},ebur128=metadata=1,"
             "ametadata=print:key=lavfi.r128.M", "-f", "null", "-"])
    err = r.stderr
    sil, start = [], None
    for m in re.finditer(r"silence_(start|end): (-?[\d.]+)", err):
        t = max(0.0, float(m.group(2)))
        if m.group(1) == "start":
            start = t
        elif start is not None:
            sil.append([round(start, 3), round(t, 3)])
            start = None
    if start is not None:
        sil.append([round(start, 3), None])  # runs to the end; closed by the caller
    curve, t = [], None
    for line in err.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"lavfi\.r128\.M=(-?[\d.]+|-inf)", line)
        if m and t is not None:
            v = float(m.group(1)) if m.group(1) != "-inf" else -70.0
            curve.append([round(t, 2), max(v, -70.0)])
    return sil, curve


def video_scan(path: Path, width: int, height: int) -> dict:
    """Every frame is decoded once at low resolution. Per-frame motion (mean absolute
    luma difference) gives the action curve; a frame every second gives exposure
    and colour, estimated three ways (whole frame, top strip, left strip) so a
    framing change cannot fake an exposure change."""
    import numpy as np
    sw, sh = (90, 160) if height >= width else (160, 90)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an",
                        "-vf", f"scale={sw}:{sh},format=yuv444p", "-f", "rawvideo", "-"],
                       capture_output=True)
    frame = sw * sh * 3
    n = len(r.stdout) // frame
    if n == 0:
        return {"motion": [], "luma": None, "u": None, "v": None}
    arr = np.frombuffer(r.stdout[: n * frame], dtype=np.uint8).reshape(n, 3, sh, sw).astype(np.float32)
    y = arr[:, 0]
    diff = np.abs(np.diff(y, axis=0)).mean(axis=(1, 2)) if n > 1 else np.zeros(1)
    fps = n / max(probe_duration(path), 1e-6)
    step = max(1, int(round(fps / 2)))  # motion curve at 2 Hz
    motion = [[round(i / fps, 2), round(float(diff[i:i + step].mean()), 2)] for i in range(0, len(diff), step)]
    samples = arr[:: max(1, int(round(fps)))]
    th, lw = max(1, sh // 6), max(1, sw // 6)
    def med3(ch):
        whole = float(samples[:, ch].mean())
        top = float(samples[:, ch, :th, :].mean())
        left = float(samples[:, ch, :, :lw].mean())
        return round(statistics.median([whole, top, left]), 2)
    return {"motion": motion, "luma": med3(0), "u": med3(1), "v": med3(2)}


_dur_cache: dict[str, float] = {}


def probe_duration(path: Path) -> float:
    k = str(path)
    if k not in _dur_cache:
        _dur_cache[k] = probe(path)["duration"]
    return _dur_cache[k]


# ---- transcription -----------------------------------------------------------------

_model = None


def whisper_model(name: str = "small.en"):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        try:
            _model = WhisperModel(name, device="cuda", compute_type="float16")
        except Exception:
            _model = WhisperModel(name, device="cpu", compute_type="int8")
    return _model


def transcribe(path: Path, prompt: str = "") -> list[dict]:
    """Word-level transcript: [{"w": word, "s": start, "e": end}, ...]."""
    global _model
    try:
        return _transcribe(whisper_model(), path, prompt)
    except RuntimeError as err:  # GPU present but CUDA libraries missing: fails at first encode
        if "cuda" not in str(err).lower() and "cublas" not in str(err).lower():
            raise
        from faster_whisper import WhisperModel
        _model = WhisperModel("small.en", device="cpu", compute_type="int8")
        return _transcribe(_model, path, prompt)


def _transcribe(model, path: Path, prompt: str) -> list[dict]:
    segs, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=False,
                               initial_prompt=prompt or None, condition_on_previous_text=False)
    words = []
    for seg in segs:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)})
    return [w for w in words if w["w"]]


# ---- judgement helpers (pure, tested offline) -------------------------------------

def speech_span(words: list[dict], silences: list[list[float]], duration: float) -> tuple[float, float]:
    """First and last moment of speech. Words win; silencedetect is the fallback."""
    if words:
        return words[0]["s"], words[-1]["e"]
    a, b = 0.0, duration
    if silences and silences[0][0] <= 0.05:
        a = silences[0][1] or duration
    if silences and (silences[-1][1] is None or silences[-1][1] >= duration - 0.05):
        b = silences[-1][0]
    return a, max(a, b)


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def false_start(words: list[dict], n: int = 4) -> float | None:
    """A take that restarts a phrase near its end and then trails off (the classic
    "I built one that manages... I built one that manages") gets an out point
    just before the restart. Returns that time, or None."""
    toks = [_norm(w["w"]) for w in words]
    if len(toks) < 2 * n:
        return None
    for i in range(len(toks) - n, int(len(toks) * 0.5) - 1, -1):
        gram = toks[i:i + n]
        if not all(gram):
            continue
        for j in range(0, i - n + 1):
            if toks[j:j + n] == gram:
                while j > 0 and i - 1 >= j + n and toks[i - 1] == toks[j - 1]:
                    i, j = i - 1, j - 1  # walk back to where the repeat really starts
                tail = len(toks) - i
                if tail <= n + 6:  # the restart is where the take ends
                    return round(max(words[i - 1]["e"], words[i]["s"] - 0.3), 3)
    return None


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()


def find_retakes(clips: list[dict], threshold: float = 0.6) -> dict[str, str]:
    """Map a weaker take -> the take that supersedes it, for clips that say the same thing."""
    out = {}
    for i, a in enumerate(clips):
        for b in clips[i + 1:]:
            if a["text"] and b["text"] and similarity(a["text"], b["text"]) >= threshold:
                # The later, more complete take wins: more words, and no false start.
                score = lambda c: (c["false_start"] is None, len(c["words"]))
                weak, strong = (a, b) if score(a) <= score(b) else (b, a)
                out[weak["id"]] = strong["id"]
    return out


def moment_score(clip: dict) -> float:
    """How big a moment is: loudness and motion during speech, plus speaking rate."""
    a, b = clip["speech"]
    loud = [v for t, v in clip["loudness"] if a <= t <= b and v > -60] or [-60.0]
    mot = [v for t, v in clip["motion"] if a <= t <= b] or [0.0]
    rate = len(clip["words"]) / max(b - a, 0.5)
    return round((statistics.mean(loud) + 60) / 10 + statistics.mean(mot) / 4 + rate / 2, 3)


def snap(t: float, words: list[dict], side: str) -> float:
    """Never cut mid-word: an in point inside a word moves to just before it,
    an out point inside a word moves to just after it."""
    for w in words:
        if w["s"] < t < w["e"]:
            return round(max(0.0, w["s"] - 0.05), 3) if side == "in" else round(w["e"] + 0.1, 3)
    return t


# ---- the inventory -----------------------------------------------------------------

def build_inventory(folder: Path, cache_dir: Path, names_prompt: str = "", log=print) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for path in list_clips(folder):
        cache = cache_dir / (path.name + ".json")
        st = path.stat()
        key = [st.st_size, int(st.st_mtime), names_prompt]
        if cache.exists():
            c = json.loads(cache.read_text(encoding="utf-8"))
            if c.get("_key") == key:
                clips.append(c)
                continue
        log(f"  analysing {path.name}")
        info = probe(path)
        sil, loud = audio_scan(path) if info["has_audio"] else ([], [])
        for s in sil:
            if s[1] is None:
                s[1] = round(info["duration"], 3)
        vid = video_scan(path, info["width"], info["height"])
        words = transcribe(path, names_prompt) if info["has_audio"] else []
        c = {
            "id": path.name, **info,
            "silences": sil, "loudness": loud, "motion": vid["motion"],
            "luma": vid["luma"], "u": vid["u"], "v": vid["v"],
            "words": words, "text": " ".join(w["w"] for w in words),
        }
        c["speech"] = list(speech_span(words, sil, info["duration"]))
        c["false_start"] = false_start(words)
        c["_key"] = key
        cache.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
        clips.append(c)
    clips.sort(key=lambda c: (c["created"] or "", c["id"]))
    for c in clips:
        c["score"] = moment_score(c)
    retakes = find_retakes(clips)
    for c in clips:
        c["superseded_by"] = retakes.get(c["id"])
    return {"clips": clips}


def brain_view(inv: dict) -> list[dict]:
    """The inventory trimmed to what an editor needs to decide, small enough for a prompt."""
    view = []
    for i, c in enumerate(inv["clips"], 1):
        view.append({
            "n": i, "clip": c["id"], "duration": round(c["duration"], 2),
            "size": f'{c["width"]}x{c["height"]}',
            "speech": [round(x, 2) for x in c["speech"]],
            "transcript": " ".join(f'[{w["s"]:.1f}]{w["w"]}' for w in c["words"]),
            "long_pauses": [s for s in c["silences"] if c["speech"][0] < s[0] < c["speech"][1]],
            "false_start_at": c["false_start"],
            "moment_score": c["score"],
            "retake_superseded_by": c["superseded_by"],
        })
    return view


def durations(inv: dict) -> dict[str, float]:
    return {c["id"]: c["duration"] for c in inv["clips"]}
