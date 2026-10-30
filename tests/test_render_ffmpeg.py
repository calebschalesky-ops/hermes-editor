"""One real render, end to end: three generated clips (portrait with audio, landscape with
audio, portrait with no audio track) through render.render, then the file is measured.
Skips when ffmpeg, numpy or Pillow is missing. No Whisper: the transcriber is a fake."""
import json
import shutil
import subprocess

import pytest

from hermes_editor import render
from hermes_editor import edl as E

if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    pytest.skip("ffmpeg not installed", allow_module_level=True)
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

W, H = 360, 640


def make(path, size, tone):
    v = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration=1"]
    a = ["-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration=1"] if tone else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", *v, *a, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    *(["-c:a", "aac"] if tone else []), "-shortest", str(path)], check=True)


def clip(cid, path, w, h, audio):
    return {"id": cid, "path": str(path), "width": w, "height": h, "has_audio": audio,
            "duration": 1.0, "silences": [], "luma": 120.0, "u": 128.0, "v": 128.0}


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                        str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def frame(path, t):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W)


def test_render_end_to_end(tmp_path):
    make(tmp_path / "p.mp4", f"{W}x{H}", 440)
    make(tmp_path / "l.mp4", f"{H}x{W}", 660)
    make(tmp_path / "m.mp4", f"{W}x{H}", None)
    inv = {"folder": str(tmp_path), "clips": [clip("p.mp4", tmp_path / "p.mp4", W, H, True),
                                              clip("l.mp4", tmp_path / "l.mp4", H, W, True),
                                              clip("m.mp4", tmp_path / "m.mp4", W, H, False)]}
    assert render.canvas_for(inv["clips"]) == (1080, 1920)   # mixed shapes, portrait majority
    pieces = render.plan_pieces([E.Cut(c["id"], 0.0, 1.0) for c in inv["clips"]], inv)
    card = E.EndCard("Hermes", "test", seconds=1.0)
    plan_s = sum(p.b - p.a for p in pieces) + card.seconds
    heard = []

    def transcribe(path):
        heard.append(path)
        return [{"w": "Hello", "s": 0.1, "e": 0.4}, {"w": "There", "s": 0.5, "e": 0.8}]

    out = tmp_path / "out.mp4"
    rep = render.render(pieces, inv, out, tmp_path / "work", (W, H), False, card, -14.0, True, transcribe,
                        log=lambda s: None)

    assert out.exists() and heard
    info = probe(out)
    kinds = {s["codec_type"]: s for s in info["streams"]}
    assert set(kinds) == {"video", "audio"}
    assert (kinds["video"]["width"], kinds["video"]["height"]) == (W, H)
    assert abs(float(info["format"]["duration"]) - plan_s) <= 0.2
    assert abs(rep["lufs"] - -14.0) <= 1.0, rep
    assert out.with_suffix(".srt").read_text(encoding="utf-8").count("Hello there") == 1  # "There" lower-cased
    # The landscape clip (1.0-2.0 s) sits on a blurred copy of itself, not black bars.
    top = frame(out, 1.5)[: H // 6]
    assert top.mean() > 20, top.mean()
