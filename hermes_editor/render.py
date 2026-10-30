"""EDL -> finished files, with ffmpeg.

One filter graph and one encode per output: every piece is trimmed from the
source, colour and exposure matched to the other clips, scaled onto the canvas
and joined with a concat filter (no concat-by-copy, which clicks at AAC joins).
Loudness is two-pass loudnorm to the EDL target (-14 LUFS by default): pass 1
measures the exact signal pass 2 will receive. Captions are transcribed from the
rendered master's own audio, so they cannot drift across joins, then burned in
from an ASS file declared at the real frame size and also saved as .srt.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import edl as E

FPS = 30
ENC_V = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]
ENC_A = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
Q = ["-hide_banner", "-nostats", "-y"]


@dataclass
class Piece:
    clip: str
    a: float
    b: float
    join_before: bool  # True when this piece starts a new clip (a real edit), not a pause trim


# ---- pure planning (tested offline) ---------------------------------------------------

def pieces_for(cut: E.Cut, silences: list[list[float]]) -> list[tuple[float, float]]:
    """Source ranges for one cut, with every internal pause longer than the pace
    allows shortened to that cap (half kept on each side, so no word is clipped)."""
    cap = E.PAUSE_CAP[cut.pace]
    ranges, start = [], cut.in_s
    for s0, s1 in silences:
        s0, s1 = max(s0, cut.in_s), min(s1, cut.out_s)
        if s1 - s0 <= cap or s0 <= cut.in_s + 0.01 or s1 >= cut.out_s - 0.01:
            continue
        ranges.append((start, s0 + cap / 2))
        start = s1 - cap / 2
    ranges.append((start, cut.out_s))
    return [(round(a, 3), round(b, 3)) for a, b in ranges if b - a > 0.05]


def plan_pieces(cuts: list[E.Cut], inv: dict) -> list[Piece]:
    sil = {c["id"]: c["silences"] for c in inv["clips"]}
    out = []
    for cut in cuts:
        for i, (a, b) in enumerate(pieces_for(cut, sil[cut.clip])):
            out.append(Piece(cut.clip, a, b, i == 0))
    return out


def grade(clip: dict, ref: dict) -> str:
    """Exposure by gamma (never an additive lift) and a small chroma shift toward the
    other clips: 80% of the way to the median, capped at +-12 luma / +-3 chroma."""
    parts = []
    y = clip.get("luma")
    if y is not None and ref.get("luma") is not None:
        d = max(-12.0, min(12.0, 0.8 * (ref["luma"] - y)))
        yn = min(max((y - 16) / 219, 0.02), 0.98)
        yt = min(max((y + d - 16) / 219, 0.02), 0.98)
        g = math.log(yn) / math.log(yt)
        if abs(g - 1) > 0.01:
            parts.append(f"eq=gamma={g:.3f}")
    du = dv = 0
    if clip.get("u") is not None and ref.get("u") is not None:
        du = round(max(-3, min(3, 0.8 * (ref["u"] - clip["u"]))))
        dv = round(max(-3, min(3, 0.8 * (ref["v"] - clip["v"]))))
    if du or dv:
        parts.append(f"lutyuv=u=val{du:+d}:v=val{dv:+d}")
    return ",".join(parts)


def reference(clips: list[dict]) -> dict:
    def med(k):
        vals = [c[k] for c in clips if c.get(k) is not None]
        return statistics.median(vals) if vals else None
    return {"luma": med("luma"), "u": med("u"), "v": med("v")}


def canvas_for(clips: list[dict]) -> tuple[int, int]:
    sizes = {(c["width"], c["height"]) for c in clips}
    if len(sizes) == 1:
        w, h = sizes.pop()
        return w - w % 2, h - h % 2
    portrait = sum(c["height"] > c["width"] for c in clips) > len(clips) / 2
    return (1080, 1920) if portrait else (1920, 1080)


def same_shape(clip: dict, canvas: tuple[int, int]) -> bool:
    """True when the clip has the canvas's aspect ratio (within 1%), so it scales with no bars."""
    W, H = canvas
    w, h = clip.get("width"), clip.get("height")
    return not (w and h) or abs(w * H - h * W) <= 0.01 * w * H


def fit_chain(i: int, canvas: tuple[int, int], fill: bool, same: bool) -> str:
    """Scale piece i's (graded, denoised) video onto the canvas. Fill crops to the canvas;
    otherwise a same-shape clip just scales, and a different-shape clip sits full-height (or
    full-width) over a blurred, darkened copy of itself instead of black bars."""
    W, H = canvas
    if fill:
        return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},unsharp=5:5:0.7:5:5:0"
    if same:
        return (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
                "unsharp=5:5:0.7:5:5:0")
    return (f"split[bg{i}][fg{i}];"
            f"[bg{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"boxblur=40:2,eq=brightness=-0.25[bb{i}];"
            f"[fg{i}]scale={W}:{H}:force_original_aspect_ratio=decrease,unsharp=5:5:0.7:5:5:0[ff{i}];"
            f"[bb{i}][ff{i}]overlay=(W-w)/2:(H-h)/2")


def build_graph(pieces: list[Piece], by_id: dict, ref: dict, canvas: tuple[int, int], fill: bool,
                end_png: str | None, end_s: float, video: bool = True,
                loud: str = "", gain_db: float = 0.0) -> tuple[list[str], str]:
    """Return (ffmpeg input args, filter_complex). Outputs are [vout] and [aout].
    Pieces from clips with no audio track get a silent input of the same length."""
    W, H = canvas
    args, chains, labels = [], [], []
    for i, p in enumerate(pieces):
        args += ["-ss", f"{p.a:.3f}", "-t", f"{p.b - p.a:.3f}", "-i", by_id[p.clip]["path"]]
    n_in = len(pieces)
    for i, p in enumerate(pieces):
        d = p.b - p.a
        if by_id[p.clip].get("has_audio", True):
            fade = min(0.015, d / 4)
            chains.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,highpass=f=80,"
                          f"equalizer=f=250:t=q:w=1:g=-1.5,equalizer=f=2800:t=q:w=1:g=2.5,"
                          f"afade=t=in:d={fade:.3f},afade=t=out:st={d - fade:.3f}:d={fade:.3f},"
                          f"atrim=0:{d:.3f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            args += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            chains.append(f"[{n_in}:a]aformat=channel_layouts=stereo,atrim=0:{d:.3f},asetpts=PTS-STARTPTS[a{i}]")
            n_in += 1
        if video:
            g = grade(by_id[p.clip], ref)
            fit = fit_chain(i, canvas, fill, same_shape(by_id[p.clip], canvas))
            chains.append(f"[{i}:v]{g + ',' if g else ''}hqdn3d=0.8:0.6:0:0,{fit},"
                          f"setsar=1,fps={FPS},format=yuv420p,trim=0:{d:.3f},setpts=PTS-STARTPTS[v{i}]")
        labels.append(i)
    n = len(pieces)
    if end_png:
        args += ["-loop", "1", "-t", f"{end_s:.3f}", "-i", end_png,
                 "-f", "lavfi", "-t", f"{end_s:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        chains.append(f"[{n_in + 1}:a]aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a{n}]")
        if video:
            chains.append(f"[{n_in}:v]scale={W}:{H},setsar=1,fps={FPS},format=yuv420p,"
                          f"fade=t=in:st=0:d=0.4,setpts=PTS-STARTPTS[v{n}]")
        labels.append(n)
    if video:
        ins = "".join(f"[v{i}][a{i}]" for i in labels)
        chains.append(f"{ins}concat=n={len(labels)}:v=1:a=1[vout][acat]")
    else:
        ins = "".join(f"[a{i}]" for i in labels)
        chains.append(f"{ins}concat=n={len(labels)}:v=0:a=1[acat]")
    # Gain to the target first, then a -2 dBFS limiter: that holds the crest factor low enough
    # for loudnorm to stay in linear mode (one constant gain) instead of riding the level.
    chains.append("[acat]acompressor=threshold=-24dB:ratio=3:attack=10:release=150:makeup=1,"
                  f"volume={gain_db:.2f}dB,alimiter=limit=0.794:level=0{',' + loud if loud else ''}[aout]")
    return args, ";".join(chains)


def loudnorm_pass2(m: dict, target: float) -> str:
    return (f"loudnorm=I={target}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
            f"offset={m['target_offset']}:linear=true:print_format=json,aresample=48000")


def parse_loudnorm(stderr: str) -> dict:
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr)
    if not blocks:
        raise RuntimeError("loudnorm printed no measurement")
    return json.loads(blocks[-1])


# ---- captions ----------------------------------------------------------------------

def _core(w: str) -> str:
    return re.sub(r"^\W+|\W+$", "", w)


def _ends_sentence(w: str) -> bool:
    return w.rstrip("\"')]”’")[-1:] in ".?!"


def fix_case(words: list[dict], names: str | list[str] = "") -> list[dict]:
    """Whisper capitalises the first word of each of its segments, which often lands
    mid-sentence ("the man Files the charge"). Lower-case a capitalised word unless it starts
    a sentence, is "I"/"I'm"-style, is an acronym or mixed-case name ("AI", "McKay"), is in
    `names` (the transcriber's --names), or is a word said more than once and capitalised
    every time. A name said once and not in `names` is lower-cased: pass it in --names."""
    if isinstance(names, str):
        names = re.split(r"[,\s]+", names)
    keep = {n.lower() for n in names if n}
    seen: dict[str, list[bool]] = {}
    for w in words:
        c = _core(w["w"])
        if c:
            seen.setdefault(c.lower(), []).append(c[0].isupper())
    out, prev = [], None
    for w in words:
        c = _core(w["w"])
        lower = (prev is not None and not _ends_sentence(prev) and c[:1].isupper()
                 and not (c == "I" or c.startswith(("I'", "I’")))
                 and not (len(c) > 1 and (c.isupper() or not c[1:].islower()))
                 and c.lower() not in keep
                 and not (len(seen[c.lower()]) > 1 and all(seen[c.lower()])))
        if lower:
            i = w["w"].index(c[0])
            w = {**w, "w": w["w"][:i] + c[0].lower() + w["w"][i + 1:]}
        out.append(w)
        prev = w["w"]
    return out


def make_cues(words: list[dict], joins: list[float], max_chars: int, max_lines: int = 2,
              min_d: float = 1.2, max_d: float = 5.5, names: str | list[str] = "") -> list[dict]:
    """Group words into cues: at most `max_lines` lines of `max_chars`, 1.2-5.5 s each,
    and no cue straddling a join. A word belongs to the side of a join its midpoint is on.
    Mid-sentence capitals from the transcriber are fixed first (see fix_case)."""
    words = fix_case(words, names)

    def seg(t):
        return sum(t >= j for j in joins)
    cues, cur = [], []

    def flush():
        if cur:
            cues.append({"s": cur[0]["s"], "e": cur[-1]["e"], "words": list(cur)})
            cur.clear()

    for w in words:
        mid = (w["s"] + w["e"]) / 2
        if cur:
            text = " ".join(x["w"] for x in cur + [w])
            same = seg((cur[0]["s"] + cur[0]["e"]) / 2) == seg(mid)
            if (not same or len(wrap(text, max_chars)) > max_lines or w["e"] - cur[0]["s"] > max_d
                    or (cur[-1]["w"][-1:] in ".?!" and cur[-1]["e"] - cur[0]["s"] >= min_d)):
                flush()
        cur.append(w)
    flush()
    for prev, c in zip(cues, cues[1:]):  # no one-word orphans: borrow a word from the cue before
        same = seg(prev["words"][-1]["s"]) == seg(c["words"][0]["s"])
        if len(c["words"]) == 1 and len(prev["words"]) > 3 and same and prev["words"][-1]["w"][-1:] not in ".?!":
            c["words"].insert(0, prev["words"].pop())
            prev["e"], c["s"] = prev["words"][-1]["e"], c["words"][0]["s"]
    for i, c in enumerate(cues):  # stretch short cues into the gap after them, never past a join
        c["words"] = [w["w"] for w in c["words"]]
        limit = cues[i + 1]["s"] if i + 1 < len(cues) else c["e"] + min_d
        nxt = [j for j in joins if j > c["e"]]
        if nxt:
            limit = min(limit, nxt[0])
        if c["e"] - c["s"] < min_d:
            c["e"] = round(min(limit, c["s"] + min_d), 3)
        c["text"] = "\n".join(wrap(" ".join(c.pop("words")), max_chars))
    return cues


def wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def _ts(t: float, sep: str = ",") -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def srt(cues: list[dict]) -> str:
    return "".join(f"{i}\n{_ts(c['s'])} --> {_ts(c['e'])}\n{c['text']}\n\n" for i, c in enumerate(cues, 1))


def ass(cues: list[dict], W: int, H: int) -> str:
    portrait = H > W
    size = round(W * 0.062) if portrait else round(H * 0.05)
    margin_v = round(H * (0.24 if portrait else 0.08))  # clears the Shorts/Reels UI at the bottom
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
            "Alignment, MarginL, MarginR, MarginV, BorderStyle, Outline, Shadow\n"
            f"Style: Cap,Arial,{size},&H00FFFFFF,&H00000000,&H64000000,-1,2,{round(W * .06)},{round(W * .06)},"
            f"{margin_v},1,{max(2, size // 14)},0\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Text\n")
    def t(x):
        cs = int(round(x * 100))
        return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"
    return head + "".join(f"Dialogue: 0,{t(c['s'])},{t(c['e'])},Cap,{c['text'].replace(chr(10), chr(92) + 'N')}\n"
                          for c in cues)


# ---- end card ----------------------------------------------------------------------

def end_card_png(card: E.EndCard, W: int, H: int, path: Path, logo: Path | None = None) -> str:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (W, H), (11, 15, 20))
    d = ImageDraw.Draw(im)
    if logo and logo.exists():
        lg = Image.open(logo).convert("RGBA")
        lg.thumbnail((int(W * 0.4), int(H * 0.22)))
        im.paste(lg, ((W - lg.width) // 2, int(H * 0.36) - lg.height - int(H * 0.03)), lg)

    def font(px):
        for name in ("bahnschrift.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(name, px)
            except OSError:
                continue
        return ImageFont.load_default(size=px)

    gold, grey = (212, 175, 55), (200, 204, 210)
    big = min(W, H) * 0.11
    title_f = font(int(big))
    while d.textlength(card.title, font=title_f) > W * 0.88 and big > 12:
        big *= 0.92
        title_f = font(int(big))
    sub_f = font(int(big * 0.36))
    y = H * 0.44 - big / 2
    d.text(((W - d.textlength(card.title, font=title_f)) / 2, y), card.title, font=title_f, fill=gold)
    ry = y + big * 1.25
    d.line((W * 0.38, ry, W * 0.62, ry), fill=gold, width=max(2, W // 640))
    if card.subtitle:
        d.text(((W - d.textlength(card.subtitle, font=sub_f)) / 2, ry + big * 0.3), card.subtitle, font=sub_f, fill=grey)
    im.save(path)
    return str(path)


# ---- running ffmpeg ----------------------------------------------------------------

def ff(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(["ffmpeg", *Q, *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-2500:])
    return r


def measure(path: Path, target: float = -14.0) -> dict:
    r = ff(["-i", str(path), "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"])
    m = parse_loudnorm(r.stderr)
    return {"lufs": float(m["input_i"]), "true_peak": float(m["input_tp"]), "lra": float(m["input_lra"])}


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def render(pieces: list[Piece], inv: dict, out: Path, work: Path, canvas: tuple[int, int], fill: bool,
           end_card: E.EndCard | None, lufs: float, captions: bool, transcribe, log=print,
           names: str = "") -> dict:
    """Render one timeline to `out` (+ `out`.srt). Returns a report of what was measured."""
    work.mkdir(parents=True, exist_ok=True)
    by_id = {c["id"]: c for c in inv["clips"]}
    ref = reference(list(by_id.values()))
    W, H = canvas
    logo = Path(inv["folder"]) / end_card.logo if end_card and end_card.logo and inv.get("folder") else None
    png = end_card_png(end_card, W, H, work / f"{out.stem}.endcard.png", logo) if end_card else None
    end_s = end_card.seconds if end_card else 0.0

    measure_ln = f"loudnorm=I={lufs}:TP=-1.5:LRA=11:print_format=json"

    def measure_chain(gain):
        args, fc = build_graph(pieces, by_id, ref, canvas, fill, png, end_s, video=False, loud=measure_ln, gain_db=gain)
        return parse_loudnorm(ff([*args, "-filter_complex", fc, "-map", "[aout]", "-f", "null", "-"]).stderr)

    log(f"  {out.name}: loudness pass 1 (measure)")
    i0 = float(measure_chain(0.0)["input_i"])
    if math.isfinite(i0):
        gain = max(-20.0, min(30.0, lufs - i0))
        m1 = measure_chain(gain)  # measure exactly what pass 2's loudnorm will receive
        loud = loudnorm_pass2(m1, lufs)
    else:  # nothing but silence (every clip mute): no level to normalise
        gain, m1, loud = 0.0, {"input_i": "-inf"}, ""

    log(f"  {out.name}: pass 2 (render)")
    clean = work / f"{out.stem}.clean.mp4"
    args, fc = build_graph(pieces, by_id, ref, canvas, fill, png, end_s, loud=loud, gain_db=gain)
    r2 = ff([*args, "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]", *ENC_V, *ENC_A,
             "-movflags", "+faststart", str(clean)])
    m2 = parse_loudnorm(r2.stderr) if loud else {}

    joins, t = [], 0.0
    for p in pieces:
        if p.join_before and t > 0:
            joins.append(round(t, 3))
        t += p.b - p.a
    if end_card:
        joins.append(round(t, 3))

    report = {"file": out.name, "pieces": len(pieces), "joins": joins,
              "loudnorm": {"type": m2.get("normalization_type"), "pass1_input_i": m1["input_i"]}}
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    if captions:
        log(f"  {out.name}: captions from the rendered audio")
        words = transcribe(clean)
        cues = make_cues(words, joins, max_chars=26 if H > W else 39, names=names)
        srt_path = out.with_suffix(".srt")
        srt_path.write_text(srt(cues), encoding="utf-8")
        ass_name = f"{out.stem}.ass"
        (work / ass_name).write_text(ass(cues, W, H), encoding="utf-8")
        # Run from the work folder so the subtitles filter gets a bare file name (no drive-letter escaping).
        ff(["-i", str(clean.resolve()), "-vf", f"subtitles={ass_name}", *ENC_V, "-c:a", "copy",
            "-movflags", "+faststart", str(tmp.resolve())], cwd=str(work))
        report["captions"] = {"cues": len(cues), "srt": srt_path.name}
    else:
        tmp.write_bytes(clean.read_bytes())
    os.replace(tmp, out)  # write to .tmp and swap: ffmpeg truncates on open
    report["duration"] = round(duration(out), 2)
    report.update(measure(out, lufs))
    return report
