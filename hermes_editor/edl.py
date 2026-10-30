"""The edit decision list: the one document every other step reads or writes.

The brain writes an EDL, the renderer reads it, a change request rewrites it.
Keeping it explicit JSON means every decision Hermes makes can be read,
diffed and edited by hand before anything is rendered.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

VERSION = 1
PACES = ("tight", "normal", "relaxed")
# Longest internal pause kept at each pace, in seconds. Longer pauses are cut down to this.
PAUSE_CAP = {"tight": 0.25, "normal": 0.5, "relaxed": 1.0}


class EDLError(ValueError):
    pass


@dataclass
class Cut:
    clip: str            # clip id (file name inside the clips folder)
    in_s: float
    out_s: float
    pace: str = "normal"
    reason: str = ""


@dataclass
class Short:
    clip: str
    in_s: float
    out_s: float
    reason: str = ""


@dataclass
class EndCard:
    title: str
    subtitle: str = ""
    seconds: float = 2.5
    logo: str = ""      # image file name inside the clips folder, shown above the title


@dataclass
class EDL:
    brief: str
    target_seconds: float | None
    pace: str
    opening: str            # clip id of the opening shot
    cuts: list[Cut]
    shorts: list[Short] = field(default_factory=list)
    end_card: EndCard | None = None
    captions: bool = True
    loudness_lufs: float = -14.0
    notes: str = ""
    history: list[str] = field(default_factory=list)
    version: int = VERSION

    # ---- derived -------------------------------------------------------
    def source_seconds(self) -> float:
        """Length of the cut before pause trimming, end card included."""
        n = sum(c.out_s - c.in_s for c in self.cuts)
        return n + (self.end_card.seconds if self.end_card else 0.0)

    # ---- io --------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for c in d["cuts"] + d["shorts"]:
            c["in"] = round(c.pop("in_s"), 3)
            c["out"] = round(c.pop("out_s"), 3)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def copy(self) -> "EDL":
        return copy.deepcopy(self)


def _num(v, name):
    try:
        return float(v)
    except (TypeError, ValueError):
        raise EDLError(f"{name} must be a number, got {v!r}")


def from_dict(d: dict, clips: dict[str, float] | None = None) -> EDL:
    """Parse and validate. `clips` maps clip id -> duration; when given, every
    reference and every in/out point is checked against the real footage."""
    if not isinstance(d, dict):
        raise EDLError("EDL must be a JSON object")
    try:
        cuts_in = d["cuts"]
    except KeyError:
        raise EDLError("EDL has no 'cuts'")
    if not isinstance(cuts_in, list) or not cuts_in:
        raise EDLError("'cuts' must be a non-empty list")

    def span(item, kind):
        if not isinstance(item, dict):
            raise EDLError(f"each {kind} must be an object")
        clip = item.get("clip")
        if not isinstance(clip, str) or not clip:
            raise EDLError(f"{kind} is missing 'clip'")
        a = _num(item.get("in", item.get("in_s")), f"{kind} {clip} in")
        b = _num(item.get("out", item.get("out_s")), f"{kind} {clip} out")
        if a < 0 or b <= a:
            raise EDLError(f"{kind} {clip}: need 0 <= in < out, got {a}..{b}")
        if clips is not None:
            if clip not in clips:
                raise EDLError(f"{kind} refers to unknown clip {clip!r}")
            if b > clips[clip] + 0.05:
                raise EDLError(f"{kind} {clip}: out {b} is past the clip's end ({clips[clip]:.2f}s)")
            b = min(b, clips[clip])
        return clip, a, b

    pace = d.get("pace", "normal")
    if pace not in PACES:
        raise EDLError(f"pace must be one of {PACES}, got {pace!r}")

    cuts = []
    for item in cuts_in:
        clip, a, b = span(item, "cut")
        p = item.get("pace", pace)
        if p not in PACES:
            raise EDLError(f"cut {clip}: pace must be one of {PACES}, got {p!r}")
        cuts.append(Cut(clip, a, b, p, str(item.get("reason", ""))))

    shorts = []
    for item in d.get("shorts", []) or []:
        clip, a, b = span(item, "short")
        if b - a > 60:
            raise EDLError(f"short {clip}: {b - a:.1f}s is longer than 60s")
        shorts.append(Short(clip, a, b, str(item.get("reason", ""))))
    if len(shorts) > 3:
        raise EDLError("at most 3 shorts")

    ec = d.get("end_card")
    end_card = None
    if ec:
        if not isinstance(ec, dict) or not ec.get("title"):
            raise EDLError("end_card needs a 'title'")
        end_card = EndCard(str(ec["title"]), str(ec.get("subtitle", "")), _num(ec.get("seconds", 2.5), "end_card seconds"),
                           str(ec.get("logo") or ""))

    target = d.get("target_seconds")
    target = None if target in (None, "") else _num(target, "target_seconds")
    opening = d.get("opening") or cuts[0].clip
    if opening != cuts[0].clip:
        raise EDLError(f"opening is {opening!r} but the first cut is {cuts[0].clip!r}")

    return EDL(
        brief=str(d.get("brief", "")),
        target_seconds=target,
        pace=pace,
        opening=opening,
        cuts=cuts,
        shorts=shorts,
        end_card=end_card,
        captions=bool(d.get("captions", True)),
        loudness_lufs=_num(d.get("loudness_lufs", -14.0), "loudness_lufs"),
        notes=str(d.get("notes", "")),
        history=[str(h) for h in d.get("history", [])],
    )


def load(path: str | Path, clips: dict[str, float] | None = None) -> EDL:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")), clips)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (it may wrap it in prose or a fence)."""
    start = text.find("{")
    if start < 0:
        raise EDLError("no JSON object in the reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise EDLError("unterminated JSON object in the reply")
