"""The deterministic brain: plain-words brief -> EDL with no model at all.

It understands a fixed vocabulary (length, pace, opening shot, end card, dropping
or moving clips, number of shorts). It is the offline fallback and the baseline
the model-backed brains are tested against.
"""
from __future__ import annotations

import re

from . import edl as E
from .analyze import snap

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3}
BIG = ("biggest", "strongest", "best", "loudest", "most energetic", "big moment", "hook")
STOP = set("the a an on with of to in and or my our your it is that this open start begin clip shot".split())


def parse_seconds(text: str) -> float | None:
    t = text.lower()
    m = re.search(r"(under|less than|max(?:imum)?|at most|around|about|no (?:more|longer) than)?\s*"
                  r"(\d+(?:\.\d+)?|a|one|two|three)\s*(seconds?|secs?|s\b|minutes?|mins?|m\b)", t)
    if not m:
        return None
    n = {"a": 1, "one": 1, "two": 2, "three": 3}.get(m.group(2))
    n = float(n if n is not None else m.group(2))
    return n * 60 if m.group(3).startswith("m") else n


def parse_pace(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(upbeat|fast|snappy|tight|punchy|quick|energetic|speed (it )?up)\b", t):
        return "tight"
    if re.search(r"\b(slow|relaxed|calm|breathe|room to breathe|gentle)\b", t):
        return "relaxed"
    return None


def parse_end_card(text: str) -> E.EndCard | None | bool:
    """EndCard to add, False to remove, None if the text does not mention one."""
    t = text.lower()
    if re.search(r"\b(no|without|drop|remove)( the)? (end card|logo|outro)", t):
        return False
    if not re.search(r"end card|logo|outro|end on", t):
        return None
    q = re.findall(r"[\"“]([^\"”]{2,60})[\"”]", text) or re.findall(r"(?:^|\s)'([^']{2,60})'(?=\s|[,.;]|$)", text)
    if q:
        return E.EndCard(q[0], q[1] if len(q) > 1 else "")
    return E.EndCard("Thanks for watching")


DROP = re.compile(r"\b(?:drop|cut|remove|skip|lose)\s+([^,.;]+)", re.I)
TARGET = re.compile(r"\b(clip|take|shot|line|part|bit|about|intro|outro)\b", re.I)


def drop_targets(text: str) -> list[str]:
    """Phrases after drop/cut/remove that name a clip ("cut it short" does not)."""
    return [m.group(1) for m in DROP.finditer(text) if TARGET.search(m.group(1))]


def keywords(phrase: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", phrase.lower()) if w not in STOP and len(w) > 2]


def match_clip(phrase: str, clips: list[dict]) -> dict | None:
    """The clip a phrase points at: "the line about X", "the biggest moment", "clip 3"."""
    p = phrase.lower()
    m = re.search(r"\bclip (\d+)\b", p) or re.search(r"\b(" + "|".join(ORDINALS) + r")\b", p)
    if m:
        k = int(m.group(1)) if m.group(1).isdigit() else ORDINALS[m.group(1)]
        return clips[k - 1] if 0 < k <= len(clips) else None
    if any(b in p for b in BIG):
        return max(clips, key=lambda c: c["score"])
    kw = keywords(phrase)
    if not kw:
        return None
    best = max(clips, key=lambda c: sum(k in c["text"].lower() for k in kw))
    return best if any(k in best["text"].lower() for k in kw) else None


def _cut_for(c: dict, pace: str) -> E.Cut:
    a = max(0.0, c["speech"][0] - 0.15)
    b = min(c["duration"], c["speech"][1] + 0.25)
    why = "speech span, head and tail silence trimmed"
    b = snap(b, c["words"], "out")
    if c.get("false_start"):
        b = min(b, c["false_start"])
        why += f"; ends before the false start at {c['false_start']:.2f}s"
    return E.Cut(c["id"], snap(a, c["words"], "in"), b, pace, why)


def _fit(cuts: list[E.Cut], target: float | None, end: E.EndCard | None, by_id: dict, history: list):
    if not target:
        return cuts
    budget = target - (end.seconds if end else 0.0)
    total = lambda: sum(c.out_s - c.in_s for c in cuts)
    # Drop the weakest clips (never the opening) until the cut fits.
    while total() > budget and len(cuts) > 1:
        weakest = min(cuts[1:], key=lambda c: by_id[c.clip]["score"])
        cuts.remove(weakest)
        history.append(f"dropped {weakest.clip} to fit {target:.0f}s (weakest moment)")
    if total() > budget:
        c = cuts[0]
        c.out_s = round(c.in_s + budget, 3)
        history.append(f"trimmed {c.clip} to fit {target:.0f}s")
    return cuts


def pick_shorts(clips: list[dict], n: int, cuts: list[E.Cut]) -> list[E.Short]:
    used = {c.clip: c for c in cuts}
    ranked = sorted((c for c in clips if c["id"] in used), key=lambda c: -c["score"])
    out = []
    for c in ranked[:n]:
        cut = used[c["id"]]
        b = min(cut.out_s, cut.in_s + 59.0)
        out.append(E.Short(c["id"], cut.in_s, b, f"moment score {c['score']}, opens on the line"))
    return out


def plan(brief: str, inv: dict, n_shorts: int = 2) -> E.EDL:
    clips = inv["clips"]
    by_id = {c["id"]: c for c in clips}
    history = ["brain: rules"]
    pace = parse_pace(brief) or "normal"
    target = parse_seconds(brief)
    end = parse_end_card(brief) or None

    keep = [c for c in clips if not c.get("superseded_by")]
    for c in clips:
        if c.get("superseded_by"):
            history.append(f"dropped {c['id']}: retake of {c['superseded_by']}")
    for phrase in drop_targets(brief):
        c = match_clip(phrase, keep)
        if c:
            keep.remove(c)
            history.append(f"dropped {c['id']} (drop {phrase.strip()})")

    m = re.search(r"\b(?:open|start|begin|lead)\s+(?:on|with)\s+([^,.;]+)", brief, re.I)
    if m:
        c = match_clip(m.group(1), keep)
        if c:
            keep.remove(c)
            keep.insert(0, c)
            history.append(f"opens on {c['id']} ({m.group(0).strip()})")

    cuts = [_cut_for(c, pace) for c in keep]
    cuts = _fit(cuts, target, end, by_id, history)
    m = re.search(r"(\d|one|two|three)\s+shorts?", brief.lower())
    if m:
        n_shorts = {"one": 1, "two": 2, "three": 3}.get(m.group(1)) or int(m.group(1))
    return E.EDL(brief=brief, target_seconds=target, pace=pace, opening=cuts[0].clip, cuts=cuts,
                 shorts=pick_shorts(clips, min(3, n_shorts), cuts), end_card=end, history=history)


def revise(prev: E.EDL, change: str, inv: dict) -> E.EDL:
    """Apply a plain-words change request to an existing EDL."""
    new = prev.copy()
    clips = inv["clips"]
    by_id = {c["id"]: c for c in clips}
    did = []
    t = change.lower()

    def in_edit(phrase):
        """Resolve a phrase against the clips as they appear in the current cut."""
        order = [by_id[c.clip] for c in new.cuts if c.clip in by_id]
        return match_clip(phrase, order)

    for phrase in drop_targets(change):
        if re.search(r"end card|logo|outro", phrase, re.I):
            continue
        c = in_edit(phrase)
        if c and len(new.cuts) > 1:
            new.cuts = [x for x in new.cuts if x.clip != c["id"]]
            new.shorts = [s for s in new.shorts if s.clip != c["id"]]
            did.append(f"dropped {c['id']} (drop {phrase.strip()})")

    pace = parse_pace(change)
    if pace:
        where = "middle" if "middle" in t else "start" if re.search(r"\b(start|opening|beginning)\b", t) \
            else "end" if re.search(r"\b(end|ending|last)\b", t) else "all"
        n = len(new.cuts)
        idx = {"all": range(n), "start": range(min(1, n)), "end": range(max(0, n - 1), n),
               "middle": range(1, n - 1) if n > 2 else range(n)}[where]
        for i in idx:
            new.cuts[i].pace = pace
        if where == "all":
            new.pace = pace
        did.append(f"pace {pace} on the {where} ({len(idx)} clip{'s' if len(idx) != 1 else ''})")

    m = re.search(r"\b(?:open|start|begin|lead)\s+(?:on|with)\s+([^,.;]+)", change, re.I)
    if m:
        c = match_clip(m.group(1), clips)
        if c:
            existing = next((x for x in new.cuts if x.clip == c["id"]), None) or _cut_for(c, new.pace)
            new.cuts = [existing] + [x for x in new.cuts if x.clip != c["id"]]
            did.append(f"opens on {c['id']}")
    m = re.search(r"\bmove\s+([^,.;]+?)\s+to the (end|start|beginning|front)", change, re.I)
    if m:
        c = in_edit(m.group(1))
        if c:
            x = next(x for x in new.cuts if x.clip == c["id"])
            new.cuts.remove(x)
            new.cuts = new.cuts + [x] if m.group(2) == "end" else [x] + new.cuts
            did.append(f"moved {c['id']} to the {m.group(2)}")

    secs = parse_seconds(change)
    if secs:
        new.target_seconds = secs
        new.cuts = _fit(new.cuts, secs, new.end_card, by_id, did)
        did.append(f"target {secs:.0f}s")

    ec = parse_end_card(change)
    if ec is False:
        new.end_card = None
        did.append("removed the end card")
    elif ec:
        new.end_card = ec
        did.append(f"end card '{ec.title}'")

    if not did:
        raise E.EDLError(f"could not understand the change request: {change!r}")
    new.opening = new.cuts[0].clip
    new.history.append(f"change: {change} -> " + "; ".join(did))
    return new
