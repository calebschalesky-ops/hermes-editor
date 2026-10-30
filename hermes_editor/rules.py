"""The deterministic brain: plain-words brief -> EDL with no model at all.

It understands a fixed vocabulary (length, pace, opening shot, end card, dropping
or moving clips, number of shorts). The brief is split into clauses, and every
clause it does not act on is written into the EDL's history as `ignored: ...`,
so nothing is dropped silently. It is the offline fallback and the baseline the
model-backed brains are tested against.
"""
from __future__ import annotations

import re
import statistics

from . import edl as E
from .analyze import action_score, action_span, has_speech, motion_peaks, snap
from .render import pieces_for

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3}
BIG = ("biggest", "strongest", "best", "loudest", "most energetic", "big moment", "hook")
STOP = set("the a an on with of to in and or my our your it is that this open start begin clip shot".split())
ACTION = re.compile(r"\b(catch(es|ing)?|caught|jump(s|ed|ing)?|lift(s|ed|ing)?|tricks?|plays?|action|"
                    r"dunks?|goals?|throws?|threw|kicks?|sprint(s|ing)?|run(s|ning)?|races?|flips?|"
                    r"landing|tackles?|dives?|swings?|spikes?|hits?)\b", re.I)
WORKS = ('"drop the second clip", "slow down the middle", "open on clip 3", '
         '"make it under 30 seconds", "no end card"')


# ---- reading the brief ----------------------------------------------------------------

VERBS = "cut|drop|remove|skip|lose|open|start|begin|lead|make|keep|add|end|slow|speed|move|use|put|give|no|finish"
SPLIT = re.compile(r"[,;]|\.(?!\d)|\s+(?:and|then|but|also|plus)\s+(?=(?:then\s+|also\s+)?(?:%s)\b)" % VERBS, re.I)
DEFAULTS = re.compile(r"\b(best takes?|captions?|subtitles?|full story)\b", re.I)


def clauses(text: str) -> list[str]:
    """Split a brief into the separate things it asks for. Quoted text stays whole."""
    quoted: list[str] = []

    def hide(m):
        quoted.append(m.group(0))
        return f"\x00{len(quoted) - 1}\x00"
    t = re.sub(r'"[^"]*"|“[^”]*”', hide, text)
    out = []
    for p in SPLIT.split(t):
        p = re.sub(r"\x00(\d+)\x00", lambda m: quoted[int(m.group(1))], p or "").strip()
        if p:
            out.append(p)
    return out


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


def parse_shorts(text: str) -> int | None:
    t = text.lower()
    if re.search(r"\bno shorts?\b", t):
        return 0
    m = re.search(r"\b(\d|one|two|three)\s+shorts?", t)
    return ({"one": 1, "two": 2, "three": 3}.get(m.group(1)) or int(m.group(1))) if m else None


DROP = re.compile(r"\b(?:drop|cut|remove|skip|lose)\s+([^,.;]+)", re.I)
TARGET = re.compile(r"\b(clip|take|shot|line|part|bit|about|intro|outro)\b", re.I)
OPEN = re.compile(r"\b(?:open|start|begin|lead)\s+(?:on|with)\s+([^,.;]+)", re.I)
MOVE = re.compile(r"\bmove\s+([^,.;]+?)\s+to the (end|start|beginning|front)", re.I)


def drop_targets(text: str) -> list[str]:
    """Phrases after drop/cut/remove that name a clip ("cut it short" does not).
    The end card has its own words and is not a clip."""
    return [m.group(1) for m in DROP.finditer(text)
            if TARGET.search(m.group(1)) and not re.search(r"end card|logo|outro", m.group(1), re.I)]


def keywords(phrase: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", phrase.lower()) if w not in STOP and len(w) > 2]


# ---- judging moments ------------------------------------------------------------------

def action_mode(text: str, clips: list[dict]) -> bool:
    """Rank moments by action, not speech, when the brief is about action or at least
    half the footage has no talking in it."""
    return bool(ACTION.search(text)) or 2 * sum(not has_speech(c) for c in clips) >= len(clips)


def strength(c: dict, action: bool) -> float:
    return action_score(c) if action else c["score"]


def why_big(c: dict, action: bool) -> str:
    if action:
        ps = motion_peaks(c, 1)
        return (f"biggest action peak, motion {ps[0][1]:.1f} at {ps[0][0]:.1f}s" if ps
                else "biggest action peak")
    return f"biggest moment by speech energy, score {c['score']}"


def match_clip(phrase: str, clips: list[dict], action: bool = False) -> dict | None:
    """The clip a phrase points at: "the line about X", "the biggest catch", "clip 3",
    "the part where he jumps" (no line says it, so the biggest action moment)."""
    p = phrase.lower()
    m = re.search(r"\bclip (\d+)\b", p) or re.search(r"\b(" + "|".join(ORDINALS) + r")\b", p)
    if m:
        k = int(m.group(1)) if m.group(1).isdigit() else ORDINALS[m.group(1)]
        return clips[k - 1] if 0 < k <= len(clips) else None
    action = action or bool(ACTION.search(p))
    if any(b in p for b in BIG):
        return max(clips, key=lambda c: strength(c, action))
    kw = keywords(phrase)
    if kw:
        best = max(clips, key=lambda c: sum(k in c["text"].lower() for k in kw))
        if any(k in best["text"].lower() for k in kw):
            return best
    if ACTION.search(p) and any(action_score(c) > 0 for c in clips):
        return max(clips, key=lambda c: action_score(c))
    return None


def _stands_out(c: dict) -> bool:
    mot = [v for _, v in c["motion"]]
    ps = motion_peaks(c, 1)
    return bool(ps) and ps[0][1] >= 1.5 * max(statistics.median(mot), 0.1)


def _cut_for(c: dict, pace: str) -> E.Cut:
    span = action_span(c) if not has_speech(c) and _stands_out(c) else None
    if span:  # no talking: cut around the action instead of keeping the whole clip
        a, b, ts = span
        why = (f"action: motion peak{'s' if len(ts) > 1 else ''} at "
               f"{', '.join(f'{t:.1f}s' for t in ts)}, cut 1.5s before to 2s after")
        return E.Cut(c["id"], snap(a, c["words"], "in"), snap(b, c["words"], "out"), pace, why)
    a = max(0.0, c["speech"][0] - 0.15)
    b = min(c["duration"], c["speech"][1] + 0.25)
    why = "speech span, head and tail silence trimmed"
    b = snap(b, c["words"], "out")
    if c.get("false_start"):
        b = min(b, c["false_start"])
        why += f"; ends before the false start at {c['false_start']:.2f}s"
    return E.Cut(c["id"], snap(a, c["words"], "in"), b, pace, why)


def _fit(cuts: list[E.Cut], target: float | None, end: E.EndCard | None, by_id: dict, history: list,
         action: bool = False):
    if not target:
        return cuts
    budget = target - (end.seconds if end else 0.0)
    total = lambda: sum(c.out_s - c.in_s for c in cuts)
    # Drop the weakest clips (never the opening) until the cut fits.
    while total() > budget and len(cuts) > 1:
        weakest = min(cuts[1:], key=lambda c: strength(by_id[c.clip], action))
        cuts.remove(weakest)
        history.append(f"dropped {weakest.clip} to fit {target:.0f}s (weakest moment)")
    if total() > budget:
        c = cuts[0]
        c.out_s = round(c.in_s + budget, 3)
        history.append(f"trimmed {c.clip} to fit {target:.0f}s")
    return cuts


def pick_shorts(clips: list[dict], n: int, cuts: list[E.Cut], action: bool = False) -> list[E.Short]:
    used = {c.clip: c for c in cuts}
    ranked = sorted((c for c in clips if c["id"] in used), key=lambda c: -strength(c, action))
    out = []
    for c in ranked[:n]:
        cut = used[c["id"]]
        b = min(cut.out_s, cut.in_s + 59.0)
        why = (f"action score {action_score(c)}, {cut.reason}" if cut.reason.startswith("action")
               else f"moment score {c['score']}, opens on the line")
        out.append(E.Short(c["id"], cut.in_s, b, why))
    return out


def _ignored(clause: str, why: str) -> str:
    return f'ignored: "{clause}" ({why})'


# ---- the two entry points -------------------------------------------------------------

def plan(brief: str, inv: dict, n_shorts: int = 2) -> E.EDL:
    clips = inv["clips"]
    by_id = {c["id"]: c for c in clips}
    history = ["brain: rules"]
    pace = target = end = None

    keep = [c for c in clips if not c.get("superseded_by")]
    for c in clips:
        if c.get("superseded_by"):
            history.append(f"dropped {c['id']}: retake of {c['superseded_by']}")
    action = action_mode(brief, keep)
    if action:
        history.append("moments ranked by action (motion peak + loudness jump), not speech")

    for cl in clauses(brief):
        acted, why_not = False, "not an instruction the rules brain knows"
        p = parse_pace(cl)
        if p:
            pace, acted = pace or p, True
        s = parse_seconds(cl)
        if s:
            target, acted = target or s, True
        ec = parse_end_card(cl)
        if ec is not None:
            end, acted = (ec or None), True
        k = parse_shorts(cl)
        if k is not None:
            n_shorts, acted = k, True
        for phrase in drop_targets(cl):
            c = match_clip(phrase, keep, action)
            if c and len(keep) > 1:
                keep.remove(c)
                history.append(f"dropped {c['id']} (drop {phrase.strip()})")
                acted = True
            else:
                why_not = f'no clip matches "{phrase.strip()}"'
        m = OPEN.search(cl)
        if m:
            c = match_clip(m.group(1), keep, action)
            if c:
                keep.remove(c)
                keep.insert(0, c)
                ph = m.group(1).lower()
                how = (f"; {why_big(c, action or bool(ACTION.search(ph)))}"
                       if any(b in ph for b in BIG) or ACTION.search(ph) else "")
                history.append(f"opens on {c['id']} ({m.group(0).strip()}{how})")
                acted = True
            else:
                why_not = f'no clip matches "{m.group(1).strip()}"'
        if DEFAULTS.search(cl) and not re.search(r"\b(no|without|drop|remove)\b", cl, re.I):
            acted = True  # already what every edit does
        if not acted:
            history.append(_ignored(cl, why_not))

    pace = pace or "normal"
    cuts = [_cut_for(c, pace) for c in keep]
    cuts = _fit(cuts, target, end, by_id, history, action)
    return E.EDL(brief=brief, target_seconds=target, pace=pace, opening=cuts[0].clip, cuts=cuts,
                 shorts=pick_shorts(clips, min(3, n_shorts), cuts, action), end_card=end, history=history)


def _length(cut: E.Cut, clip: dict, pace: str) -> float:
    """Rendered length of a cut at a pace: exactly what the renderer will keep."""
    probe = E.Cut(cut.clip, cut.in_s, cut.out_s, pace)
    return sum(b - a for a, b in pieces_for(probe, clip["silences"]))


def revise(prev: E.EDL, change: str, inv: dict) -> E.EDL:
    """Apply a plain-words change request to an existing EDL. Parts it cannot act on are
    recorded as ignored; if it can act on none of it, that is an EDLError."""
    new = prev.copy()
    clips = inv["clips"]
    by_id = {c["id"]: c for c in clips}
    did, ignored = [], []

    def in_edit(phrase):
        """Resolve a phrase against the clips as they appear in the current cut."""
        order = [by_id[c.clip] for c in new.cuts if c.clip in by_id]
        return match_clip(phrase, order, action_mode(change, order))

    for cl in clauses(change):
        n_did, why_not = len(did), "not an instruction the rules brain knows"
        t = cl.lower()
        for phrase in drop_targets(cl):
            c = in_edit(phrase)
            if c and len(new.cuts) > 1:
                new.cuts = [x for x in new.cuts if x.clip != c["id"]]
                new.shorts = [s for s in new.shorts if s.clip != c["id"]]
                did.append(f"dropped {c['id']} (drop {phrase.strip()})")
            else:
                why_not = f'no clip in the edit matches "{phrase.strip()}"'

        pace = parse_pace(cl)
        if pace:
            where = "middle" if "middle" in t else "start" if re.search(r"\b(start|opening|beginning)\b", t) \
                else "end" if re.search(r"\b(end|ending|last)\b", t) else "all"
            n = len(new.cuts)
            idx = {"all": range(n), "start": range(min(1, n)), "end": range(max(0, n - 1), n),
                   "middle": range(1, n - 1) if n > 2 else range(n)}[where]
            delta, flat = 0.0, []
            for i in idx:
                cut = new.cuts[i]
                before = _length(cut, by_id[cut.clip], cut.pace)
                after = _length(cut, by_id[cut.clip], pace)
                if abs(after - before) < 0.05:
                    flat.append((cut.clip, min(E.PAUSE_CAP[cut.pace], E.PAUSE_CAP[pace])))
                delta += after - before
                cut.pace = pace
            if where == "all":
                new.pace = pace
            did.append(f"pace {pace} on the {where} ({len(idx)} clip{'s' if len(idx) != 1 else ''}, "
                       f"{delta:+.1f}s)")
            for clip, cap in flat:  # honest about a change that changes nothing
                did.append(f"pace set to {pace} on {clip}; it has no pause longer than {cap:g}s, "
                           f"so its length won't change")

        m = OPEN.search(cl)
        if m:
            c = match_clip(m.group(1), clips, action_mode(cl, clips))
            if c:
                existing = next((x for x in new.cuts if x.clip == c["id"]), None) or _cut_for(c, new.pace)
                new.cuts = [existing] + [x for x in new.cuts if x.clip != c["id"]]
                did.append(f"opens on {c['id']}")
            else:
                why_not = f'no clip matches "{m.group(1).strip()}"'
        m = MOVE.search(cl)
        if m:
            c = in_edit(m.group(1))
            if c:
                x = next(x for x in new.cuts if x.clip == c["id"])
                new.cuts.remove(x)
                new.cuts = new.cuts + [x] if m.group(2) == "end" else [x] + new.cuts
                did.append(f"moved {c['id']} to the {m.group(2)}")
            else:
                why_not = f'no clip in the edit matches "{m.group(1).strip()}"'

        secs = parse_seconds(cl)
        if secs:
            new.target_seconds = secs
            new.cuts = _fit(new.cuts, secs, new.end_card, by_id, did)
            did.append(f"target {secs:.0f}s")

        ec = parse_end_card(cl)
        if ec is False:
            new.end_card = None
            did.append("removed the end card")
        elif ec:
            new.end_card = ec
            did.append(f"end card '{ec.title}'")

        if len(did) == n_did:
            ignored.append((cl, why_not))

    if not did:
        parts = "; ".join(f'"{cl}": {why}' for cl, why in ignored) or "it is empty"
        raise E.EDLError(f"could not understand the change request ({parts}). Phrases that work: {WORKS}")
    new.opening = new.cuts[0].clip
    new.history.append(f"change: {change} -> " + "; ".join(did))
    new.history += [_ignored(cl, why) for cl, why in ignored]
    return new
