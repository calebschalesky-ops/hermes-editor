"""The headline step: a plain-words brief becomes an explicit edit decision list.

Three ways to think, chosen in this order by `--brain auto`:
  api     Anthropic Messages API, when ANTHROPIC_API_KEY is set
  cli     the local Claude Code CLI in headless mode (`claude -p`), no key needed
  rules   the deterministic parser in rules.py (offline; also the fallback)

Whatever the model returns is parsed and validated against the real footage.
A reply that fails validation gets one retry with the error, then falls back to rules.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

from . import edl as E
from . import rules
from .analyze import brain_view, durations, snap

MODEL = os.environ.get("HERMES_MODEL", "claude-opus-5-5")

GUIDE = """You are the editor in Hermes, an AI video editor. You turn a client's plain-words brief
into an edit decision list (EDL) for their raw clips. Your only output is the EDL.

Editing judgement:
- Keep the best takes. If two clips say the same thing, keep the more complete one.
- Trim dead air at the head and tail of every clip; set in/out on word boundaries using the
  [time]word transcript, a little before the first word and a little after the last.
- If a take restarts a phrase and trails off (false_start_at), end it before the restart.
- Order clips so the story makes sense; the opening clip is the hook the brief asks for.
- Respect target length: the sum of (out - in) plus the end card must fit. Internal pauses get
  shortened at render time by pace (tight 0.25s, normal 0.5s, relaxed 1.0s cap), so a cut can
  run a little over before pauses are trimmed.
- Shorts (1-3, vertical 9:16, max 60s) are the strongest self-contained moments. A short opens on
  the claim, not the wind-up, and needs no setup from elsewhere in the video.
- Only use clip ids that appear in the inventory. Never invent footage.

Return ONLY one JSON object, no prose, with this shape:
{
  "brief": str, "target_seconds": number|null, "pace": "tight"|"normal"|"relaxed",
  "opening": clip id (must equal cuts[0].clip),
  "cuts": [{"clip": id, "in": s, "out": s, "pace": "tight"|"normal"|"relaxed", "reason": str}],
  "shorts": [{"clip": id, "in": s, "out": s, "reason": str}],
  "end_card": {"title": str, "subtitle": str, "seconds": number} | null,
  "captions": true, "loudness_lufs": -14, "notes": str
}"""


def plan_prompt(brief: str, inv: dict, n_shorts: int) -> str:
    return (f"{GUIDE}\n\nBRIEF: {brief or '(none: use sensible defaults, full story, best takes)'}\n"
            f"SHORTS WANTED: {n_shorts}\n\nINVENTORY:\n{json.dumps(brain_view(inv), indent=1)}")


def change_prompt(prev: E.EDL, change: str, inv: dict) -> str:
    return (f"{GUIDE}\n\nThis is a CHANGE REQUEST on an existing edit. Clip numbers and 'first/second'"
            f" refer to the order in the CURRENT EDL. Change only what is asked; keep everything else.\n"
            f"CHANGE: {change}\n\nCURRENT EDL:\n{json.dumps(prev.to_dict(), indent=1)}\n\n"
            f"INVENTORY:\n{json.dumps(brain_view(inv), indent=1)}")


# ---- clients -------------------------------------------------------------------------

class ApiClient:
    name = "api"

    def __init__(self, model: str = MODEL):
        import anthropic  # optional dependency
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=4000,
                                          messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def find_claude() -> str | None:
    for name in ("claude.cmd", "claude.exe", "claude"):
        p = shutil.which(name)
        if p:
            return p
    return None


class CliClient:
    """Claude Code in headless print mode. No tools, no session saved, run from an
    empty temp folder so no project instructions leak into the edit."""
    name = "cli"

    def __init__(self, exe: str | None = None, timeout: int = 600):
        self.exe = exe or find_claude()
        if not self.exe:
            raise RuntimeError("claude CLI not found on PATH")
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([self.exe, "-p", "--output-format", "text", "--tools", "",
                                "--no-session-persistence", "--strict-mcp-config"],
                               input=prompt, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=tmp, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"claude -p exited {r.returncode}: {r.stderr.strip()[:300]}")
        return r.stdout


def pick_client(kind: str = "auto"):
    if kind == "rules":
        return None
    if kind in ("api", "auto") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ApiClient()
        except ImportError:
            if kind == "api":
                raise
    if kind in ("cli", "auto") and find_claude():
        return CliClient()
    if kind in ("api", "cli"):
        raise RuntimeError(f"brain '{kind}' is not available on this machine")
    return None


# ---- the step -------------------------------------------------------------------------

def _tidy(e: E.EDL, inv: dict) -> E.EDL:
    """Model output gets the same guarantees as rules output: no mid-word cuts."""
    words = {c["id"]: c["words"] for c in inv["clips"]}
    for c in e.cuts:
        c.in_s, c.out_s = snap(c.in_s, words[c.clip], "in"), snap(c.out_s, words[c.clip], "out")
    return e


def _ask(client, prompt: str, inv: dict, log) -> E.EDL:
    reply = client.complete(prompt)
    try:
        return E.from_dict(E.extract_json(reply), durations(inv))
    except (E.EDLError, json.JSONDecodeError) as err:
        log(f"  brain reply rejected ({err}); asking once more")
        reply = client.complete(prompt + f"\n\nYour previous reply was rejected: {err}. "
                                          "Return a corrected JSON object only.")
        return E.from_dict(E.extract_json(reply), durations(inv))


def plan(brief: str, inv: dict, client=None, n_shorts: int = 2, log=print) -> E.EDL:
    if client is not None:
        try:
            e = _ask(client, plan_prompt(brief, inv, n_shorts), inv, log)
            e.brief = brief
            e.history = [f"brain: {client.name}"] + e.history
            return _tidy(e, inv)
        except Exception as err:  # never leave the user with nothing
            log(f"  {client.name} brain failed ({err}); falling back to rules")
            e = rules.plan(brief, inv, n_shorts)
            e.history.append(f"fallback: {client.name} failed: {str(err)[:120]}")
            return e
    return rules.plan(brief, inv, n_shorts)


def revise(prev: E.EDL, change: str, inv: dict, client=None, log=print) -> E.EDL:
    if client is not None:
        try:
            e = _ask(client, change_prompt(prev, change, inv), inv, log)
            e.history = prev.history + [f"change ({client.name}): {change}"]
            if e.notes:
                e.history[-1] += f" -> {e.notes}"
            return _tidy(e, inv)
        except Exception as err:
            log(f"  {client.name} brain failed ({err}); falling back to rules")
    return rules.revise(prev, change, inv)
