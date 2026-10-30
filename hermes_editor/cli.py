"""hermes: raw clips + plain words in, finished video + shorts + captions out.

    python -m hermes_editor CLIPS --brief "keep it under a minute, open on the biggest moment"
    python -m hermes_editor CLIPS --change "slow down the middle, drop the second clip"

Each run writes a new version folder (out/v1, out/v2, ...) holding the EDL it used,
the video, its .srt, the shorts and a report of what was measured. Source clips are
only ever read. Everything stays in the output folder for you to review.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import analyze, brain, render
from . import edl as E


def versions(out: Path) -> list[int]:
    return sorted(int(p.name[1:]) for p in out.glob("v*") if p.name[1:].isdigit() and (p / "edl.json").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hermes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", type=Path, help="folder of raw clips")
    ap.add_argument("--brief", default="", help="the edit, in plain words")
    ap.add_argument("--change", help="a change request applied to the latest version's EDL")
    ap.add_argument("--edl", type=Path, help="render this EDL file as-is (hand-edited, say)")
    ap.add_argument("--out", type=Path, default=Path("hermes-out"))
    ap.add_argument("--brain", choices=["auto", "api", "cli", "rules"], default="auto")
    ap.add_argument("--shorts", type=int, default=2, help="how many vertical shorts (0-3)")
    ap.add_argument("--names", default="", help="proper nouns to bias the transcriber, e.g. 'Schalesky'")
    ap.add_argument("--logo", help="image file in the clips folder to put on the end card")
    ap.add_argument("--plan-only", action="store_true", help="write the EDL, render nothing")
    a = ap.parse_args(argv)

    t0 = time.time()
    log = lambda s: print(s, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    folder = a.clips.resolve()

    log("1/4 looking at the footage")
    inv = analyze.build_inventory(folder, a.out / "cache", a.names, log)
    inv["folder"] = str(folder)
    for c in inv["clips"]:
        c["path"] = str(folder / c["id"])
    client = brain.pick_client(a.brain)
    log(f"    {len(inv['clips'])} clips, {sum(c['duration'] for c in inv['clips']):.1f}s of footage; "
        f"brain: {client.name if client else 'rules'}")

    done = versions(a.out)
    prev = None
    try:
        if a.change:
            if not done:
                ap.error("--change needs an earlier version in --out")
            prev = E.load(a.out / f"v{done[-1]}" / "edl.json", analyze.durations(inv))
            log(f"2/4 change request on v{done[-1]}: {a.change}")
            edit = brain.revise(prev, a.change, inv, client, log)
        elif a.edl:
            log(f"2/4 using {a.edl}")
            edit = E.load(a.edl, analyze.durations(inv))
        else:
            log(f"2/4 brief -> edit decision list: {a.brief or '(defaults)'}")
            edit = brain.plan(a.brief, inv, client, max(0, min(3, a.shorts)), log)
    except E.EDLError as err:  # one plain line and no new version, never a traceback
        print(f"hermes: {err}", file=sys.stderr, flush=True)
        return 2
    if a.logo:
        if not (folder / a.logo).is_file():
            ap.error(f"--logo {a.logo} is not a file in {folder}")
        edit.end_card = edit.end_card or E.EndCard("")
        edit.end_card.logo = a.logo

    vdir = a.out / f"v{(done[-1] if done else 0) + 1}"
    vdir.mkdir()
    edit.save(vdir / "edl.json")
    t_plan = time.time() - t0
    log(f"    {len(edit.cuts)} cuts, {len(edit.shorts)} shorts -> {vdir / 'edl.json'}")
    for h in edit.history:
        log(f"    - {h}")
    if a.plan_only:
        return 0

    words_of = lambda p: analyze.transcribe(p, a.names)
    report = {"version": vdir.name, "brain": edit.history[0] if edit.history else "", "outputs": []}

    log("3/4 rendering the video")
    canvas = render.canvas_for([c for c in inv["clips"] if c["id"] in {x.clip for x in edit.cuts}])
    pieces = render.plan_pieces(edit.cuts, inv)
    report["outputs"].append(render.render(pieces, inv, vdir / "video.mp4", vdir / "work", canvas, False,
                                           edit.end_card, edit.loudness_lufs, edit.captions, words_of, log))

    log("4/4 cutting the shorts")
    prev_dir = a.out / f"v{done[-1]}" if prev else None
    for i, s in enumerate(edit.shorts, 1):
        out = vdir / f"short-{i}.mp4"
        same = prev and i <= len(prev.shorts) and prev.shorts[i - 1] == s and (prev_dir / out.name).exists()
        if same:  # unchanged by the change request: reuse, don't re-render
            for ext in (".mp4", ".srt"):
                if (prev_dir / f"short-{i}{ext}").exists():
                    shutil.copy2(prev_dir / f"short-{i}{ext}", vdir / f"short-{i}{ext}")
            report["outputs"].append({"file": out.name, "reused_from": prev_dir.name,
                                      "duration": round(render.duration(out), 2), **render.measure(out)})
            continue
        cut = E.Cut(s.clip, s.in_s, s.out_s, "tight")
        report["outputs"].append(render.render(render.plan_pieces([cut], inv), inv, out, vdir / "work",
                                               (1080, 1920), True, None, edit.loudness_lufs, True, words_of, log))

    report["seconds"] = {"plan": round(t_plan, 1), "total": round(time.time() - t0, 1)}
    (vdir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(vdir / "work", ignore_errors=True)
    for o in report["outputs"]:
        log(f"    {o['file']}: {o.get('duration', '')}s, {o['lufs']:.1f} LUFS, peak {o['true_peak']:.1f} dBTP")
    log(f"done in {report['seconds']['total']:.0f}s -> {vdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
