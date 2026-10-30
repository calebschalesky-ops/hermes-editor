import json

import pytest

from hermes_editor import edl as E

CLIPS = {"a.mp4": 6.0, "b.mp4": 8.0}


def good():
    return {"brief": "short", "target_seconds": 20, "pace": "tight", "opening": "b.mp4",
            "cuts": [{"clip": "b.mp4", "in": 0.4, "out": 7.5, "reason": "hook"},
                     {"clip": "a.mp4", "in": 0.2, "out": 5.0, "pace": "relaxed"}],
            "shorts": [{"clip": "b.mp4", "in": 1.0, "out": 7.0}],
            "end_card": {"title": "Acme Gym", "subtitle": "Book a class", "seconds": 2}}


def test_parses_and_round_trips(tmp_path):
    e = E.from_dict(good(), CLIPS)
    assert [c.clip for c in e.cuts] == ["b.mp4", "a.mp4"]
    assert e.cuts[0].pace == "tight" and e.cuts[1].pace == "relaxed"  # per-cut pace overrides
    assert e.end_card.title == "Acme Gym"
    assert e.source_seconds() == pytest.approx(7.1 + 4.8 + 2)
    p = tmp_path / "edl.json"
    e.save(p)
    d = json.loads(p.read_text())
    assert d["cuts"][0]["in"] == 0.4 and "in_s" not in d["cuts"][0]
    assert E.load(p, CLIPS) == e


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.pop("cuts"), "no 'cuts'"),
    (lambda d: d.update(cuts=[]), "non-empty"),
    (lambda d: d["cuts"][0].update({"in": 5, "out": 4}), "in < out"),
    (lambda d: d["cuts"][0].update(clip="zzz.mp4"), "unknown clip"),
    (lambda d: d["cuts"][1].update(out=9.0), "past the clip's end"),
    (lambda d: d.update(pace="warp"), "pace"),
    (lambda d: d.update(opening="a.mp4"), "opening"),
    (lambda d: d.update(end_card={"subtitle": "x"}), "title"),
    (lambda d: d.update(shorts=[{"clip": "a.mp4", "in": 0, "out": 1}] * 4), "at most 3"),
])
def test_rejects_bad_edl(mutate, msg):
    d = good()
    mutate(d)
    with pytest.raises(E.EDLError, match=msg):
        E.from_dict(d, CLIPS)


def test_extract_json_from_chatty_reply():
    reply = 'Here is the edit:\n```json\n{"cuts": [{"clip": "a.mp4", "in": 0, "out": 1, "reason": "has } brace"}]}\n```\nDone.'
    assert E.extract_json(reply)["cuts"][0]["reason"] == "has } brace"
    with pytest.raises(E.EDLError):
        E.extract_json("no json here")
