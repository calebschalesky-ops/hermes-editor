"""Brief -> EDL: the headline step, with a fake model and with the offline rules."""
import json

from hermes_editor import brain, rules
from conftest import FakeClient

MODEL_EDL = {
    "brief": "x", "target_seconds": 30, "pace": "tight", "opening": "c.mp4",
    "cuts": [{"clip": "c.mp4", "in": 0.35, "out": 5.5, "reason": "biggest moment first"},
             {"clip": "a.mp4", "in": 0.4, "out": 5.3}, {"clip": "d.mp4", "in": 0.3, "out": 3.4}],
    "shorts": [{"clip": "c.mp4", "in": 0.35, "out": 5.5, "reason": "self-contained claim"}],
    "end_card": {"title": "Sam's Gym", "subtitle": "", "seconds": 2}, "captions": True, "loudness_lufs": -14,
}


def test_fake_model_brief_becomes_validated_edl(inv):
    client = FakeClient("Sure! " + json.dumps(MODEL_EDL))
    e = brain.plan("keep it under 30 seconds, open on the biggest moment, logo at the end", inv, client, log=lambda s: None)
    assert e.opening == "c.mp4" and [c.clip for c in e.cuts] == ["c.mp4", "a.mp4", "d.mp4"]
    assert e.end_card.title == "Sam's Gym" and e.history[0] == "brain: fake"
    assert e.brief.startswith("keep it under 30")
    p = client.prompts[0]  # the model sees the brief, the transcripts and the schema
    assert "open on the biggest moment" in p and "]topped" in p and '"cuts"' in p


def test_snaps_model_cuts_off_mid_word(inv):
    d = json.loads(json.dumps(MODEL_EDL))
    first_word = inv["clips"][2]["words"][0]
    d["cuts"][0]["in"] = first_word["s"] + 0.1  # inside "Last"
    e = brain.plan("", inv, FakeClient(d), log=lambda s: None)
    assert e.cuts[0].in_s < first_word["s"]


def test_invalid_reply_retried_then_accepted(inv):
    bad = dict(MODEL_EDL, cuts=[{"clip": "ghost.mp4", "in": 0, "out": 2}], opening="ghost.mp4")
    client = FakeClient(bad, MODEL_EDL)
    e = brain.plan("anything", inv, client, log=lambda s: None)
    assert len(client.prompts) == 2 and "unknown clip" in client.prompts[1]
    assert e.opening == "c.mp4"


def test_broken_model_falls_back_to_rules(inv):
    client = FakeClient("I can't do that", "still no")
    e = brain.plan("under 20 seconds", inv, client, log=lambda s: None)
    assert e.history[0] == "brain: rules" and any("fallback" in h for h in e.history)
    assert e.target_seconds == 20


def test_rules_brief_without_a_model(inv):
    e = rules.plan('keep it under 15 seconds, upbeat, open on the biggest moment, end card "Sam\'s Gym"', inv)
    assert e.pace == "tight" and e.target_seconds == 15
    assert e.opening == "c.mp4"                       # highest moment score
    assert e.end_card.title == "Sam's Gym"
    assert sum(c.out_s - c.in_s for c in e.cuts) + e.end_card.seconds <= 15
    assert e.cuts[0].in_s < inv["clips"][2]["speech"][0]  # head trimmed to just before the first word


def test_rules_open_on_a_named_line_and_drop(inv):
    e = rules.plan("open on the line about kids classes, drop the clip about the free class", inv)
    assert e.opening == "b.mp4" and "d.mp4" not in [c.clip for c in e.cuts]


def test_rules_no_brief_keeps_everything(inv):
    e = rules.plan("", inv)
    assert [c.clip for c in e.cuts] == ["a.mp4", "b.mp4", "c.mp4", "d.mp4"] and e.end_card is None
    assert 1 <= len(e.shorts) <= 3 and e.shorts[0].clip == "c.mp4"


def test_retakes_and_false_starts(inv):
    inv["clips"][0]["superseded_by"] = "b.mp4"
    inv["clips"][1]["false_start"] = 3.0
    e = rules.plan("", inv)
    assert "a.mp4" not in [c.clip for c in e.cuts]
    assert next(c for c in e.cuts if c.clip == "b.mp4").out_s <= 3.0


def test_auto_brain_without_key_or_cli_is_rules(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(brain, "find_claude", lambda: None)
    assert brain.pick_client("auto") is None
