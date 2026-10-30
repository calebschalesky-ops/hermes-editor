"""Footage with no talking: the action peaks decide the cut, the opening and the shorts."""
import json

import pytest

from hermes_editor import analyze, brain, rules
from conftest import FakeClient, action_clip, clip, curve


@pytest.fixture
def sport():
    """A loud, fast talker, and two wordless clips: a small hop and a big catch at 7s."""
    talker = clip("talk.mp4", "Welcome back everybody today is a huge day at the club and I am so excited", score=6.0)
    talker["loudness"] = curve(talker["duration"], -14.0, [])
    return {"clips": [
        talker,
        action_clip("hop.mp4", 10.0, motion=[(3.0, 5.0)]),
        action_clip("catch.mp4", 12.0, motion=[(7.0, 14.0), (9.5, 9.0)], loud=[(7.2, -12.0)]),
    ]}


def test_peaks_are_spaced_biggest_first_and_skip_the_settling_camera():
    c = action_clip("x.mp4", 10.0, motion=[(0.0, 30.0), (4.0, 9.0), (4.5, 8.0), (8.0, 6.0)])
    assert analyze.motion_peaks(c) == [[4.0, 9.0], [8.0, 6.0]]   # the flat 1.0 stretch is not a peak


def test_chatter_is_not_speech():
    talking = clip("t.mp4", "one two three four five six seven eight")
    chatter = clip("c.mp4", "go ahead recording", duration=20.0)
    assert analyze.has_speech(talking) and not analyze.has_speech(chatter)


def test_action_span_merges_close_peaks_and_borrows_at_the_edge():
    c = action_clip("x.mp4", 12.0, motion=[(7.0, 14.0), (9.5, 9.0)])
    assert analyze.action_span(c) == (5.5, 11.5, [7.0, 9.5])
    edge = action_clip("e.mp4", 6.0, motion=[(5.5, 10.0)])
    a, b, _ = analyze.action_span(edge)
    assert (a, b) == (2.5, 6.0)                    # still 3.5s long


def test_a_loud_jump_counts_a_steady_loud_voice_does_not(sport):
    talker, hop, catch = sport["clips"]
    assert analyze.action_score(catch) > analyze.action_score(hop) > analyze.action_score(talker)
    quiet = action_clip("q.mp4", 12.0, motion=[(7.0, 14.0), (9.5, 9.0)])
    assert analyze.action_score(catch) > analyze.action_score(quiet)


def test_biggest_catch_opens_on_the_catch_not_the_loudest_talker(sport):
    e = rules.plan("open on the biggest catch", sport)
    assert e.opening == "catch.mp4"
    assert any("biggest action peak, motion 14.0 at 7.0s" in h for h in e.history)
    first = e.cuts[0]
    assert (first.in_s, first.out_s) == (5.5, 11.5) and first.reason.startswith("action")
    hop = next(c for c in e.cuts if c.clip == "hop.mp4")
    assert (hop.in_s, hop.out_s) == (1.5, 5.0)     # not the whole 10s clip
    talk = next(c for c in e.cuts if c.clip == "talk.mp4")
    assert talk.reason.startswith("speech span")   # talking clips still cut on words


def test_mostly_wordless_footage_ranks_by_action_without_being_told(sport):
    e = rules.plan("open on the biggest moment", sport)
    assert e.opening == "catch.mp4"
    assert any("ranked by action" in h for h in e.history)
    assert e.shorts[0].clip == "catch.mp4" and (e.shorts[0].in_s, e.shorts[0].out_s) == (5.5, 11.5)
    assert "action score" in e.shorts[0].reason


def test_talking_footage_keeps_speech_ranking(inv):
    e = rules.plan("open on the biggest moment", inv)
    assert e.opening == "c.mp4" and not any("ranked by action" in h for h in e.history)


def test_the_part_where_he_jumps_and_the_intro_that_cannot_be_found(sport):
    e = rules.plan("open on the part where he jumps, cut the boring intro", sport)
    assert e.opening == "catch.mp4"
    assert 'ignored: "cut the boring intro" (no clip matches "the boring intro")' in e.history


def test_flat_broll_is_kept_whole():
    flat = action_clip("pan.mp4", 6.0, base_motion=2.0, motion=[(3.0, 2.4)])
    e = rules.plan("", {"clips": [flat]})
    assert (e.cuts[0].in_s, e.cuts[0].out_s) == (0.0, 6.0)


def test_the_model_sees_when_the_action_happens(sport):
    view = {v["clip"]: v for v in analyze.brain_view(sport)}
    assert view["catch.mp4"]["has_speech"] is False and view["talk.mp4"]["has_speech"] is True
    assert view["catch.mp4"]["motion_peaks"][0] == [7.0, 14.0]
    assert view["catch.mp4"]["loud_moments"][0] == [7.2, -12.0]
    assert view["catch.mp4"]["action_window"] == [5.5, 11.5]
    reply = {"opening": "catch.mp4", "pace": "tight", "notes": "no slow-motion footage, so no replay",
             "cuts": [{"clip": "catch.mp4", "in": 5.5, "out": 11.5}]}
    client = FakeClient(json.dumps(reply))
    e = brain.plan("open on the biggest catch", sport, client, log=lambda s: None)
    p = client.prompts[0]
    assert '"motion_peaks"' in p and "peak - 1.5s" in p and "never whoever talks loudest" in p
    assert e.history[-1] == "notes: no slow-motion footage, so no replay"
