"""Change requests take the previous EDL and re-cut it."""
import json

import pytest

from hermes_editor import brain, rules
from hermes_editor import edl as E
from conftest import FakeClient


def base(inv):
    return rules.plan('end card "Sam\'s Gym"', inv)  # a, b, c, d in order


def test_slow_down_the_middle_and_drop_the_second_clip(inv):
    prev = base(inv)
    new = rules.revise(prev, "slow down the middle, drop the second clip", inv)
    assert [c.clip for c in new.cuts] == ["a.mp4", "c.mp4", "d.mp4"]
    assert [c.pace for c in new.cuts] == ["normal", "relaxed", "normal"]
    assert new.end_card == prev.end_card                       # untouched parts survive
    assert new.history[-1].startswith("change: slow down the middle")
    assert [c.clip for c in prev.cuts] == ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]  # previous EDL not mutated


def test_reorder_length_and_end_card(inv):
    prev = base(inv)
    new = rules.revise(prev, "open on clip 3, no end card", inv)
    assert new.opening == "c.mp4" and new.end_card is None
    new = rules.revise(new, "move the clip about the free class to the start", inv)
    assert new.cuts[0].clip == "d.mp4"
    new = rules.revise(new, "make it under 10 seconds", inv)
    assert sum(c.out_s - c.in_s for c in new.cuts) <= 10


def test_dropping_a_clip_drops_its_short(inv):
    prev = base(inv)
    assert prev.shorts[0].clip == "c.mp4"
    new = rules.revise(prev, "drop the third clip", inv)
    assert all(s.clip != "c.mp4" for s in new.shorts)


def test_nonsense_change_is_an_error_not_a_silent_no_op(inv):
    with pytest.raises(E.EDLError, match="could not understand"):
        rules.revise(base(inv), "make it pop more", inv)


def test_change_through_fake_model(inv):
    prev = base(inv)
    d = prev.to_dict()
    d["cuts"] = [c for c in d["cuts"] if c["clip"] != "b.mp4"]
    d["cuts"][1]["pace"] = "relaxed"
    d["notes"] = "dropped b, relaxed the middle"
    client = FakeClient(json.dumps(d))
    new = brain.revise(prev, "slow down the middle, drop the second clip", inv, client, log=lambda s: None)
    assert [c.clip for c in new.cuts] == ["a.mp4", "c.mp4", "d.mp4"] and new.cuts[1].pace == "relaxed"
    assert "CURRENT EDL" in client.prompts[0] and '"b.mp4"' in client.prompts[0]
    assert new.history[-1].startswith("change (fake)") and "relaxed the middle" in new.history[-1]


def test_model_failure_on_change_falls_back_to_rules(inv):
    new = brain.revise(base(inv), "drop the second clip", inv, FakeClient("nope", "nope"), log=lambda s: None)
    assert "b.mp4" not in [c.clip for c in new.cuts]
