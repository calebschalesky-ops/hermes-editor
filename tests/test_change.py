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
    with pytest.raises(E.EDLError, match="could not understand") as err:
        rules.revise(base(inv), "make it pop more", inv)
    assert '"make it pop more": not an instruction' in str(err.value) and "drop the second clip" in str(err.value)


def test_part_of_a_change_it_cannot_do_is_recorded_not_dropped(inv):
    new = rules.revise(base(inv), "drop the second clip, add some sparkle", inv)
    assert "b.mp4" not in [c.clip for c in new.cuts]
    assert new.history[-1] == 'ignored: "add some sparkle" (not an instruction the rules brain knows)'


def test_slowing_down_clips_with_no_long_pauses_says_the_length_will_not_change(inv):
    new = rules.revise(base(inv), "slow down the middle", inv)
    assert [c.pace for c in new.cuts] == ["normal", "relaxed", "relaxed", "normal"]
    h = new.history[-1]
    assert "pace relaxed on the middle (2 clips, +0.0s)" in h
    assert "pace set to relaxed on b.mp4; it has no pause longer than 0.5s, so its length won't change" in h


def test_slowing_down_a_clip_with_a_long_pause_lengthens_it(inv):
    b = inv["clips"][1]
    b["silences"].append([2.0, 3.5])           # a 1.5s pause between words
    new = rules.revise(base(inv), "slow down the middle", inv)
    h = new.history[-1]
    assert "(2 clips, +0.5s)" in h and "on b.mp4" not in h and "on c.mp4;" in h


def test_cli_prints_one_line_and_makes_no_version_for_a_change_it_cannot_do(inv, tmp_path, monkeypatch, capsys):
    from hermes_editor import analyze, cli
    monkeypatch.setattr(analyze, "build_inventory", lambda *a, **k: inv)
    (tmp_path / "clips").mkdir()
    out = tmp_path / "out"
    (out / "v1").mkdir(parents=True)
    base(inv).save(out / "v1" / "edl.json")
    code = cli.main([str(tmp_path / "clips"), "--out", str(out), "--brain", "rules", "--change", "make it pop more"])
    err = capsys.readouterr().err.strip()
    assert code == 2 and not (out / "v2").exists()
    assert len(err.splitlines()) == 1 and err.startswith("hermes: could not understand") and "Traceback" not in err


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
