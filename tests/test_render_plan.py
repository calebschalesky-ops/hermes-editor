"""The renderer's planning steps, checked without running ffmpeg."""
import pytest

from hermes_editor import analyze, render
from hermes_editor import edl as E


def test_long_pauses_are_capped_by_pace():
    sil = [[0.0, 0.5], [2.0, 4.0], [5.0, 5.2], [7.0, 8.0]]
    tight = render.pieces_for(E.Cut("x", 0.4, 7.2, "tight"), sil)
    assert tight == [(0.4, 2.125), (3.875, 7.2)]               # 2 s pause -> 0.25 s; 0.2 s pause kept
    relaxed = render.pieces_for(E.Cut("x", 0.4, 7.2, "relaxed"), sil)
    assert relaxed == [(0.4, 2.5), (3.5, 7.2)]
    assert sum(b - a for a, b in relaxed) > sum(b - a for a, b in tight)


def test_joins_only_between_clips(inv):
    cuts = [E.Cut("a.mp4", 0.3, 4.0), E.Cut("b.mp4", 0.3, 4.0)]
    inv["clips"][0]["silences"] = [[1.0, 3.0]]
    ps = render.plan_pieces(cuts, inv)
    assert [p.join_before for p in ps] == [True, False, True]


def test_grade_is_gamma_and_capped():
    ref = {"luma": 140.0, "u": 128.0, "v": 128.0}
    g = render.grade({"luma": 100.0, "u": 120.0, "v": 131.0}, ref)
    assert g.startswith("eq=gamma=1.") and "brightness" not in g
    assert "u=val+3" in g and "v=val-2" in g                   # chroma capped at +-3
    assert render.grade({"luma": 140.0, "u": 128.0, "v": 128.0}, ref) == ""


def test_loudness_target_and_two_pass():
    m = {"input_i": "-28.1", "input_tp": "-9.0", "input_lra": "4.2", "input_thresh": "-38.5", "target_offset": "0.3"}
    f = render.loudnorm_pass2(m, -14.0)
    assert "I=-14.0" in f and "measured_I=-28.1" in f and "linear=true" in f


def test_graph_shape():
    by_id = {"a": {"path": "a.mp4", "luma": 120, "u": 128, "v": 128}}
    ps = [render.Piece("a", 0.5, 2.0, True), render.Piece("a", 3.0, 4.0, False)]
    args, fc = render.build_graph(ps, by_id, render.reference(list(by_id.values())), (1080, 1920), True,
                                  "card.png", 2.0, loud="loudnorm=I=-14", gain_db=14.5)
    assert "volume=14.50dB,alimiter=limit=0.794" in fc     # gain, then limit, then loudnorm
    assert args.count("-i") == 4 and "concat=n=3:v=1:a=1" in fc and fc.endswith("loudnorm=I=-14[aout]")
    assert "crop=1080:1920" in fc and "hqdn3d=0.8:0.6:0:0" in fc   # 9:16 reframe; no temporal denoise
    args, fc = render.build_graph(ps, by_id, {}, (1080, 1920), True, None, 0, video=False)
    assert "[v" not in fc and "concat=n=2:v=0:a=1" in fc


def test_captions_never_straddle_a_join():
    ws = [{"w": w, "s": i * 0.5, "e": i * 0.5 + 0.4} for i, w in enumerate(
        "one two three four five six seven eight nine ten eleven twelve".split())]
    cues = render.make_cues(ws, joins=[2.9], max_chars=26)
    for c in cues:
        assert not (c["s"] < 2.9 < c["e"])
        assert all(len(line) <= 26 for line in c["text"].split("\n")) and c["text"].count("\n") <= 1
    orphan = [{"w": w, "s": i * 0.4, "e": i * 0.4 + 0.3} for i, w in enumerate(
        "I built one that manages all of my different AI systems.".split())]
    cues2 = render.make_cues(orphan, joins=[], max_chars=26)
    assert all(" " in c["text"] for c in cues2), cues2          # no single-word cue
    srt = render.srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> ")
    ass = render.ass(cues, 720, 1280)
    assert "PlayResX: 720" in ass and "PlayResY: 1280" in ass


def test_false_start_and_retake_detection():
    from conftest import words
    w = words("I built one that manages all of my systems. I built one that manages", 0.2)
    t = analyze.false_start(w)
    assert t is not None and t < w[9]["s"]                      # ends before the second "I built"
    assert analyze.false_start(words("a clean take with nothing repeated in it at all", 0)) is None
    clips = [{"id": "1", "text": "come try a free class this weekend", "words": [1] * 7, "false_start": None},
             {"id": "2", "text": "come try a free class this weekend with us", "words": [1] * 9, "false_start": None}]
    assert analyze.find_retakes(clips) == {"1": "2"}


@pytest.mark.parametrize("t, side, want", [(1.1, "in", 0.95), (1.1, "out", 1.42), (0.5, "in", 0.5)])
def test_snap(t, side, want):
    assert analyze.snap(t, [{"w": "x", "s": 1.0, "e": 1.32}], side) == pytest.approx(want)
