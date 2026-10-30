import json

import pytest


def words(text, start, step=0.4):
    out, t = [], start
    for w in text.split():
        out.append({"w": w, "s": round(t, 3), "e": round(t + step * 0.8, 3)})
        t += step
    return out


def clip(cid, text, start=0.5, duration=None, score=1.0, silences=None, false_start=None, superseded_by=None):
    ws = words(text, start)
    dur = duration or ws[-1]["e"] + 0.8
    return {"id": cid, "duration": dur, "width": 720, "height": 1280, "has_audio": True, "created": "",
            "silences": silences or [[0.0, start], [ws[-1]["e"], dur]], "loudness": [], "motion": [],
            "luma": 120.0, "u": 128.0, "v": 128.0, "words": ws, "text": " ".join(w["w"] for w in ws),
            "speech": [ws[0]["s"], ws[-1]["e"]], "false_start": false_start, "score": score,
            "superseded_by": superseded_by}


def curve(duration, base, bumps, hz=2):
    """A synthetic [t, v] curve: flat at `base`, with each (t, v) bump set at its sample."""
    pts = [[round(i / hz, 2), base] for i in range(int(duration * hz) + 1)]
    for t, v in bumps:
        pts[int(round(t * hz))][1] = v
    return pts


def action_clip(cid, duration=12.0, motion=(), loud=(), base_motion=1.0, base_loud=-30.0, text=""):
    """A clip with no talking: a flat motion and loudness curve with peaks where the action is."""
    c = clip(cid, text or "x", duration=duration)
    if not text:
        c.update(words=[], text="", speech=[0.0, duration], silences=[])
    c.update(motion=curve(duration, base_motion, motion), loudness=curve(duration, base_loud, loud, hz=10),
             score=0.5)
    return c


@pytest.fixture
def inv():
    return {"clips": [
        clip("a.mp4", "Hello my name is Sam and I run a climbing gym in town", score=2.0),
        clip("b.mp4", "We teach kids classes on Saturdays and adults every weeknight", score=1.0),
        clip("c.mp4", "Last week a nine year old topped the hardest wall in the building", score=4.0),
        clip("d.mp4", "Come try a free class this weekend", score=1.5),
    ]}


class FakeClient:
    """Stands in for the model: returns canned replies and records the prompts."""
    name = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r)
