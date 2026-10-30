"""The local page: form parsing, job folders, path safety and a full job run with a fake pipeline."""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hermes_editor import web

BOUNDARY = "----hermesTestBoundary7MA4YWxk"


def multipart(fields=(), files=()):
    out = b""
    for name, value in fields:
        out += (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n").encode() + value.encode() + b"\r\n"
    for name, fname, data in files:
        out += (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    return out + f"--{BOUNDARY}--\r\n".encode()


CT = f"multipart/form-data; boundary={BOUNDARY}"
# binary that contains CRLFs, a lone dash run and every byte value
BLOB = bytes(range(256)) * 3 + b"\r\n--not-the-boundary\r\n" + bytes(range(255, -1, -1))


def fake_cli(argv):
    """Stands in for hermes_editor.cli.main: prints the same steps and writes a version folder."""
    clips, out = Path(argv[0]), Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    done = sorted(int(p.name[1:]) for p in out.glob("v*"))
    v = out / f"v{(done[-1] if done else 0) + 1}"
    v.mkdir()
    for step in ("1/4 looking at the footage", "2/4 brief -> edit decision list", "3/4 rendering the video", "4/4 cutting the shorts"):
        print(step)
    names = sorted(p.name for p in clips.iterdir() if p.suffix.lower() in web.VIDEO_EXT)
    change = argv[argv.index("--change") + 1] if "--change" in argv else None
    edl = {"brief": "" if change else argv[argv.index("--brief") + 1], "target_seconds": None, "pace": "normal",
           "opening": names[0], "cuts": [{"clip": n, "in": 0.0, "out": 1.0, "pace": "normal", "reason": "r"} for n in names],
           "shorts": [{"clip": names[0], "in": 0.0, "out": 1.0, "reason": "s"}], "end_card": None,
           "history": ["rules"] + ([f"change: {change}"] if change else [])}
    (v / "edl.json").write_text(json.dumps(edl))
    (v / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(1000))
    (v / "video.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    (v / "short-1.mp4").write_bytes(b"short")
    (v / "report.json").write_text(json.dumps({"outputs": [{"file": "video.mp4", "duration": 1.0, "lufs": -14.0, "true_peak": -1.5}],
                                               "seconds": {"total": 1.0}}))
    print(f"done in 0s -> {v}")
    return 0


def body_file(tmp_path, data):
    p = tmp_path / "body.bin"
    p.write_bytes(data)
    return p


# ---- parsing -----------------------------------------------------------------------------------

def test_multipart_round_trips_text_and_binary():
    data = multipart([("brief", "keep it short\r\nand upbeat"), ("shorts", "1")],
                     [("clips", "a.mp4", BLOB), ("clips", "b.mov", b"")])
    parts = web.parse_multipart(data, web.boundary_of(CT))
    assert [(p.name, p.filename) for p in parts] == [("brief", None), ("shorts", None), ("clips", "a.mp4"), ("clips", "b.mov")]
    assert data[parts[0].start:parts[0].end] == b"keep it short\r\nand upbeat"
    assert data[parts[2].start:parts[2].end] == BLOB
    assert parts[3].start == parts[3].end


def test_boundary_and_malformed_bodies():
    assert web.boundary_of('multipart/form-data; boundary="a b"') == b"a b"
    with pytest.raises(ValueError):
        web.boundary_of("application/json")
    with pytest.raises(ValueError):
        web.parse_multipart(b"no boundary here", b"xyz")
    cut_short = multipart([], [("clips", "a.mp4", BLOB)])[:-40]
    with pytest.raises(ValueError):
        web.parse_multipart(cut_short, BOUNDARY.encode())


@pytest.mark.parametrize("given, want", [
    ("clip.MP4", "clip.MP4"),
    ("../../etc/passwd.mp4", "passwd.mp4"),
    ("C:\\Videos\\day one\\take 2.mov", "take-2.mov"),
    ("..", "clip"),
    (".mp4", "clip.mp4"),
    ("weird<>|name?.mkv", "weird-name.mkv"),
])
def test_safe_name(given, want):
    assert web.safe_name(given) == want


def test_unique_name(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "a-2.mp4").write_bytes(b"")
    assert web.unique_name(tmp_path, "a.mp4") == "a-3.mp4"
    assert web.unique_name(tmp_path, "b.mp4") == "b.mp4"


def test_parse_range():
    assert web.parse_range(None, 100) is None
    assert web.parse_range("bytes=0-", 100) == (0, 99)
    assert web.parse_range("bytes=10-19", 100) == (10, 19)
    assert web.parse_range("bytes=90-500", 100) == (90, 99)
    assert web.parse_range("bytes=-10", 100) == (90, 99)
    assert web.parse_range("items=0-1", 100) is None
    with pytest.raises(ValueError):
        web.parse_range("bytes=100-", 100)


def test_inside_refuses_escapes(tmp_path):
    base = tmp_path / "job" / "out"
    (base / "v1").mkdir(parents=True)
    (base / "v1" / "video.mp4").write_bytes(b"x")
    (tmp_path / "secret.mp4").write_bytes(b"x")
    assert web.inside(base, "v1", "video.mp4") == (base / "v1" / "video.mp4").resolve()
    assert web.inside(base, "v1", "missing.mp4") is None
    assert web.inside(base, "..", "secret.mp4") is None
    assert web.inside(base, "v1", "../../../secret.mp4") is None
    assert web.inside(base, "v1", "..\\..\\..\\secret.mp4") is None
    assert web.inside(base, "v1") is None  # a folder, not a file


# ---- jobs --------------------------------------------------------------------------------------

def test_create_job_saves_clips_and_fields(tmp_path):
    app = web.App(tmp_path / "jobs", brain="rules")
    data = multipart([("brief", "under a minute"), ("shorts", "9"), ("brain", "nonsense"), ("names", "Schalesky")],
                     [("clips", "../take 1.mp4", BLOB), ("clips", "take 1.mp4", b"two"), ("logo", "brand.PNG", b"png"),
                      ("logo", "", b""), ("other", "x.mp4", b"ignored")])
    job = app.create_job(body_file(tmp_path, data), CT)
    assert job.brief == "under a minute" and job.shorts == 3 and job.brain == "rules" and job.names == "Schalesky"
    assert job.clips == ["take-1.mp4", "take-1-2.mp4"]
    assert (job.clips_dir / "take-1.mp4").read_bytes() == BLOB
    assert job.logo == "logo.png" and (job.clips_dir / "logo.png").read_bytes() == b"png"
    assert sorted(p.name for p in job.clips_dir.iterdir()) == ["logo.png", "take-1-2.mp4", "take-1.mp4"]
    assert job.root.parent == app.jobs_dir and web.JOB_ID.match(job.id)
    assert json.loads((job.root / "job.json").read_text())["clips"] == job.clips


@pytest.mark.parametrize("files", [[], [("clips", "notes.txt", b"x")], [("clips", "a.mp4", b"x"), ("logo", "l.gif", b"x")]])
def test_create_job_rejects_bad_forms_and_leaves_nothing(tmp_path, files):
    app = web.App(tmp_path / "jobs")
    with pytest.raises(ValueError):
        app.create_job(body_file(tmp_path, multipart([("brief", "x")], files)), CT)
    assert list(app.jobs_dir.iterdir()) == []


def test_argv_for_first_cut_and_change(tmp_path):
    job = web.Job("0123456789ab", tmp_path, brief="short", brain="rules", shorts=1, logo="logo.png")
    assert "--names" not in job.argv()
    job.names = "Summit Climbing"
    a = job.argv()
    assert a[a.index("--names") + 1] == "Summit Climbing"
    assert a[0] == str(tmp_path / "clips") and a[a.index("--out") + 1] == str(tmp_path / "out")
    assert a[a.index("--brief") + 1] == "short" and a[a.index("--logo") + 1] == "logo.png" and "--change" not in a
    b = job.argv("drop the second clip")
    assert b[b.index("--change") + 1] == "drop the second clip" and "--brief" not in b


def test_job_runs_and_changes(tmp_path):
    app = web.App(tmp_path / "jobs", brain="rules", cli_main=fake_cli)
    job = app.create_job(body_file(tmp_path, multipart([("brief", "b")], [("clips", "a.mp4", b"a"), ("clips", "b.mp4", b"b")])), CT)
    app.start(job).join(5)
    s = app.status(job)
    assert s["state"] == "done", s["lines"]
    assert "3/4 rendering the video" in s["lines"]
    assert [v["name"] for v in s["versions"]] == ["v1"]
    assert {"video.mp4", "video.srt", "short-1.mp4", "edl.json", "report.json"} <= set(s["versions"][0]["files"])
    assert (job.root / "log.txt").read_text().count("1/4") == 1

    app.start(job, "drop the second clip").join(5)
    s = app.status(job)
    assert s["state"] == "done" and [v["name"] for v in s["versions"]] == ["v1", "v2"]
    assert s["versions"][1]["edl"]["history"][-1] == "change: drop the second clip"
    assert app.file_path(job, "v2", "video.mp4") is not None
    assert app.file_path(job, "v2", "../job.json") is None
    assert app.file_path(job, "v2", "work.txt") is None
    assert app.file_path(job, "..", "job.json") is None

    fresh = web.App(tmp_path / "jobs", cli_main=fake_cli)   # a server restart finds the job again
    again = fresh.get(job.id)
    assert again.state == "done" and again.clips == ["a.mp4", "b.mp4"] and again.last_change == "drop the second clip"
    assert fresh.get("../../etc") is None and fresh.get("ffffffffffff") is None


def test_a_crash_or_bad_argument_marks_the_job_failed(tmp_path):
    def crash(argv):
        print("1/4 looking at the footage")
        raise RuntimeError("ffmpeg not found")

    def bad_arg(argv):
        raise SystemExit(2)

    for main, expect in ((crash, "error: RuntimeError: ffmpeg not found"), (bad_arg, None)):
        app = web.App(tmp_path / main.__name__, cli_main=main)
        job = app.create_job(body_file(tmp_path, multipart([], [("clips", "a.mp4", b"a")])), CT)
        app.start(job).join(5)
        assert job.state == "failed"
        if expect:
            assert job.lines[-1] == expect


def test_a_second_start_while_running_is_refused(tmp_path):
    gate = threading.Event()
    app = web.App(tmp_path / "jobs", cli_main=lambda argv: gate.wait(5) and 0)
    job = app.create_job(body_file(tmp_path, multipart([], [("clips", "a.mp4", b"a")])), CT)
    t = app.start(job)
    with pytest.raises(RuntimeError):
        app.start(job, "again")
    gate.set()
    t.join(5)
    assert job.state == "done"


def chips(box):
    import re
    block = web.PAGE.split(f'data-for="{box}"', 1)[1].split("</div>", 1)[0]
    return re.findall(r'class="chip">([^<]+)<', block)


def test_suggested_phrases_work_with_the_rules_brain(inv):
    """Every chip the page offers must be understood even with no model, or the page invites an error."""
    from hermes_editor import rules
    edl = rules.plan(" ".join(c + "." for c in chips("brief")), inv, 2)
    assert edl.target_seconds == 60 and edl.end_card and len(edl.history) > 1
    assert len(chips("changeText")) >= 4
    for change in chips("changeText"):
        rules.revise(edl, change + ".", inv)   # raises if the phrase is not understood


# ---- the live server on 127.0.0.1 ---------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    app = web.App(tmp_path / "jobs", brain="rules", cli_main=fake_cli)
    srv = web.make_server(app, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_server_end_to_end(server):
    app, base = server
    assert app_bound_to_loopback(base)
    code, headers, page = call(base + "/")
    assert code == 200 and b"Make my video" in page and b'DEFAULT_BRAIN = "rules"' in page
    assert "http" not in headers["Content-Security-Policy"].replace("'self'", "")   # no outside origins

    code, _, body = call(base + "/jobs", multipart([("brief", "short")], [("clips", "a.mp4", BLOB)]), {"Content-Type": CT})
    assert code == 202
    jid = json.loads(body)["id"]
    for _ in range(100):
        s = json.loads(call(f"{base}/jobs/{jid}")[2])
        if s["state"] == "done":
            break
        time.sleep(0.05)
    assert s["state"] == "done" and s["versions"][0]["name"] == "v1"

    code, headers, data = call(f"{base}/jobs/{jid}/v1/video.mp4", headers={"Range": "bytes=4-11"})
    assert code == 206 and data == b"ftypmp42" and headers["Content-Range"].endswith("/1012")
    assert call(f"{base}/jobs/{jid}/v1/video.mp4", headers={"Range": "bytes=5000-"})[0] == 416
    assert call(f"{base}/jobs/{jid}/v1/video.srt")[1]["Content-Type"].startswith("text/plain")
    for bad in ("/v1/..%2F..%2Fjob.json", "/v1/../../job.json", "/..%2Fclips/a.mp4", "/v1/edl.json/../../job.json"):
        assert call(f"{base}/jobs/{jid}{bad}")[0] == 404, bad
    assert call(f"{base}/jobs/{jid}/clips/a.mp4")[0] == 404   # source clips are not served

    hdr = {"Content-Type": "application/json"}
    assert call(f"{base}/jobs/{jid}/change", b'{"text": ""}', hdr)[0] == 400
    assert call(f"{base}/jobs/{jid}/change", b"not json", hdr)[0] == 400
    assert call(f"{base}/jobs/{jid}/change", json.dumps({"text": "shorter"}).encode(), hdr)[0] == 202


def test_server_refuses_other_hosts_and_bad_forms(server):
    _, base = server
    assert call(base + "/", headers={"Host": "evil.example"})[0] == 403
    assert call(base + "/jobs", b"x", {"Origin": "http://evil.example", "Content-Type": CT})[0] == 403
    code, _, body = call(base + "/jobs", multipart([], [("clips", "a.txt", b"x")]), {"Content-Type": CT})
    assert code == 400 and b"not a video" in body
    assert call(base + "/jobs", b"{}", {"Content-Type": "application/json"})[0] == 400
    assert call(base + "/jobs/ffffffffffff")[0] == 404
    assert call(base + "/nope")[0] == 404


def app_bound_to_loopback(base):
    return base.startswith("http://127.0.0.1:") and web.HOST == "127.0.0.1"
