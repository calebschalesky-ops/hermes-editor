"""A page on your own computer for Hermes: add clips, describe the edit, watch it cut, ask for changes.

    python -m hermes_editor.web                 # then open http://127.0.0.1:8640
    python -m hermes_editor.web --brain rules   # plan with the built-in rules, no model

The server listens on 127.0.0.1 only, so nothing else on the network can reach it. Clips you add
are copied into a job folder (hermes-jobs/<id>/clips) and the pipeline is the same one the command
line runs: each job calls hermes_editor.cli.main in a background thread and the page reads the
version folders it writes (out/v1, out/v2, ...). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import secrets
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8640
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}  # same list analyze.py reads
LOGO_EXT = {".png", ".jpg", ".jpeg", ".webp"}
SERVED = {".mp4": "video/mp4", ".srt": "text/plain; charset=utf-8", ".json": "application/json"}
MAX_BODY = 16 * 1024 ** 3
JOB_ID = re.compile(r"^[0-9a-f]{12}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,150}$")
MAX_LINES = 3000
META = ("brief", "brain", "shorts", "logo", "names", "clips", "last_change")   # kept in job.json


# ---- request parsing -------------------------------------------------------------------------

@dataclass
class Part:
    name: str
    filename: str | None
    start: int      # byte offsets of the part's data inside the body
    end: int


def boundary_of(content_type: str) -> bytes:
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("expected multipart/form-data")
    m = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type)
    if not m:
        raise ValueError("multipart body has no boundary")
    return (m.group(1) or m.group(2)).encode("latin-1")


def _disposition(headers: str) -> tuple[str, str | None]:
    for line in headers.split("\r\n"):
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-disposition":
            name = re.search(r'(?<![a-z*])name="([^"]*)"', value)
            fname = re.search(r'filename="([^"]*)"', value)
            return (name.group(1) if name else ""), (fname.group(1) if fname else None)
    raise ValueError("part has no Content-Disposition")


def parse_multipart(buf, boundary: bytes) -> list[Part]:
    """Find each part of a multipart/form-data body. `buf` is bytes or an mmap of the body on disk,
    so a few gigabytes of clips are never held in memory; data is sliced out by offset."""
    delim = b"--" + boundary
    pos = buf.find(delim)
    if pos < 0:
        raise ValueError("boundary not found in body")
    parts = []
    while True:
        pos += len(delim)
        if buf[pos:pos + 2] == b"--":
            return parts
        if buf[pos:pos + 2] != b"\r\n":
            raise ValueError("malformed multipart body")
        head_end = buf.find(b"\r\n\r\n", pos + 2)
        if head_end < 0:
            raise ValueError("malformed part headers")
        name, fname = _disposition(bytes(buf[pos + 2:head_end]).decode("utf-8", "replace"))
        start = head_end + 4
        nxt = buf.find(b"\r\n" + delim, start)
        if nxt < 0:
            raise ValueError("multipart body is cut short")
        parts.append(Part(name, fname, start, nxt))
        pos = nxt + 2


def safe_name(name: str) -> str:
    """A browser-supplied file name reduced to a plain name: no folders, no `..`, no odd characters."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_")[:100] or "clip"
    ext = re.sub(r"[^A-Za-z0-9]", "", ext)[:8]
    return f"{stem}.{ext}" if ext else stem


def unique_name(folder: Path, name: str) -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    out, n = name, 2
    while (folder / out).exists():
        out = f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}"
        n += 1
    return out


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """`Range: bytes=a-b` -> inclusive (start, end) clamped to the file, None for the whole file.
    Raises ValueError for a range outside the file."""
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not m or not (m.group(1) or m.group(2)):
        return None
    if m.group(1):
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
    else:
        start, end = max(0, size - int(m.group(2))), size - 1
    if start > end or start >= size:
        raise ValueError("range not satisfiable")
    return start, end


def inside(base: Path, *parts: str) -> Path | None:
    """base/parts if that is a file that really sits inside base, else None."""
    if any(not FILE_NAME.match(p) for p in parts):
        return None
    root = base.resolve()
    p = root.joinpath(*parts).resolve()
    return p if p.is_file() and p.is_relative_to(root) else None


# ---- jobs ------------------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    root: Path
    brief: str = ""
    brain: str = "auto"
    shorts: int = 2
    logo: str = ""
    names: str = ""             # proper nouns for the transcriber (--names)
    clips: list[str] = field(default_factory=list)
    state: str = "new"          # new, queued, running, done, failed
    lines: list[str] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0
    last_change: str = ""

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    def add_line(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        self.lines.append(line)
        del self.lines[:-MAX_LINES]
        with open(self.root / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_meta(self) -> None:
        meta = {k: getattr(self, k) for k in META}
        (self.root / "job.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def argv(self, change: str | None = None) -> list[str]:
        a = [str(self.clips_dir), "--out", str(self.out_dir), "--brain", self.brain, "--shorts", str(self.shorts)]
        if self.logo:
            a += ["--logo", self.logo]
        if self.names:
            a += ["--names", self.names]
        return a + (["--change", change] if change else ["--brief", self.brief])


def versions(out: Path) -> list[Path]:
    vs = [p for p in out.glob("v*") if p.name[1:].isdigit() and (p / "edl.json").is_file()] if out.is_dir() else []
    return sorted(vs, key=lambda p: int(p.name[1:]))


class _LineSink:
    def __init__(self, job: Job):
        self.job, self.buf = job, ""

    def write(self, s: str) -> None:
        self.buf += s.replace("\r", "\n")
        *done, self.buf = self.buf.split("\n")
        for line in done:
            self.job.add_line(line)

    def flush(self) -> None:
        if self.buf:
            self.job.add_line(self.buf)
            self.buf = ""


class _Router:
    """Stands in for sys.stdout/sys.stderr and sends what a job thread prints to that job's log.
    cli.main reports progress with print(); everything else still reaches the console."""

    def __init__(self, fallback):
        self.fallback, self.sinks = fallback, {}

    def write(self, s):
        sink = self.sinks.get(threading.get_ident())
        (sink or self.fallback).write(s)
        return len(s)

    def flush(self):
        self.fallback.flush()

    def __getattr__(self, name):
        return getattr(self.fallback, name)


def _routers() -> tuple[_Router, _Router]:
    if not isinstance(sys.stdout, _Router):
        sys.stdout = _Router(sys.stdout)
    if not isinstance(sys.stderr, _Router):
        sys.stderr = _Router(sys.stderr)
    return sys.stdout, sys.stderr


class App:
    def __init__(self, jobs_dir: Path, brain: str = "auto", cli_main=None):
        self.jobs_dir = Path(jobs_dir).resolve()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.brain = brain
        self.cli_main = cli_main
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.run_lock = threading.Lock()   # one render at a time: each one already uses every core

    # -- creating a job from the form --
    def create_job(self, body_path: Path, content_type: str) -> Job:
        jid = secrets.token_hex(6)
        job = Job(jid, self.jobs_dir / jid, brain=self.brain)
        job.clips_dir.mkdir(parents=True)
        try:
            with open(body_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                for part in parse_multipart(buf, boundary_of(content_type)):
                    self._take_part(job, buf, part)
            if not job.clips:
                raise ValueError("add at least one video clip")
        except Exception:
            shutil.rmtree(job.root, ignore_errors=True)
            raise
        job.save_meta()
        with self.lock:
            self.jobs[jid] = job
        return job

    def _take_part(self, job: Job, buf, part: Part) -> None:
        if part.filename is None:
            text = bytes(buf[part.start:part.end]).decode("utf-8", "replace").strip()
            if part.name == "brief":
                job.brief = text[:4000]
            elif part.name == "shorts" and text.isdigit():
                job.shorts = max(0, min(3, int(text)))
            elif part.name == "names":
                job.names = text[:300]
            elif part.name == "brain" and text in ("auto", "api", "cli", "rules"):
                job.brain = text
            return
        if not part.filename or part.end == part.start:   # an empty file input
            return
        name = safe_name(part.filename)
        ext = Path(name).suffix.lower()
        if part.name == "clips":
            if ext not in VIDEO_EXT:
                raise ValueError(f"{part.filename} is not a video file ({', '.join(sorted(VIDEO_EXT))})")
        elif part.name == "logo":
            if ext not in LOGO_EXT:
                raise ValueError(f"the logo must be an image ({', '.join(sorted(LOGO_EXT))})")
            name = "logo" + ext
        else:
            return
        name = unique_name(job.clips_dir, name)
        with open(job.clips_dir / name, "wb") as f:
            for i in range(part.start, part.end, 1 << 20):
                f.write(buf[i:min(i + (1 << 20), part.end)])
        if part.name == "logo":
            job.logo = name
        else:
            job.clips.append(name)

    # -- finding a job, including one from before a restart --
    def get(self, jid: str) -> Job | None:
        if not JOB_ID.match(jid or ""):
            return None
        with self.lock:
            if jid in self.jobs:
                return self.jobs[jid]
            root = self.jobs_dir / jid
            if not (root / "job.json").is_file():
                return None
            meta = json.loads((root / "job.json").read_text(encoding="utf-8"))
            job = Job(jid, root, **{k: meta[k] for k in META if k in meta})
            log = root / "log.txt"
            job.lines = log.read_text(encoding="utf-8").splitlines()[-MAX_LINES:] if log.is_file() else []
            job.state = "done" if versions(job.out_dir) else "failed"
            self.jobs[jid] = job
            return job

    # -- running the pipeline --
    def start(self, job: Job, change: str | None = None) -> threading.Thread:
        with self.lock:
            if job.state in ("queued", "running"):
                raise RuntimeError("this job is still running")
            job.state = "queued"
            job.started, job.finished = time.time(), 0.0
            if change:
                job.last_change = change
                job.save_meta()
        t = threading.Thread(target=self._run, args=(job, change), name=f"hermes-{job.id}", daemon=True)
        t.start()
        return t

    def _run(self, job: Job, change: str | None) -> None:
        job.add_line(f"queued: {'change: ' + change if change else 'first cut'}")
        with self.run_lock:
            job.state = "running"
            job.started = time.time()
            main = self.cli_main
            if main is None:
                from .cli import main
            out, err = _routers()
            sink = _LineSink(job)
            me = threading.get_ident()
            out.sinks[me] = err.sinks[me] = sink
            try:
                code = main(job.argv(change))
                ok = not code
            except SystemExit as e:   # argparse reports a bad argument this way
                ok = e.code in (0, None)
            except Exception as e:
                sink.flush()
                job.add_line(f"error: {type(e).__name__}: {e}")
                ok = False
            finally:
                out.sinks.pop(me, None)
                err.sinks.pop(me, None)
                sink.flush()
            job.state = "done" if ok else "failed"
            job.finished = time.time()

    # -- what the page reads --
    def status(self, job: Job) -> dict:
        vs = []
        for v in versions(job.out_dir):
            info = {"name": v.name, "edl": json.loads((v / "edl.json").read_text(encoding="utf-8")),
                    "report": None,
                    "files": sorted(p.name for p in v.iterdir() if p.is_file() and p.suffix in SERVED)}
            if (v / "report.json").is_file():
                info["report"] = json.loads((v / "report.json").read_text(encoding="utf-8"))
            vs.append(info)
        end = job.finished or (time.time() if job.state in ("queued", "running") else job.started)
        return {"id": job.id, "state": job.state, "brief": job.brief, "brain": job.brain, "shorts": job.shorts,
                "logo": job.logo, "clips": job.clips, "last_change": job.last_change, "lines": job.lines[-400:],
                "elapsed": round(max(0.0, end - job.started), 1) if job.started else 0, "versions": vs,
                "folder": str(job.root)}

    def file_path(self, job: Job, version: str, name: str) -> Path | None:
        if not re.fullmatch(r"v\d+", version or "") or Path(name or "").suffix not in SERVED:
            return None
        return inside(job.out_dir, version, name)


# ---- the server ------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "hermes-web"
    app: App = None   # set by make_server

    def log_message(self, fmt, *args):
        pass

    # Only answer pages asked for by name as this machine, so a web page elsewhere cannot rebind a
    # hostname onto 127.0.0.1 and drive the server.
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost"):
            return False
        origin = self.headers.get("Origin")
        return origin is None or re.fullmatch(r"http://(127\.0\.0\.1|localhost)(:\d+)?", origin) is not None

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _error(self, code: int, msg: str) -> None:
        self._json(code, {"error": msg})

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "open this page as 127.0.0.1 or localhost")
        path = self.path.split("?", 1)[0]
        if path == "/":
            page = PAGE.replace("__DEFAULT_BRAIN__", self.app.brain)
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8",
                              {"Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; "
                               "script-src 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self'"})
        m = re.fullmatch(r"/jobs/([0-9a-f]{12})", path)
        if m:
            job = self.app.get(m.group(1))
            return self._json(200, self.app.status(job)) if job else self._error(404, "no such job")
        m = re.fullmatch(r"/jobs/([0-9a-f]{12})/(v\d+)/([^/]+)", path)
        if m:
            job = self.app.get(m.group(1))
            p = self.app.file_path(job, m.group(2), m.group(3)) if job else None
            return self._file(p) if p else self._error(404, "no such file")
        self._error(404, "not found")

    def _file(self, p: Path) -> None:
        size = p.stat().st_size
        try:
            rng = parse_range(self.headers.get("Range"), size)
        except ValueError:
            return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
        start, end = rng or (0, size - 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", SERVED[p.suffix])
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if p.suffix == ".srt":
            self.send_header("Content-Disposition", f'inline; filename="{p.name}"')
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(p, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass   # the video player moved on to another range

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not self._host_ok():
            if 0 < length <= 1 << 20:   # read what was sent, or Windows resets the connection on close
                self.rfile.read(length)
            return self._error(403, "open this page as 127.0.0.1 or localhost")
        if length <= 0 or length > MAX_BODY:
            return self._error(411 if length <= 0 else 413, "the request needs a body of a sane size")
        if path == "/jobs":
            return self._new_job(length)
        m = re.fullmatch(r"/jobs/([0-9a-f]{12})/change", path)
        if m:
            if length > 64 * 1024:
                return self._error(413, "keep the change to a few sentences")
            raw = self.rfile.read(length)
            job = self.app.get(m.group(1))
            if not job:
                return self._error(404, "no such job")
            try:
                text = str(json.loads(raw).get("text", "")).strip()[:2000]
            except (ValueError, AttributeError):
                return self._error(400, "expected JSON like {\"text\": \"...\"}")
            if not text:
                return self._error(400, "say what to change")
            if not versions(job.out_dir):
                return self._error(409, "there is no finished cut to change yet")
            try:
                self.app.start(job, text)
            except RuntimeError as e:
                return self._error(409, str(e))
            return self._json(202, {"id": job.id})
        self._error(404, "not found")

    def _new_job(self, length: int) -> None:
        incoming = self.app.jobs_dir / f".incoming-{secrets.token_hex(6)}"
        try:
            with open(incoming, "wb") as f:
                left = length
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        return self._error(400, "the browser stopped sending before the clips arrived")
                    f.write(chunk)
                    left -= len(chunk)
            try:
                job = self.app.create_job(incoming, self.headers.get("Content-Type", ""))
            except ValueError as e:
                return self._error(400, str(e))
        finally:
            incoming.unlink(missing_ok=True)
        self.app.start(job)
        self._json(202, {"id": job.id})


def make_server(app: App, port: int = PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((HOST, port), handler)
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hermes-web", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--jobs", type=Path, default=Path("hermes-jobs"), help="where job folders are kept")
    ap.add_argument("--brain", choices=["auto", "api", "cli", "rules"], default=os.environ.get("HERMES_BRAIN", "auto"),
                    help="who plans the edit by default (the page can override it per job)")
    a = ap.parse_args(argv)
    app = App(a.jobs, a.brain)
    try:
        srv = make_server(app, a.port)
    except OSError as e:
        print(f"could not listen on {HOST}:{a.port} ({e}); try --port {a.port + 1}", file=sys.stderr)
        return 1
    print(f"Hermes is running at http://{HOST}:{srv.server_address[1]}  (jobs in {app.jobs_dir})", flush=True)
    print("Only this computer can open it. Press Ctrl+C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


# ---- the page --------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hermes video editor</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%231d4f43'/%3E%3Cpath d='M9 8v16M23 8v16M9 16h14' stroke='%23f3c969' stroke-width='3.2' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
:root{
  --bg:#f4f2ec; --bg2:#ebe7dd; --card:#fffdf8; --ink:#1a1d1f; --ink2:#43484d; --muted:#767b80; --line:#e2ddd1;
  --accent:#1d4f43; --accent-ink:#fff; --accent-soft:#e3ede8; --gold:#b9882c; --gold-soft:#f6ecd5;
  --bad:#9b2c2c; --bad-soft:#f8e5e3; --shadow:0 1px 2px rgba(30,25,15,.06),0 8px 24px rgba(30,25,15,.06);
  --r:16px; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#121514; --bg2:#181c1b; --card:#1b201f; --ink:#ecebe6; --ink2:#c4c6c2; --muted:#8d938f; --line:#2b3230;
  --accent:#5fb39a; --accent-ink:#0d1714; --accent-soft:#1d2e29; --gold:#e0b45a; --gold-soft:#2c2617;
  --bad:#f08b80; --bad-soft:#331d1b; --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);
}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  background-image:radial-gradient(1200px 400px at 50% -200px,var(--bg2),transparent)}
a{color:var(--accent)}
.wrap{max-width:920px;margin:0 auto;padding:0 20px}
header{padding:28px 0 8px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.mark{width:42px;height:42px;border-radius:12px;background:#1d4f43;display:grid;place-items:center;flex:none;box-shadow:var(--shadow)}
.brand{flex:1;min-width:0}
.brand h1{font-size:22px;margin:0;letter-spacing:-.01em}
.brand p{margin:0;color:var(--muted);font-size:14px}
.local{margin-left:auto;font-size:13px;color:var(--ink2);background:var(--card);border:1px solid var(--line);
  border-radius:999px;padding:6px 12px;display:flex;align-items:center;gap:8px}
.local i{width:8px;height:8px;border-radius:50%;background:#3a9a6e;box-shadow:0 0 0 3px rgba(58,154,110,.18)}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:24px;margin:18px 0}
.step{display:flex;gap:14px;align-items:flex-start;margin-bottom:6px}
.num{flex:none;width:28px;height:28px;border-radius:50%;background:var(--accent-soft);color:var(--accent);
  font-weight:700;font-size:14px;display:grid;place-items:center;margin-top:1px}
.step h2{font-size:17px;margin:0}
.step p{margin:2px 0 0;color:var(--muted);font-size:14px}
.block{margin:0 0 26px 42px}
.block:last-child{margin-bottom:0}
.drop{border:1.5px dashed var(--line);border-radius:14px;padding:26px 16px;text-align:center;cursor:pointer;
  transition:border-color .15s,background .15s;background:var(--bg)}
.drop:hover,.drop.over{border-color:var(--accent);background:var(--accent-soft)}
.drop strong{display:block;font-size:16px}
.drop span{color:var(--muted);font-size:14px}
.files{list-style:none;margin:12px 0 0;padding:0;display:grid;gap:8px}
.files li{display:flex;align-items:center;gap:10px;border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--card)}
.files .ico{width:30px;height:30px;border-radius:8px;background:var(--gold-soft);color:var(--gold);display:grid;place-items:center;flex:none}
.files .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px}
.files .sz{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
.x{border:0;background:transparent;color:var(--muted);font-size:20px;line-height:1;cursor:pointer;padding:2px 6px;border-radius:6px}
.x:hover{background:var(--bg2);color:var(--ink)}
textarea{width:100%;min-height:96px;resize:vertical;font:inherit;color:var(--ink);background:var(--bg);
  border:1px solid var(--line);border-radius:12px;padding:12px 14px;outline:none}
textarea:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--ink2);border-radius:999px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.opts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.opt label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px}
.seg{display:flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--bg)}
.seg button{flex:1;border:0;background:transparent;font:inherit;font-size:14px;padding:8px 0;cursor:pointer;color:var(--ink2)}
.seg button.on{background:var(--accent);color:var(--accent-ink);font-weight:600}
.wide-opt{grid-column:1/-1}
input[type=text]{width:100%;font:inherit;font-size:14px;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:9px 12px;outline:none}
input[type=text]:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
select{width:100%;font:inherit;font-size:14px;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:8px 10px;outline:none}
.logo-pick{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:10px;padding:5px 6px 5px 10px;background:var(--bg);min-height:40px}
.logo-pick span{flex:1;font-size:14px;color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.logo-pick img{width:28px;height:28px;object-fit:contain;border-radius:6px;background:var(--card)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:10px;border:0;border-radius:12px;font:inherit;font-weight:600;
  font-size:16px;padding:13px 22px;cursor:pointer;background:var(--accent);color:var(--accent-ink);transition:transform .05s,opacity .15s}
.btn:active{transform:translateY(1px)}
.btn[disabled]{opacity:.45;cursor:not-allowed}
.btn.ghost{background:transparent;color:var(--accent);border:1px solid var(--line)}
.btn.small{font-size:14px;padding:9px 14px;border-radius:10px}
.go{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-left:42px}
.hint{color:var(--muted);font-size:14px}
.err{background:var(--bad-soft);color:var(--bad);border-radius:12px;padding:12px 14px;font-size:14px;margin-top:14px}
.hide{display:none!important}
.stages{list-style:none;margin:6px 0 0;padding:0;display:grid;gap:4px}
.stages li{display:flex;align-items:center;gap:12px;padding:10px 4px;color:var(--muted)}
.stages .dot{width:22px;height:22px;border-radius:50%;border:2px solid var(--line);flex:none;display:grid;place-items:center}
.stages li.on{color:var(--ink);font-weight:600}
.stages li.on .dot{border-color:var(--accent);border-top-color:transparent;animation:spin .9s linear infinite}
.stages li.ok{color:var(--ink2)}
.stages li.ok .dot{border-color:var(--accent);background:var(--accent)}
.stages li.ok .dot::after{content:"";width:6px;height:10px;border:solid var(--accent-ink);border-width:0 2px 2px 0;transform:translateY(-1px) rotate(45deg)}
@keyframes spin{to{transform:rotate(360deg)}}
.prog-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
.prog-head h2{margin:0;font-size:19px}
.clock{font-variant-numeric:tabular-nums;color:var(--muted);font-size:14px}
.now{margin:8px 0 0;color:var(--ink2);font-size:14px;font-family:var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
details.log{margin-top:14px}
details.log summary{cursor:pointer;color:var(--muted);font-size:14px}
pre{margin:10px 0 0;max-height:280px;overflow:auto;background:var(--bg);border:1px solid var(--line);border-radius:10px;
  padding:12px;font:12.5px/1.5 var(--mono);color:var(--ink2);white-space:pre-wrap;word-break:break-word}
.tabs{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.tab{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:5px 14px;font:inherit;font-size:14px;cursor:pointer;color:var(--ink2)}
.tab.on{background:var(--ink);color:var(--card);border-color:var(--ink)}
.res-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.res-head h2{margin:0;font-size:20px}
.player{background:#0c0d0e;border-radius:14px;overflow:hidden;display:flex;justify-content:center}
.player video{display:block;max-width:100%;max-height:72vh;width:auto;height:auto}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.stat b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:12.5px}
.links{display:flex;gap:10px;flex-wrap:wrap}
.lnk{display:inline-flex;align-items:center;gap:6px;font-size:14px;text-decoration:none;border:1px solid var(--line);border-radius:10px;padding:7px 12px;background:var(--card);color:var(--ink2)}
.lnk:hover{border-color:var(--accent);color:var(--accent)}
h3{font-size:15px;margin:26px 0 12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.shorts{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px}
.short video{width:100%;aspect-ratio:9/16;background:#0c0d0e;border-radius:12px;display:block;object-fit:contain}
.short .cap{font-size:13.5px;color:var(--ink2);margin:8px 0 6px}
.short .cap b{color:var(--ink)}
.edl{list-style:none;margin:0;padding:0;display:grid;gap:10px}
.cut{display:grid;grid-template-columns:34px 1fr;gap:12px;border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:var(--bg)}
.cut .n{width:30px;height:30px;border-radius:8px;background:var(--ink);color:var(--card);display:grid;place-items:center;font-weight:700;font-size:14px}
.cut .top{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center}
.cut .clip{font-weight:600;font-size:14.5px;word-break:break-all}
.cut .tc{font-family:var(--mono);font-size:13px;color:var(--ink2)}
.pill{font-size:12px;border-radius:999px;padding:2px 9px;background:var(--accent-soft);color:var(--accent);font-weight:600}
.pill.gold{background:var(--gold-soft);color:var(--gold)}
.cut .why{margin:6px 0 0;color:var(--ink2);font-size:14px}
.bar{position:relative;height:6px;border-radius:3px;background:var(--line);margin-top:9px;overflow:hidden}
.bar i{position:absolute;top:0;bottom:0;background:var(--accent);border-radius:3px}
.note{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:12px 14px;font-size:14px;color:var(--ink2)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:14px}
.kv dt{color:var(--muted)}
.kv dd{margin:0}
.change{display:grid;gap:10px}
.change .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
footer{color:var(--muted);font-size:13px;padding:10px 0 40px;text-align:center}
@media (max-width:640px){
  .wrap{padding:0 16px}
  .card{padding:18px}
  .block,.go{margin-left:0}
  .opts{grid-template-columns:1fr}
  .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .brand{flex-basis:calc(100% - 60px)}
  .local{margin-left:0}
  .btn.wide{width:100%}
  .shorts{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="mark"><svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true"><path d="M8 6v20M24 6v20M8 16h16" stroke="#f3c969" stroke-width="3.6" stroke-linecap="round" fill="none"/></svg></div>
  <div class="brand"><h1>Hermes</h1><p>Your raw clips in. A finished video, shorts and captions out.</p></div>
  <div class="local" title="The page and your clips stay on this computer"><i></i>Running on this computer</div>
</header>

<form id="form" class="card" autocomplete="off">
  <div class="step"><div class="num">1</div><div><h2>Add your clips</h2><p>The takes you filmed, in any order. They are copied into a job folder; the originals are never changed.</p></div></div>
  <div class="block">
    <div id="drop" class="drop" tabindex="0" role="button" aria-label="Choose video clips">
      <strong>Drop video clips here</strong><span>or click to choose them &middot; .mp4 .mov .m4v .mkv .webm .avi</span>
    </div>
    <input id="clips" type="file" accept="video/*,.mp4,.mov,.m4v,.mkv,.webm,.avi" multiple class="hide">
    <ul id="files" class="files"></ul>
  </div>

  <div class="step"><div class="num">2</div><div><h2>Describe the edit</h2><p>In your own words. Leave it empty and Hermes uses sensible defaults.</p></div></div>
  <div class="block">
    <textarea id="brief" placeholder="Keep it under a minute, open on the biggest moment, upbeat pace. End card that says &quot;Summit Climbing&quot;."></textarea>
    <div class="chips" data-for="brief">
      <button type="button" class="chip">Keep it under a minute</button>
      <button type="button" class="chip">Open on the biggest moment</button>
      <button type="button" class="chip">Upbeat pace</button>
      <button type="button" class="chip">Relaxed pace</button>
      <button type="button" class="chip">End card that says "Your Name"</button>
    </div>
  </div>

  <div class="step"><div class="num">3</div><div><h2>Options</h2><p>Short vertical clips, a logo for the end card, and who plans the cut.</p></div></div>
  <div class="block opts">
    <div class="opt"><label>Vertical shorts</label>
      <div class="seg" id="shorts"><button type="button" data-v="0">0</button><button type="button" data-v="1">1</button><button type="button" data-v="2" class="on">2</button><button type="button" data-v="3">3</button></div></div>
    <div class="opt"><label>Logo (optional)</label>
      <div class="logo-pick"><img id="logoimg" class="hide" alt=""><span id="logoname">No logo</span>
        <button type="button" class="btn ghost small" id="logobtn">Choose</button></div>
      <input id="logo" type="file" accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp" class="hide"></div>
    <div class="opt"><label for="brain">Plans the edit</label>
      <select id="brain">
        <option value="auto">Claude (best available)</option>
        <option value="cli">Claude Code on this computer</option>
        <option value="api">Claude API key</option>
        <option value="rules">Built-in rules, no model</option>
      </select></div>
    <div class="opt wide-opt"><label for="names">Names to spell right in the captions (optional)</label>
      <input id="names" type="text" maxlength="300" placeholder="Summit Climbing, Coach Dana"></div>
  </div>

  <div class="go">
    <button class="btn wide" id="start" type="submit" disabled>Make my video</button>
    <span class="hint" id="hint">Add at least one clip to start.</span>
  </div>
  <div id="formerr" class="err hide"></div>
</form>

<section id="progress" class="card hide" aria-live="polite">
  <div class="prog-head"><h2 id="ptitle">Editing your video</h2><span class="clock" id="clock">0:00</span></div>
  <ul class="stages" id="stages">
    <li><span class="dot"></span>Looking at the footage</li>
    <li><span class="dot"></span>Planning the edit</li>
    <li><span class="dot"></span>Rendering the video</li>
    <li><span class="dot"></span>Cutting the shorts</li>
  </ul>
  <p class="now" id="now"></p>
  <div id="runerr" class="err hide"></div>
  <details class="log"><summary>Show every step</summary><pre id="log"></pre></details>
</section>

<section id="result" class="card hide">
  <div class="res-head"><h2>Your video</h2><div class="tabs" id="tabs"></div></div>
  <div class="player"><video id="main" controls playsinline preload="metadata"></video></div>
  <div class="stats" id="stats"></div>
  <div class="links" id="links"></div>

  <div id="shortsWrap"><h3>Shorts</h3><div class="shorts" id="shortsList"></div></div>

  <h3>Edit decisions</h3>
  <ol class="edl" id="edl"></ol>
  <div id="extra"></div>

  <h3>Ask for a change</h3>
  <div class="change">
    <textarea id="changeText" placeholder="Slow down the middle, drop the second clip."></textarea>
    <div class="chips" data-for="changeText">
      <button type="button" class="chip">Make it under 30 seconds</button>
      <button type="button" class="chip">Slow down the middle</button>
      <button type="button" class="chip">Drop the second clip</button>
      <button type="button" class="chip">Open on clip 2</button>
      <button type="button" class="chip">No end card</button>
    </div>
    <div class="row"><button class="btn" id="changeBtn" type="button">Re-cut it</button><span class="hint" id="changeHint"></span></div>
    <div id="changeErr" class="err hide"></div>
  </div>
</section>

<section id="again" class="hide" style="text-align:center;margin:8px 0 24px">
  <button class="btn ghost" id="newBtn" type="button">Start a new video</button>
</section>

<footer id="foot">Clips, renders and the edit plan stay in the job folder on this computer. When Claude plans the edit, it is sent the brief and what was said in the clips, not the video.</footer>
</div>

<script>
(() => {
const $ = id => document.getElementById(id);
const DEFAULT_BRAIN = "__DEFAULT_BRAIN__";
let picked = [], logoFile = null, shorts = 2, job = null, timer = null, shownVersion = null, lastState = "";

const fmtSize = b => b > 1e9 ? (b/1e9).toFixed(2)+" GB" : b > 1e6 ? (b/1e6).toFixed(1)+" MB" : Math.max(1, Math.round(b/1e3))+" KB";
const tc = s => { s = Math.max(0, +s || 0); const m = Math.floor(s/60); return m + ":" + (s - m*60).toFixed(1).padStart(4, "0"); };
const clock = s => { s = Math.round(s); return Math.floor(s/60) + ":" + String(s%60).padStart(2, "0"); };
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const VIDEO = /\.(mp4|mov|m4v|mkv|webm|avi)$/i;

$("brain").value = DEFAULT_BRAIN;

// ---- form ----
function refreshFiles() {
  const ul = $("files"); ul.innerHTML = "";
  picked.forEach((f, i) => {
    const li = el("li");
    const ico = el("span", "ico"); ico.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="13" height="14" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></svg>';
    const x = el("button", "x", "×"); x.type = "button"; x.title = "Remove"; x.onclick = () => { picked.splice(i, 1); refreshFiles(); };
    li.append(ico, el("span", "nm", f.name), el("span", "sz", fmtSize(f.size)), x);
    ul.append(li);
  });
  $("start").disabled = !picked.length;
  const total = picked.reduce((a, f) => a + f.size, 0);
  $("hint").textContent = picked.length ? `${picked.length} clip${picked.length > 1 ? "s" : ""}, ${fmtSize(total)}.` : "Add at least one clip to start.";
}
function addFiles(list) {
  const bad = [];
  for (const f of list) {
    if (!VIDEO.test(f.name)) { bad.push(f.name); continue; }
    if (!picked.some(p => p.name === f.name && p.size === f.size)) picked.push(f);
  }
  showErr($("formerr"), bad.length ? "Not a video file: " + bad.join(", ") : "");
  refreshFiles();
}
$("drop").onclick = () => $("clips").click();
$("drop").onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("clips").click(); } };
$("clips").onchange = e => { addFiles(e.target.files); e.target.value = ""; };
["dragenter", "dragover"].forEach(t => $("drop").addEventListener(t, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave", "drop"].forEach(t => $("drop").addEventListener(t, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", e => addFiles(e.dataTransfer.files));

document.querySelectorAll(".chips").forEach(box => box.addEventListener("click", e => {
  if (!e.target.classList.contains("chip")) return;
  const ta = $(box.dataset.for), t = e.target.textContent.trim();
  ta.value = ta.value.trim() ? ta.value.trim().replace(/[.,;]?$/, ". ") + t + "." : t + ".";
  ta.focus();
}));
$("shorts").onclick = e => {
  const b = e.target.closest("button"); if (!b) return;
  shorts = +b.dataset.v;
  [...$("shorts").children].forEach(c => c.classList.toggle("on", c === b));
};
$("logobtn").onclick = () => $("logo").click();
$("logo").onchange = e => {
  logoFile = e.target.files[0] || null;
  $("logoname").textContent = logoFile ? logoFile.name : "No logo";
  if (logoFile) { $("logoimg").src = URL.createObjectURL(logoFile); $("logoimg").classList.remove("hide"); }
  else $("logoimg").classList.add("hide");
};

function showErr(box, msg) { box.textContent = msg || ""; box.classList.toggle("hide", !msg); }

$("form").onsubmit = async e => {
  e.preventDefault();
  if (!picked.length) return;
  const fd = new FormData();
  fd.append("brief", $("brief").value.trim());
  fd.append("shorts", String(shorts));
  fd.append("brain", $("brain").value);
  fd.append("names", $("names").value.trim());
  picked.forEach(f => fd.append("clips", f, f.name));
  if (logoFile) fd.append("logo", logoFile, logoFile.name);
  $("start").disabled = true; $("start").textContent = "Adding clips…";
  showErr($("formerr"), "");
  try {
    const r = await fetch("/jobs", {method: "POST", body: fd});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    location.hash = "job=" + j.id;
    openJob(j.id);
  } catch (err) {
    showErr($("formerr"), "Could not start: " + err.message);
    $("start").disabled = false;
  } finally { $("start").textContent = "Make my video"; }
};

// ---- job ----
function openJob(id) {
  job = {id}; shownVersion = null; lastState = "";
  $("form").classList.add("hide");
  $("again").classList.remove("hide");
  poll();
}
async function poll() {
  clearTimeout(timer);
  let s;
  try {
    const r = await fetch("/jobs/" + job.id);
    if (r.status === 404) { location.hash = ""; return reset(); }
    s = await r.json();
  } catch (err) { timer = setTimeout(poll, 2000); return; }
  job = s;
  const busy = s.state === "queued" || s.state === "running";
  renderProgress(s, busy);
  if (s.versions.length && (!busy || shownVersion)) renderResult(s, busy);
  if (busy) timer = setTimeout(poll, 1000);
  lastState = s.state;
}
function renderProgress(s, busy) {
  const show = busy || s.state === "failed";
  $("progress").classList.toggle("hide", !show);
  if (!show) return;
  $("ptitle").textContent = s.state === "failed" ? "The edit stopped" : s.state === "queued" ? "Waiting for the previous edit to finish" :
    (s.last_change && s.versions.length ? "Re-cutting: " + s.last_change : "Editing your video");
  $("clock").textContent = clock(s.elapsed);
  // stage from the CLI's own "n/4" lines, counted from the start of this run
  let from = 0; s.lines.forEach((l, i) => { if (l.startsWith("queued:")) from = i; });
  const run = s.lines.slice(from);
  let stage = -1; run.forEach(l => { const m = /^([1-4])\/4 /.exec(l); if (m) stage = +m[1] - 1; });
  [...$("stages").children].forEach((li, i) => {
    li.className = s.state === "failed" ? (i < stage ? "ok" : "") : i < stage ? "ok" : i === stage ? "on" : "";
  });
  const last = run.filter(l => !l.startsWith("queued:")).slice(-1)[0] || "";
  $("now").textContent = last.trim();
  $("log").textContent = s.lines.join("\n");
  const pre = $("log"); pre.scrollTop = pre.scrollHeight;
  const why = (run.filter(l => l.startsWith("error:")).slice(-1)[0] || "").replace(/^error:\s*(\w+Error:\s*)?/, "");
  showErr($("runerr"), s.state === "failed" ? (why ? why.charAt(0).toUpperCase() + why.slice(1) + "." : "Hermes could not finish this run.") +
    (s.versions.length ? " The last finished version is still below." : "") + " Job folder: " + s.folder : "");
  if (s.state === "failed") $("progress").querySelector("details").open = true;
}
function renderResult(s, busy) {
  $("result").classList.remove("hide");
  const names = s.versions.map(v => v.name);
  const latest = names[names.length - 1];
  if (!shownVersion || (!busy && lastState && lastState !== s.state) || !names.includes(shownVersion)) shownVersion = latest;
  const tabs = $("tabs"); tabs.innerHTML = "";
  if (names.length > 1) names.forEach(n => {
    const b = el("button", "tab" + (n === shownVersion ? " on" : ""), n === latest ? n + " · latest" : n);
    b.type = "button"; b.onclick = () => { shownVersion = n; renderResult(job, false); };
    tabs.append(b);
  });
  const v = s.versions.find(x => x.name === shownVersion);
  const base = `/jobs/${s.id}/${v.name}/`;
  const has = f => v.files.includes(f);
  const main = $("main");
  if (has("video.mp4")) { if (main.dataset.src !== base + "video.mp4") { main.src = base + "video.mp4#t=0.3"; main.dataset.src = base + "video.mp4"; } main.parentElement.classList.remove("hide"); }
  else main.parentElement.classList.add("hide");

  const outs = (v.report && v.report.outputs) || [];
  const vid = outs.find(o => o.file === "video.mp4") || {};
  const stats = $("stats"); stats.innerHTML = "";
  const stat = (b, span) => { const d = el("div", "stat"); d.append(el("b", null, b), el("span", null, span)); stats.append(d); };
  if (vid.duration != null) stat(tc(vid.duration), "length");
  if (vid.lufs != null) stat(vid.lufs.toFixed(1) + " LUFS", "loudness");
  if (vid.true_peak != null) stat(vid.true_peak.toFixed(1) + " dB", "true peak");
  if (v.report && v.report.seconds) stat(clock(v.report.seconds.total), "time to make");
  stats.classList.toggle("hide", !stats.children.length);

  const links = $("links"); links.innerHTML = "";
  const link = (f, label) => { if (!has(f)) return; const a = el("a", "lnk", label); a.href = base + f; if (!f.endsWith(".mp4")) a.target = "_blank"; else a.download = f; links.append(a); };
  link("video.mp4", "Save video.mp4"); link("video.srt", "Captions (video.srt)"); link("edl.json", "Edit list (edl.json)"); link("report.json", "Measurements (report.json)");

  const sl = $("shortsList"); sl.innerHTML = "";
  const edl = v.edl || {};
  (edl.shorts || []).forEach((sh, i) => {
    const f = `short-${i + 1}.mp4`; if (!has(f)) return;
    const box = el("div", "short");
    const vv = el("video"); vv.controls = true; vv.playsInline = true; vv.preload = "metadata"; vv.src = base + f + "#t=0.3";  // show a real frame, not a black first one
    const o = outs.find(x => x.file === f) || {};
    const cap = el("div", "cap"); cap.append(el("b", null, `Short ${i + 1}`), document.createTextNode(` · ${tc(o.duration != null ? o.duration : sh.out - sh.in)} · from ${sh.clip}`));
    const l = el("div", "links");
    const a1 = el("a", "lnk", "Save"); a1.href = base + f; a1.download = f; l.append(a1);
    if (has(`short-${i + 1}.srt`)) { const a2 = el("a", "lnk", "Captions"); a2.href = base + `short-${i + 1}.srt`; a2.target = "_blank"; l.append(a2); }
    box.append(vv, cap);
    if (sh.reason) box.append(el("div", "cap", sh.reason));
    box.append(l);
    sl.append(box);
  });
  $("shortsWrap").classList.toggle("hide", !sl.children.length);

  // the EDL as a readable list
  const ol = $("edl"); ol.innerHTML = "";
  const durOf = {}; (edl.cuts || []).forEach(c => durOf[c.clip] = Math.max(durOf[c.clip] || 0, c.out));
  (edl.cuts || []).forEach((c, i) => {
    const li = el("li", "cut");
    li.append(el("div", "n", String(i + 1)));
    const body = el("div");
    const top = el("div", "top");
    top.append(el("span", "clip", c.clip), el("span", "tc", `${tc(c.in)} → ${tc(c.out)} · ${(c.out - c.in).toFixed(1)} s`));
    if (c.pace) top.append(el("span", "pill", c.pace + " pace"));
    if (edl.opening === c.clip && i === 0) top.append(el("span", "pill gold", "opening"));
    body.append(top);
    if (c.reason) body.append(el("p", "why", c.reason));
    ol.append(li); li.append(body);
  });
  const extra = $("extra"); extra.innerHTML = "";
  const used = new Set((edl.cuts || []).map(c => c.clip));
  const left = (s.clips || []).filter(c => !used.has(c));
  const dl = el("dl", "kv");
  const kv = (k, val) => { dl.append(el("dt", null, k), el("dd", null, val)); };
  if (edl.target_seconds) kv("Target length", `under ${edl.target_seconds} s`);
  if (edl.end_card) kv("End card", [edl.end_card.title, edl.end_card.subtitle].filter(Boolean).join(" — ") + (edl.end_card.logo ? " (with logo)" : "") || "logo only");
  if (left.length) kv("Left out", left.join(", "));
  kv("Brief", edl.brief || "(defaults)");
  if (edl.history && edl.history.length) kv("History", edl.history.join(" → "));
  if (dl.children.length) { const n = el("div", "note"); n.style.marginTop = "12px"; n.append(dl); extra.append(n); }
  if (edl.notes) { const n = el("div", "note", edl.notes); n.style.marginTop = "10px"; extra.append(n); }

  $("changeBtn").disabled = busy;
  $("changeHint").textContent = busy ? "Re-cutting now…" : `Changes apply to ${latest} and make ${"v" + (+latest.slice(1) + 1)}.`;
}
$("changeBtn").onclick = async () => {
  const text = $("changeText").value.trim();
  if (!text) return showErr($("changeErr"), "Say what to change, in your own words.");
  showErr($("changeErr"), "");
  $("changeBtn").disabled = true;
  try {
    const r = await fetch(`/jobs/${job.id}/change`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text})});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    $("changeText").value = "";
    shownVersion = null; lastState = "";
    $("result").classList.add("hide");
    window.scrollTo(0, 0);
    poll();
  } catch (err) { showErr($("changeErr"), "Could not start the change: " + err.message); $("changeBtn").disabled = false; }
};
function reset() {
  clearTimeout(timer); job = null; picked = []; refreshFiles();
  ["progress", "result", "again"].forEach(id => $(id).classList.add("hide"));
  $("form").classList.remove("hide");
  $("main").removeAttribute("src"); $("main").dataset.src = "";
}
$("newBtn").onclick = () => { location.hash = ""; reset(); window.scrollTo({top: 0}); };
const fromHash = () => { const m = /job=([0-9a-f]{12})/.exec(location.hash); if (m && (!job || job.id !== m[1])) openJob(m[1]); };
addEventListener("hashchange", fromHash);
fromHash();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
