"""Guards that fail the build if the repo learns to publish, or leaks something personal."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "hermes_editor"

# Anything that could upload, post, send or sign in. The only network the package may touch is
# the optional Anthropic API client in brain.py, which sends a prompt and receives an EDL.
FORBIDDEN = [
    r"\bupload", r"\bpublish", r"\bsign[_ ]?in\b", r"\blogin\b", r"\boauth",
    r"\brequests\b", r"\burllib", r"\bhttp\.client", r"\bhttpx\b", r"\baiohttp", r"\bsocket\b",
    r"\bsmtplib", r"\bftplib", r"\bparamiko", r"\bboto3", r"\bgoogleapiclient", r"\byoutube",
    r"\btiktok", r"\binstagram", r"\bselenium", r"\bplaywright", r"\bwebbrowser", r"\btweepy",
    r"\.post\(", r"\.put\(",
]


def test_package_has_no_upload_or_publish_code():
    hits = []
    for f in PKG.rglob("*.py"):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for pat in FORBIDDEN:
                if re.search(pat, line, re.I):
                    hits.append(f"{f.name}:{n}: {pat}: {line.strip()}")
    assert not hits, "publish/network code found:\n" + "\n".join(hits)


def test_only_brain_imports_a_network_client():
    for f in PKG.rglob("*.py"):
        if f.name != "brain.py":
            assert "anthropic" not in f.read_text(encoding="utf-8"), f.name


def test_the_guard_itself_catches_things():
    bad = "import requests\nrequests.post('https://example.com/upload', files=f)\n"
    assert any(re.search(p, bad, re.I) for p in FORBIDDEN)


PERSONAL = [
    (r"[A-Za-z]:[\\/]+Users[\\/]", "Windows home path"),
    (r"/(?:Users|home)/[a-z]", "home path"),
    (r"AppData|OneDrive", "personal folder"),
    (r"\blocal_[0-9a-f]{6,}", "session id"),
    (r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)(?!anthropic\.com)[A-Za-z0-9-]+\.[A-Za-z.]{2,}", "email"),
    (r"sk-ant-[A-Za-z0-9]", "API key"),
]
TEXT = {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".txt", ".srt", ".cfg", ".ini", ""}


def test_repo_is_scrubbed():
    hits = []
    for f in ROOT.rglob("*"):
        if not f.is_file() or ".git" in f.parts or "__pycache__" in f.parts or ".pytest_cache" in f.parts:
            continue
        if f.suffix.lower() not in TEXT or f == Path(__file__).resolve():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for pat, what in PERSONAL:
            for m in re.finditer(pat, text):
                hits.append(f"{f.relative_to(ROOT)}: {what}: {m.group(0)}")
    assert not hits, "personal data in the repo:\n" + "\n".join(hits)


def test_scrub_patterns_catch_things():
    sample = "C:\\Users\\someone\\clips  me@example.com  local_83e3f349"
    assert sum(bool(re.search(p, sample)) for p, _ in PERSONAL) >= 3
