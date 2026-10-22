"""The vLLM sidecar: started on a free port, found ready, stopped; a crashing binary falls back."""

import os
import stat
import sys
import textwrap

import pytest

from chon import vllm_sidecar as vs

FAKE = textwrap.dedent('''\
    #!{py}
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    assert args[0] == "serve"
    port = int(args[args.index("--port") + 1])
    with open({record!r}, "w") as f:
        json.dump(args, f)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            body = json.dumps({{"data": [{{"id": "chon-reasoner"}}]}}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def _write(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_starts_and_stops(tmp_path, monkeypatch):
    record = tmp_path / "args.json"
    binary = _write(tmp_path / "vllm", FAKE.format(py=sys.executable, record=str(record)))
    monkeypatch.setattr(vs, "gpu_budget", lambda cfg: 0.4)
    url = vs.ensure("google/gemma-4-12B-it", {"bin": binary, "quantization": "fp8", "start_timeout_s": 60,
                                               "log": str(tmp_path / "v.log")})
    assert url and url.startswith("http://127.0.0.1:") and vs._alive(url) == "chon-reasoner"
    import json

    args = json.loads(record.read_text())
    assert args[1] == "google/gemma-4-12B-it" and "--quantization" in args and "fp8" in args
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.4"
    vs._stop_all()
    assert vs._alive(url, timeout=1) is None


def test_crash_and_missing_binary_give_none(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "gpu_budget", lambda cfg: 0.4)
    bad = _write(tmp_path / "vllm-bad", "#!/bin/sh\nexit 3\n")
    assert vs.ensure("m", {"bin": bad, "start_timeout_s": 30, "log": str(tmp_path / "b.log")}) is None
    monkeypatch.setattr(vs, "DEFAULT_BINS", ())
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("CHON_VLLM_BIN", raising=False)
    assert vs.ensure("m", {}) is None


def test_no_gpu_means_fallback(monkeypatch, tmp_path):
    binary = _write(tmp_path / "vllm", "#!/bin/sh\nsleep 100\n")
    monkeypatch.setattr(vs, "gpu_budget", lambda cfg: None)
    assert vs.ensure("m", {"bin": binary}) is None


def test_retries_without_text_only_option(tmp_path, monkeypatch):
    record = tmp_path / "args.json"
    picky = FAKE.format(py=sys.executable, record=str(record)).replace(
        'assert args[0] == "serve"', 'assert args[0] == "serve"\nif "--limit-mm-per-prompt" in args: sys.exit(2)')
    binary = _write(tmp_path / "vllm", picky)
    monkeypatch.setattr(vs, "gpu_budget", lambda cfg: 0.4)
    url = vs.ensure("m", {"bin": binary, "start_timeout_s": 60, "log": str(tmp_path / "v.log")})
    import json

    assert url and "--limit-mm-per-prompt" not in json.loads(record.read_text())
    vs._stop_all()
