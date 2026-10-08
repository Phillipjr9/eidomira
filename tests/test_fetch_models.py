"""The downloader: it must fetch, it must verify, and it must refuse.

Three artefacts, one of which — the swap model — carries non-commercial weights that cannot
legally be shipped in a paid product. A tool that fetches it because it was listed in a table
is a tool that puts a licence problem in `models/` while nobody is looking, so the refusal is
the part of this file worth testing hardest.

The fetch path is exercised against a local HTTP server rather than a stub, because the things
that go wrong here are transport-shaped: a truncated body, a connection that dies mid-stream,
a file that arrives complete and is not the file that was recorded.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import settings
from tools import fetch_models

PAYLOAD = b"not really a model, but the same 554 MB problem in miniature" * 64


class Handler(BaseHTTPRequestHandler):
    """Serves whatever the test put in `bodies`, so a source can change between fetches."""

    bodies: dict[str, bytes] = {}

    def do_GET(self):  # noqa: N802 - the name is fixed by BaseHTTPRequestHandler
        body = self.bodies.get(self.path, b"")
        if not body:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def source():
    """A local stand-in for Hugging Face, on a port the OS picks."""
    Handler.bodies = {"/model.onnx": PAYLOAD}
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/model.onnx"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def artefact(source):
    """The restorer's shape — commercial, fetches without a licence prompt — pointed at the
    local server so nothing leaves the machine."""
    original = fetch_models.by_role("restorer")
    return fetch_models.dataclasses.replace(
        original, urls=(source,), filename="model.onnx")


# ── the refusal ─────────────────────────────────────────────────────────────────

def test_the_swap_weights_are_not_fetched_without_a_licence(tmp_path, capsys):
    """The one that matters. It is listed, it is needed, and it is not ours to take."""
    code = fetch_models.fetch(fetch_models.by_role("swap"), tmp_path,
                              tmp_path / "lock.json", accept_licence=False)

    assert code == 2, "the non-commercial weights were fetched anyway"
    assert not (tmp_path / "inswapper_128.onnx").exists()
    assert not (tmp_path / "lock.json").exists(), "a refusal recorded a lock entry"
    message = capsys.readouterr().err
    assert "non-commercial" in message
    assert "--accept-licence" in message, "the way forward is not stated"


def test_the_licence_flag_is_about_the_operator_not_the_tool(tmp_path, artefact):
    """With the flag, the same call proceeds — the tool is not the gate, the licence is."""
    code = fetch_models.fetch(artefact, tmp_path, tmp_path / "lock.json", accept_licence=True)
    assert code == 0
    assert (tmp_path / "model.onnx").read_bytes() == PAYLOAD


# ── fetching, recording, verifying ──────────────────────────────────────────────

def test_a_fetch_records_what_it_received(tmp_path, artefact, source):
    lock_path = tmp_path / "lock.json"
    assert fetch_models.fetch(artefact, tmp_path, lock_path, True) == 0

    recorded = json.loads(lock_path.read_text())[artefact.filename]
    assert recorded["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert recorded["bytes"] == len(PAYLOAD)
    assert recorded["url"] == source
    assert recorded["commercial"] is True


def test_the_same_bytes_fetch_twice_without_complaint(tmp_path, artefact):
    lock_path = tmp_path / "lock.json"
    assert fetch_models.fetch(artefact, tmp_path, lock_path, True) == 0
    assert fetch_models.fetch(artefact, tmp_path, lock_path, True) == 0


def test_a_source_that_changes_is_refused_and_not_left_on_disk(tmp_path, artefact):
    """A mirror that starts serving something else — a different quantisation, or something
    else entirely — must not overwrite a verified file."""
    lock_path = tmp_path / "lock.json"
    assert fetch_models.fetch(artefact, tmp_path, lock_path, True) == 0

    Handler.bodies["/model.onnx"] = b"a different file entirely"
    assert fetch_models.fetch(artefact, tmp_path, lock_path, True) == 3
    assert not (tmp_path / "model.onnx").exists(), "the unverified file was left in place"


def test_a_broken_transport_leaves_nothing_behind(tmp_path, artefact):
    """A half-written model fails later at load time with a parse error that says nothing
    about the connection that caused it. It is written to `.part` and moved into place."""
    def dies_mid_stream(url, timeout=None):
        class Partial:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size=-1):
                raise OSError("connection reset by peer")

        return Partial()

    code = fetch_models.fetch(artefact, tmp_path, tmp_path / "lock.json", True,
                              opener=dies_mid_stream)

    assert code == 1, "a failed download reported success"
    assert list(tmp_path.iterdir()) == [], f"debris left behind: {list(tmp_path.iterdir())}"


def test_every_source_is_tried_in_order(tmp_path, artefact):
    """The first URL is often a mirror of a mirror. A dead one must not stop the fetch."""
    dead = fetch_models.dataclasses.replace(
        artefact, urls=("http://127.0.0.1:1/nothing.onnx",) + artefact.urls)
    assert fetch_models.fetch(dead, tmp_path, tmp_path / "lock.json", True) == 0
    assert (tmp_path / "model.onnx").exists()


# ── reporting ───────────────────────────────────────────────────────────────────

def test_check_says_what_is_present_and_what_it_is(tmp_path, artefact, capsys):
    fetch_models.fetch(artefact, tmp_path, tmp_path / "lock.json", True)
    # The full table, so the licence column is really the one the operator will read.
    assert fetch_models.report(tmp_path, tmp_path / "lock.json") == 0
    assert fetch_models.report(tmp_path, tmp_path / "lock.json",
                               artefacts=(artefact,)) == 0
    printed = capsys.readouterr().out
    assert "present" in printed
    assert "NON-COMMERCIAL" in printed, "the swap model's licence is not called out"
    assert "commercial ok" in printed


def test_check_flags_a_file_that_does_not_match_the_lock(tmp_path, artefact, capsys):
    lock_path = tmp_path / "lock.json"
    fetch_models.fetch(artefact, tmp_path, lock_path, True)
    (tmp_path / artefact.filename).write_bytes(b"swapped out behind the lock's back")

    assert fetch_models.report(tmp_path, lock_path, artefacts=(artefact,)) == 1
    assert "PRESENT BUT DIFFERENT" in capsys.readouterr().out


# ── and the table has to agree with the application ─────────────────────────────

def test_the_tool_and_the_app_agree_on_the_filenames():
    """The tool's whole purpose is to put a file where the app looks for it. Two lists that
    drift apart produce a download that succeeds and an engine that still says no model."""
    assert fetch_models.by_role("swap").filename == settings.model_path.name
    assert fetch_models.by_role("parser").filename == settings.parser_model_path.name
    assert fetch_models.by_role("restorer").filename == settings.restoration_model_path.name


def test_every_artefact_declares_a_licence_and_a_setting():
    for artefact in fetch_models.ARTEFACTS:
        assert artefact.licence, f"{artefact.role} has no licence stated"
        assert artefact.setting.startswith("STUDIO_"), artefact.setting
        assert artefact.urls, f"{artefact.role} has no source"
        assert isinstance(artefact.commercial, bool)
    roles = {artefact.role for artefact in fetch_models.ARTEFACTS}
    assert roles == {"swap", "parser", "restorer"}
