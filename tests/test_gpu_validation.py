"""`tools/gpu_validation.py` has to run to the end on someone else's machine.

It is the one tool in this repository that is run by a person who did not write it, on a
machine nobody here can see, at the moment they are deciding whether to spend money. A crash
halfway through is worse than useless: it costs them a notebook session and leaves them with
no number.

Writing it went badly twice. A variable shadowing mistake meant the run died with `NameError`
*after* printing half the timings, in the extrapolation block — the one place a reader would
have trusted. And a failing `insightface` build, which is the likeliest failure on a shared
notebook, skipped the frame budget entirely rather than reporting what it could measure. Both
are covered below, and the second is forced deterministically rather than depending on whether
the machine running the suite happens to have insightface.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnx", reason="the stand-in generator needs onnx")
pytest.importorskip("onnxruntime", reason="the adapters need onnxruntime")

ROOT = Path(__file__).resolve().parent.parent


def run(models: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gpu_validation.py"), "--models", str(models)],
        cwd=ROOT, capture_output=True, text=True, timeout=600, env=environment)


def test_the_script_runs_to_the_end_and_prints_a_frame_budget(tmp_path):
    """No traceback, and the table that is the point of running it."""
    result = run(tmp_path / "standin")

    assert "Traceback" not in result.stderr, result.stderr
    # Exit 1 is not failure: on a host with no GPU the script reports CPU timings and says so.
    assert result.returncode in (0, 1), result.stderr
    assert "will be used" in result.stdout
    assert "one frame" in result.stdout
    assert "does not prove  quality" in result.stdout


def test_a_missing_swap_still_produces_a_budget_and_admits_the_gap(tmp_path):
    """`insightface` is one stage of four and builds from source, so it fails sometimes.

    Forced with a module that refuses to import rather than by uninstalling the real one: the
    point is to pin the behaviour on every machine, including the ones that have it installed.
    """
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "insightface.py").write_text('raise ImportError("hidden by the test")\n',
                                            encoding="utf-8")

    result = run(tmp_path / "standin", {"PYTHONPATH": str(blocker)})

    assert "Traceback" not in result.stderr, result.stderr
    assert "unavailable" in result.stdout
    assert "a floor, not a budget" in result.stdout
    # The numbers are still printed — that is the whole point of continuing.
    assert "one frame" in result.stdout
    assert "fps" in result.stdout
