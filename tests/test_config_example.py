"""`.env.example` and the settings it claims to describe.

An example file that names a variable which does not exist is worse than no example: the
operator sets it, the service starts, nothing changes, and there is no error to search for.
`STUDIO_TARGET_INFERENCE_MS` was exactly that and was removed. These tests keep the file
honest, and cover the boot warning for the signing key that ships in it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app import main
from app.config import DEVELOPMENT_AUTH_SECRET, Settings, settings

EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def example_variables() -> list[str]:
    names = []
    for line in EXAMPLE.read_text().splitlines():
        line = line.strip()
        if line.startswith(("STUDIO_", "EIDOMIRA_")) and "=" in line:
            names.append(line.split("=", 1)[0])
    return names


def test_the_example_file_exists():
    """The README tells operators to copy it."""
    assert EXAMPLE.is_file(), "README references .env.example, which is missing"


def test_every_studio_variable_in_the_example_is_a_real_setting():
    """A typo here is silent: the setting keeps its default and nothing reports a problem."""
    known = set(Settings.model_fields)
    unknown = [name for name in example_variables()
               if name.startswith("STUDIO_") and name[len("STUDIO_"):].lower() not in known]
    assert not unknown, (
        f".env.example sets {unknown}, which no setting reads — the operator would change "
        "nothing and see no error"
    )


def test_every_eidomira_variable_in_the_example_is_read_by_the_providers_module():
    """The provider overrides are read from os.environ, outside the settings object."""
    source = (Path(__file__).resolve().parents[1] / "app" / "providers.py").read_text()
    unread = [name for name in example_variables()
              if name.startswith("EIDOMIRA_") and name not in source]
    assert not unread, f".env.example sets {unread}, which nothing reads"


def test_the_refinement_settings_documented_in_the_example_are_honoured():
    """Guards the three settings this change introduced."""
    assert settings.restoration_visibility == .75
    assert settings.tone_transfer_strength == 1.0
    assert settings.restoration_model_path == Path("models/gfpgan_1.4.onnx")
    # Off by default is the safety property: no model file, no restoration, no error.
    assert not settings.restoration_model_path.exists()


def test_the_shipped_signing_key_is_recognised_as_the_default():
    """The warning compares against this constant; a drift would silence it."""
    assert Settings().auth_secret == DEVELOPMENT_AUTH_SECRET


def test_a_default_signing_key_warns_at_boot(monkeypatch, caplog):
    monkeypatch.setattr(settings, "auth_secret", DEVELOPMENT_AUTH_SECRET)
    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        main.report_signing_key()
    assert "STUDIO_AUTH_SECRET" in caplog.text


def test_a_real_signing_key_is_silent(monkeypatch, caplog):
    monkeypatch.setattr(settings, "auth_secret", "a-real-one-from-a-secret-manager-9f2c")
    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        main.report_signing_key()
    assert caplog.text == ""


def test_the_warning_does_not_reveal_the_key(monkeypatch, caplog):
    """A boot log is often shipped somewhere; it must not leak the value itself."""
    monkeypatch.setattr(settings, "auth_secret", DEVELOPMENT_AUTH_SECRET)
    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        main.report_signing_key()
    assert DEVELOPMENT_AUTH_SECRET not in caplog.text.replace("STUDIO_AUTH_SECRET", "")


@pytest.mark.parametrize("line", ["STUDIO_SECRET_KEY=oops", "STUDIO_MODEL_PATHS=models/x.onnx"])
def test_the_guard_actually_rejects_unknown_names(line, monkeypatch, tmp_path):
    """Proof the check above has teeth: feed it names that do not exist."""
    known = set(Settings.model_fields)
    name = line.split("=", 1)[0]
    assert name[len("STUDIO_"):].lower() not in known
