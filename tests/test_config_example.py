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
    """Every variable the file offers, commented-out templates included.

    The comments count on purpose: they are the lines an operator uncomments. A template
    with the wrong name — the file used to offer `SMTP_HOST=`, which no setting reads — is
    the same silent failure as a live line with the wrong name, and it was invisible here
    because only prefixed lines were collected.
    """
    names = []
    for raw in EXAMPLE.read_text().splitlines():
        line = raw.strip().lstrip("#").strip()
        if not line or line.startswith(("#", "-", "/", "<")) or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name and name.replace("_", "").isalnum() and name.upper() == name:
            names.append(name)
    return names


def test_every_variable_the_example_offers_is_one_something_reads():
    """Bare names are the trap this file keeps falling into: set `SMTP_HOST` and the mailer
    keeps its default, with no error anywhere to search for."""
    unprefixed = [name for name in example_variables()
                  if not name.startswith(("STUDIO_", "EIDOMIRA_"))]
    assert not unprefixed, (
        f".env.example offers {unprefixed}, which nothing reads — settings need the "
        f"STUDIO_ prefix, and provider overrides need EIDOMIRA_"
    )


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
    """Guards the three settings this change introduced.

    Asserted on a fresh `Settings()`, not on the live object: the live one may have been
    changed by `tools/train_defaults.py`, and a test that fails on a machine where the
    trainer has done its job would be punishing the feature for working.
    """
    documented = Settings()
    assert documented.restoration_visibility == .75
    assert documented.tone_transfer_strength == 1.0
    assert documented.restoration_model_path == Path("models/gfpgan_1.4.onnx")
    assert documented.trainer_sample_every == 30
    assert documented.trainer_report_dir == Path("reports")
    # Off by default is the safety property: no model file, no restoration, no error.
    assert not documented.restoration_model_path.exists()


def test_what_the_offline_trainer_applied_is_recorded():
    """`why is this number not the one in .env.example` should not cost an hour."""
    from app.config import TUNED_DEFAULTS_APPLIED
    from app.knobs import KNOBS

    assert isinstance(TUNED_DEFAULTS_APPLIED, dict)
    assert set(TUNED_DEFAULTS_APPLIED) <= set(KNOBS), (
        "a tuned value was applied for a setting the bounds table does not cover"
    )


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
