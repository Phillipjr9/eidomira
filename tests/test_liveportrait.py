"""Tests for LivePortrait engine wiring and interface compatibility."""
from __future__ import annotations

import numpy as np
import pytest

from app.engines.factory import LIVEPORTRAIT_BACKEND, create_engine
from app.engines.liveportrait import LivePortraitEngine


def test_liveportrait_interface_standin():
    engine = LivePortraitEngine()
    assert engine.name == "liveportrait"
    assert engine.accelerated is True

    test_img = np.zeros((128, 128, 3), dtype=np.uint8)
    enrollment = engine.enroll(test_img)
    assert enrollment.identity["authorized"] is True

    verified, conf = engine.verify_self(test_img, enrollment.identity)
    assert verified is True
    assert conf == 1.0

    frame_result = engine.process(test_img, enrollment.identity, verified=True)
    assert frame_result.face_found is True
    assert frame_result.verified is True
    assert frame_result.image.shape == test_img.shape


def test_liveportrait_factory_wiring(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "backend", LIVEPORTRAIT_BACKEND)
    engine = create_engine()
    assert isinstance(engine, LivePortraitEngine)
