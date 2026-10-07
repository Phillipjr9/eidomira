import numpy as np
from app.engines.diagnostic import DiagnosticEngine


def test_diagnostic_roundtrip():
    engine = DiagnosticEngine()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    enrollment = engine.enroll(image)
    result = engine.process(image, enrollment.identity, True)
    assert result.image.shape == image.shape
    assert result.verified


def test_small_enrollment_rejected():
    engine = DiagnosticEngine()
    try:
        engine.enroll(np.zeros((20, 20, 3), dtype=np.uint8))
        assert False
    except ValueError:
        assert True
