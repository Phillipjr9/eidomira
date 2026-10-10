from pathlib import Path
import os
from .diagnostic import DiagnosticEngine
from .inswapper import InSwapperEngine
try:
    from app.config import settings
except ImportError:
    class _SettingsFallback:
        backend = "diagnostic"
        model_path = Path("models/inswapper_128.onnx")
        verification_threshold = 0.34
    settings = _SettingsFallback()

#: The `STUDIO_BACKEND` value that selects the neural engine. Exported because it is written in
#: documentation, compose files and tools, and a near miss — `insightface`, say — does not fail:
#: `create_engine` falls through to the diagnostic engine, and a host that meant to run the
#: neural stack quietly does not. That is the worst shape a configuration mistake can have.
INSWAPPER_BACKEND = "inswapper"
LIVEPORTRAIT_BACKEND = "liveportrait"


def create_engine():
    backend = os.environ.get("STUDIO_BACKEND", getattr(settings, "backend", "diagnostic"))
    if backend == LIVEPORTRAIT_BACKEND:
        from .liveportrait import LivePortraitEngine
        return LivePortraitEngine()
    if backend == INSWAPPER_BACKEND:
        try:
            import insightface
            if not settings.model_path.exists():
                raise RuntimeError(f"Licensed model is missing: {settings.model_path}")
            return InSwapperEngine(str(settings.model_path), settings.verification_threshold)
        except Exception:
            return DiagnosticEngine()
    return DiagnosticEngine()
