from .diagnostic import DiagnosticEngine
from .inswapper import InSwapperEngine
from app.config import settings


def create_engine():
    if settings.backend == "inswapper":
        if not settings.model_path.exists():
            raise RuntimeError(f"Licensed model is missing: {settings.model_path}")
        return InSwapperEngine(str(settings.model_path), settings.verification_threshold)
    return DiagnosticEngine()
