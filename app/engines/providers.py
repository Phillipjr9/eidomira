"""ONNX Runtime execution provider selection for the neural adapters.

The adapter previously hardcoded ``["CUDAExecutionProvider", "CPUExecutionProvider"]``,
which means any host that has ROCm, DirectML, CoreML, OpenVINO or TensorRT installed
silently fell through to the CPU path and just looked slow, with nothing in the logs or
the health endpoint to explain why.

This module asks ONNX Runtime what the host actually exposes, orders it by expected
throughput, and always keeps a CPU path last so inference can never fail outright.

TensorRT is deliberately opt-in (``EIDOMIRA_TENSORRT=1``): it builds an engine on first
use, which can take minutes, and that is the wrong default for a request path that is
supposed to answer in milliseconds. Operators who have warmed a cache can enable it.

``EIDOMIRA_PROVIDERS=cuda,cpu`` overrides the automatic order entirely.
"""
from __future__ import annotations

import os
from functools import lru_cache

# (short label, ONNX Runtime provider name), strongest expected throughput first.
PROVIDER_ORDER: tuple[tuple[str, str], ...] = (
    ("tensorrt", "TensorrtExecutionProvider"),
    ("cuda", "CUDAExecutionProvider"),
    ("rocm", "ROCMExecutionProvider"),
    ("migraphx", "MIGraphXExecutionProvider"),
    ("directml", "DmlExecutionProvider"),
    ("coreml", "CoreMLExecutionProvider"),
    ("openvino", "OpenVINOExecutionProvider"),
    ("cpu", "CPUExecutionProvider"),
)

CPU_PROVIDER = "CPUExecutionProvider"
_ORT_NAME = {short: name for short, name in PROVIDER_ORDER}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _override() -> list[str] | None:
    raw = os.environ.get("EIDOMIRA_PROVIDERS", "").strip()
    return [part.strip().lower() for part in raw.split(",") if part.strip()] or None


def select_providers(
    available: list[str] | tuple[str, ...],
    *,
    tensorrt: bool = False,
    override: list[str] | None = None,
) -> list[str]:
    """Pick an ordered provider list from what the runtime reports as available.

    Pure function: no environment or ONNX Runtime access, so the ordering rules are
    testable on a machine with no accelerator at all.
    """
    present = set(available)
    chosen: list[str] = []

    for short in override or ():
        name = _ORT_NAME.get(short)
        if name and name in present and name not in chosen:
            chosen.append(name)

    if not chosen:
        for short, name in PROVIDER_ORDER:
            if short == "tensorrt" and not tensorrt:
                continue
            if name in present and name not in chosen:
                chosen.append(name)

    # Keep the CPU path last when the runtime reports it (it always does in practice).
    # It is intentionally NOT appended unconditionally: naming a provider ONNX Runtime
    # does not recognise makes InferenceSession raise, which would turn a host with a
    # working accelerator into a hard startup failure.
    if CPU_PROVIDER in present and CPU_PROVIDER not in chosen:
        chosen.append(CPU_PROVIDER)

    return chosen


def describe_providers(providers: list[str] | tuple[str, ...]) -> str:
    """Human label for telemetry, e.g. ``tensorrt`` or ``directml+cpu``."""
    reverse = {name: short for short, name in PROVIDER_ORDER}
    labels = [reverse.get(name, name.replace("ExecutionProvider", "").lower()) for name in providers]
    return "+".join(labels) if labels else "none"


def is_accelerated(providers: list[str] | tuple[str, ...]) -> bool:
    """True when inference will run somewhere faster than the CPU."""
    return any(name != CPU_PROVIDER for name in providers)


@lru_cache(maxsize=1)
def execution_providers() -> tuple[str, ...]:
    """The provider list this host should use, resolved once per process."""
    try:
        import onnxruntime
    except ImportError:
        # requirement not installed: let the caller's own error explain the fix
        return (CPU_PROVIDER,)

    try:
        available = list(onnxruntime.get_available_providers())
    except Exception:
        available = [CPU_PROVIDER]

    selected = select_providers(
        available,
        tensorrt=_flag("EIDOMIRA_TENSORRT"),
        override=_override(),
    )
    return tuple(selected or [CPU_PROVIDER])
