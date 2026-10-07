"""Provider selection rules for the neural adapters.

`select_providers` is a pure function on purpose: these rules must be verifiable on a
build machine with no accelerator at all, which is where they would otherwise rot.
"""
import pytest

from app.providers import (
    CPU_PROVIDER,
    describe_providers,
    is_accelerated,
    select_providers,
)

CUDA = "CUDAExecutionProvider"
TRT = "TensorrtExecutionProvider"
ROCM = "ROCMExecutionProvider"
DML = "DmlExecutionProvider"
COREML = "CoreMLExecutionProvider"
OPENVINO = "OpenVINOExecutionProvider"


def test_cpu_only_host_still_gets_a_usable_provider():
    assert select_providers([CPU_PROVIDER]) == [CPU_PROVIDER]


def test_accelerator_is_preferred_over_cpu():
    assert select_providers([CPU_PROVIDER, CUDA]) == [CUDA, CPU_PROVIDER]
    assert select_providers([CPU_PROVIDER, ROCM]) == [ROCM, CPU_PROVIDER]


def test_cpu_terminates_the_list_whenever_the_runtime_reports_it():
    """The CPU fallback must sort last, never in front of an accelerator.

    It is only included when ONNX Runtime reports it: naming an unknown provider makes
    InferenceSession raise, so a host with a working accelerator must not be broken by
    a defensive-but-invented CPU entry.
    """
    assert select_providers([CUDA, CPU_PROVIDER]) == [CUDA, CPU_PROVIDER]
    assert select_providers([CPU_PROVIDER, CUDA, DML]) == [CUDA, DML, CPU_PROVIDER]
    # a runtime that somehow omits CPU yields the accelerator alone, not a broken list
    assert select_providers([CUDA]) == [CUDA]


def test_tensorrt_is_opt_in():
    available = [CPU_PROVIDER, CUDA, TRT]
    assert select_providers(available) == [CUDA, CPU_PROVIDER]
    assert select_providers(available, tensorrt=True) == [TRT, CUDA, CPU_PROVIDER]


def test_order_follows_expected_throughput():
    available = [CPU_PROVIDER, CUDA, ROCM, DML, COREML, OPENVINO]
    assert select_providers(available) == [CUDA, ROCM, DML, COREML, OPENVINO, CPU_PROVIDER]


def test_unavailable_providers_are_ignored():
    assert select_providers([CPU_PROVIDER, COREML]) == [COREML, CPU_PROVIDER]


def test_explicit_override_wins_and_is_deduplicated():
    available = [CPU_PROVIDER, CUDA, DML]
    assert select_providers(available, override=["directml", "directml"]) == [DML, CPU_PROVIDER]
    assert select_providers(available, override=["cpu"]) == [CPU_PROVIDER]


def test_override_naming_an_unavailable_provider_falls_back_to_automatic_order():
    assert select_providers([CPU_PROVIDER, CUDA], override=["directml"]) == [CUDA, CPU_PROVIDER]


def test_unknown_overrides_are_dropped_rather_than_guessed():
    assert select_providers([CPU_PROVIDER, CUDA], override=["bogus"]) == [CUDA, CPU_PROVIDER]


def test_describe_labels_are_readable():
    assert describe_providers([CUDA, CPU_PROVIDER]) == "cuda+cpu"
    assert describe_providers([TRT]) == "tensorrt"
    assert describe_providers([DML]) == "directml"
    assert describe_providers([]) == "none"
    # an unknown provider still produces something printable
    assert describe_providers(["SomeFutureExecutionProvider"]) == "somefuture"


def test_acceleration_flag():
    assert is_accelerated([CUDA, CPU_PROVIDER]) is True
    assert is_accelerated([CPU_PROVIDER]) is False
    assert is_accelerated([]) is False


def test_resolution_is_cached_and_reports_a_real_list():
    from app.providers import execution_providers

    first = execution_providers()
    assert first and first[-1] == CPU_PROVIDER or CUDA in first, first
    assert execution_providers() is first, "resolution should be cached per process"


@pytest.mark.parametrize("available", [
    [CPU_PROVIDER],
    [CPU_PROVIDER, CUDA],
    [CPU_PROVIDER, DML],
    [COREML, CPU_PROVIDER],
    [CUDA, ROCM],
])
def test_every_selection_is_non_empty_and_only_lists_available_providers(available):
    chosen = select_providers(available)
    assert chosen, available
    assert set(chosen) <= set(available), chosen
    if CPU_PROVIDER in available:
        assert chosen[-1] == CPU_PROVIDER, chosen
