"""Validate the neural stack on a borrowed GPU — Kaggle, Colab, or any hourly box.

Written for the situation this project is actually in: the pipeline is built, the licensed
artefacts are not bought yet, and the only hardware available is somebody else's free tier. A
notebook cannot host the live studio — no public IP, no inbound UDP, no persistent process —
but it can answer the question that has to be answered *before* anything is paid for: does the
neural path run, on a GPU, at a usable speed?

    # in a Kaggle or Colab cell (enable Internet access first)
    !git clone https://github.com/<owner>/eidomira.git && cd eidomira && python tools/gpu_validation.py

    # or upload the repository as a Kaggle dataset and point at it
    python tools/gpu_validation.py --repo /kaggle/input/eidomira

It reports, with numbers rather than adjectives:

    the GPU ONNX Runtime actually resolved (and whether it is accelerated)
    per-stage latency: parser mask, compositor blend, restorer, one swap pass
    the boost at scale=2, which is four swap passes, so its cost is 4x measurable
    whether the compositor still preserves the occluder the parser labelled
    an extrapolated frame budget at 384 / 768 / 960 px — the three widths the adaptive
      controller in app/adaptive.py actually uses

With no licensed weights present it runs against `tools/stand_in_models.py`, which have the
right shapes and no trained parameters: that proves the plumbing and measures the *host*, not
the quality. Point `--models` at a directory holding real weights to measure those instead,
and expect very different numbers — a real inswapper pass is far more work than an identity
convolution.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Runnable either way — `python tools/gpu_validation.py` from a notebook cell, or
# `python -m tools.gpu_validation`. Run as a script, sys.path[0] is tools/ and `import app`
# fails; the repository root has to be on the path before anything under it is imported.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str]] = []


def report(name: str, value: str) -> None:
    """One line of the report, kept and printed so the whole thing can be copied out."""
    RESULTS.append((name, value))
    print(f"  {name:<34} {value}")


def time_it(call, repeats: int = 5) -> tuple[float, float]:
    """(median, worst) milliseconds. Median because a shared notebook has noisy neighbours."""
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), max(samples)


def describe_host() -> bool:
    """What the *application* will resolve, and whether that is a GPU.

    Deliberately not "is CUDA in the provider list": ONNX Runtime lists providers it has
    compiled in, including ones this host cannot use, and `AzureExecutionProvider` sorts first
    on a plain CPU build — so a naive first-entry check reports an accelerated host on a
    machine with no GPU at all. That is the same false positive that makes a deployment look
    successful on a CPU box, and it is worth not repeating in the tool that measures it.

    `app/providers.py` already answers this properly: it filters to what the host actually
    exposes and orders by expected throughput, and the engine uses it. This reports the same
    thing the studio would.
    """
    import onnxruntime as ort

    from app.providers import describe_providers, execution_providers

    resolved = list(execution_providers())
    summary = describe_providers(resolved)
    report("onnxruntime", ort.__version__)
    report("compiled in", ", ".join(ort.get_available_providers()))
    report("will be used", ", ".join(resolved))
    accelerated = summary != "cpu"
    report("accelerated", "yes" if accelerated else "NO — everything below is CPU timing")
    if not accelerated:
        if gpu_is_present():
            explain_the_fallback()
        else:
            print("\n  note: no GPU provider will be used, and no GPU is attached. On Kaggle,")
            print("  check Settings -> Accelerator -> GPU, then make sure onnxruntime-gpu is")
            print("  the version installed (pip uninstall -y onnxruntime onnxruntime-gpu first).")
    return accelerated


def explain_the_fallback() -> None:
    """A GPU is present and ONNX Runtime still will not use it. Say why.

    The usual cause is a library the provider cannot load, and ONNX Runtime reports that on
    the exception it raises when asked for a CUDA session — but only if something asks. The
    generic 'no GPU provider' line sends people looking in the wrong place, and this is the
    most common way a Kaggle run appears to work while measuring a CPU.
    """
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "probe",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    try:
        ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    except Exception as error:
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        report("cuda refusal", (lines[0] if lines else type(error).__name__)[:120])
        print("\n  ONNX Runtime will not open the CUDA provider. The two usual reasons:")
        print("    a CPU-only onnxruntime installed over the GPU one — run:")
        print("      !pip uninstall -y onnxruntime onnxruntime-gpu && pip install -q onnxruntime-gpu")
        print("    libraries the provider cannot load — run:")
        print("      !pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12")
    else:
        report("cuda refusal", "none — a CUDA session opens, so the provider is usable")


def describe_gpu() -> None:
    """`nvidia-smi` if the host has one. Absent is a fact worth reporting, not an error: a
    CPU-only notebook is a perfectly reasonable place to run this and get CPU timings."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        report("nvidia-smi", "not present")
        return
    line = completed.stdout.strip().splitlines()
    report("nvidia-smi", line[0] if completed.returncode == 0 and line else "not present")
    return completed.returncode == 0 and bool(line)


def gpu_is_present() -> bool:
    try:
        completed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def ensure_models(directory: Path) -> Path:
    """Use real weights if they are there, otherwise build the stand-ins."""
    wanted = ("inswapper_128.onnx", "face_parser.onnx", "gfpgan_1.4.onnx")
    if all((directory / name).is_file() for name in wanted):
        report("models", f"real weights from {directory}")
        return directory

    from tools import stand_in_models

    report("models", "stand-ins (correct shapes, no trained weights)")
    stand_in_models.build(directory)
    return directory


def measure(models: Path, source=None) -> None:
    import cv2
    import numpy as np

    frame = source
    if frame is None:
        # A synthetic face-ish frame: enough structure for a resize, a conv and a blend, and
        # nothing that requires a dataset the notebook may not have.
        frame = np.zeros((720, 1280, 3), np.uint8)
        cv2.ellipse(frame, (640, 360), (150, 210), 0, 0, 360, (198, 172, 148), -1)
        cv2.ellipse(frame, (590, 320), (26, 14), 0, 0, 360, (40, 40, 60), -1)
        cv2.ellipse(frame, (690, 320), (26, 14), 0, 0, 360, (40, 40, 60), -1)
        cv2.ellipse(frame, (640, 470), (60, 22), 0, 0, 360, (90, 70, 90), -1)
        cv2.rectangle(frame, (0, 0), (1280, 200), (60, 60, 70), -1)
    height, width = frame.shape[:2]
    bbox = (int(width * .25), int(height * .10), int(width * .75), int(height * .85))

    # ── the parser and the compositor ──────────────────────────────────────────
    from app.compositor import SemanticCompositor

    compositor = SemanticCompositor(str(models / "face_parser.onnx"), 0.035)
    report("parser size", str(compositor.size))
    report("compositor fault", compositor.fault or "none")
    mask_ms, mask_worst = time_it(lambda: compositor.mask(frame, bbox))
    report("parser mask", f"{mask_ms:.1f} ms median, {mask_worst:.1f} ms worst")
    report("mask coverage", f"{(compositor.mask(frame, bbox) > .5).mean() * 100:.1f}% of the frame")

    swapped = frame.copy()
    blend_ms, _ = time_it(lambda: compositor.blend(frame, swapped, bbox))
    report("compositor blend", f"{blend_ms:.1f} ms median")

    # ── the restorer ───────────────────────────────────────────────────────────
    from app.enhance import FaceRestorer

    restorer = FaceRestorer(str(models / "gfpgan_1.4.onnx"), .75)
    report("restorer size", str(restorer.size))
    restore_ms, _ = time_it(lambda: restorer.enhance(frame, bbox))
    report("restorer", f"{restore_ms:.1f} ms median")
    report("restorer fault", restorer.fault or "none")

    # ── the swap, through the loader the engine uses ───────────────────────────
    #
    # Not fatal if this fails. `insightface` builds from source and can fail on a shared
    # notebook; the swap is one stage of four, and a run that reports the other three still
    # answers the question. What it must not do is print a frame budget that silently omits
    # the most expensive stage, so the omission is stated on the line above the table.
    swapper = None
    swap_ms = 0.0
    try:
        import insightface
    except ImportError:
        report("swap", "insightface is not installed — no swap timing")
    else:
        try:
            swapper = insightface.model_zoo.get_model(str(models / "inswapper_128.onnx"),
                                                     providers=compositor.providers)
        except Exception as error:
            report("swap", f"could not load the model: {type(error).__name__}: {error}")

    if swapper is None:
        report("swap model", "unavailable — the swap pass is missing from the budget below")
    else:
        import numpy as np
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(1, 512)).astype(np.float32)
        crop = frame[:128, :128, ::-1].copy()

        def one_pass():
            blob = crop.astype(np.float32).transpose(2, 0, 1)[None]
            return swapper.forward(blob, latent)

        swap_ms, swap_worst = time_it(one_pass, repeats=10)
        report("swap, one 128px pass", f"{swap_ms:.1f} ms median, {swap_worst:.1f} ms worst")

        # ── the boost, which is four of those ─────────────────────────────────
        from app import boost

        def swap_once(pixels):
            blob = pixels[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]
            out = swapper.forward(blob, latent)[0].transpose(1, 2, 0)
            return np.clip(out * 255, 0, 255).astype(np.uint8)[:, :, ::-1]

        boosted = boost.boosted_face(lambda dx, dy: crop, swap_once, scale=2)
        report("boost scale=2 canvas", f"{boosted.shape[1]}x{boosted.shape[0]} (4 swap passes)")

    # ── what that means for a frame budget ────────────────────────────────────
    full = swap_ms + mask_ms + restore_ms + blend_ms
    if swapper is None:
        print("\n  one frame, WITHOUT the swap stage — a floor, not a budget:")
        print("    the swap is the largest stage of the four, so treat these as the least it")
        print("    can cost, not as what a frame will cost.")
    else:
        print("\n  one frame, all stages in sequence (before the boost):")
    for label, width_px in (("speed   ", 384), ("balanced", 768), ("quality ", 960)):
        # Cost scales with pixels, so this is an honest extrapolation for a convolution-
        # shaped model and a rough guide for a real one. The three widths are the ones
        # app/adaptive.py actually targets.
        share = (width_px * width_px) / float(width * width)
        scaled = full * share
        report(f"  {label} @ {width_px}px",
               f"{scaled:.1f} ms/frame -> {1000.0 / max(scaled, 1e-6):.1f} fps")

    if swapper is not None:
        print("\n  above is BEFORE the pixel boost. At scale=2 the swap runs four times per")
        print("  face, so add roughly 3x the swap time to the figures above.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, default=Path("models/standin"),
                        help="directory holding the three ONNX files (default: build stand-ins)")
    parser.add_argument("--photo", type=Path, default=None,
                        help="an image to measure with instead of the synthetic frame")
    options = parser.parse_args(argv)

    print("\n  Eidomira neural-path validation\n")

    # A fresh notebook has none of this. Say exactly what to run rather than raising an
    # ImportError four frames deep, because the person running this is following a page of
    # instructions and the instruction is the useful output.
    missing = []
    for module, package in (("onnxruntime", "onnxruntime-gpu"), ("onnx", "onnx")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("  this notebook is missing:", ", ".join(missing))
        print("\n  run this cell first:\n")
        print(f"    !pip install -q {' '.join(missing)}")
        print("\n  on Kaggle, also check Settings -> Accelerator -> GPU, and Internet -> on.")
        return 2

    describe_gpu()
    accelerated = describe_host()


    models = ensure_models(options.models)

    source = None
    if options.photo and options.photo.is_file():
        import cv2
        source = cv2.imread(str(options.photo))[:, :, ::-1]
        report("photo", f"{options.photo.name} {source.shape[1]}x{source.shape[0]}")

    print()
    measure(models, source)

    print("\n  what this does and does not prove:")
    print("    proves  the plumbing runs on this host, and how fast this GPU is at these shapes")
    print("    does not prove  quality. With stand-ins there are no trained weights to judge;")
    print("                    with real weights, quality is a separate measurement.")
    print("\n  a notebook cannot host the live studio — no public IP, no inbound UDP, no")
    print("  persistent process. See docs/elastic-compute-plan.md for where that runs.")
    if not accelerated:
        print("\n  RESULT: ran on CPU. Attach a GPU and run again for meaningful numbers.")
    return 0 if accelerated else 1


if __name__ == "__main__":
    sys.exit(main())
