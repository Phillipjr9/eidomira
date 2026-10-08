"""Run a real neural face swap on a machine with internet and GPU access (e.g. Kaggle).

Downloads the real inswapper_128.onnx weights from Hugging Face if not already present,
initializes the real InsightFace InSwapperEngine with buffalo_l face detection,
and renders an actual face swap between two portraits.

Usage:
    python tools/run_real_swap.py
    python tools/run_real_swap.py --source static/persona-01.jpg --target static/persona-02.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
DEFAULT_MODEL = ROOT / "models" / "inswapper_128.onnx"


def preload_cuda():
    import ctypes
    import glob
    import os
    search_paths = []
    for p in sys.path:
        search_paths.extend(glob.glob(os.path.join(p, "nvidia", "*", "lib")))
    search_paths.append("/usr/local/cuda/lib64")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(search_paths) + (":" + current_ld if current_ld else "")
    for path in search_paths:
        for so_file in glob.glob(os.path.join(path, "*.so*")):
            try:
                ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass


def download_model(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100_000_000:
        print(f"Model already downloaded: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading real neural swap weights from {MODEL_URL}…")
    partial = destination.with_suffix(".part")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=300) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = 0
        while True:
            chunk = response.read(2 << 20)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded / total * 100)
                if pct >= last_pct + 10:
                    print(f"  {pct}% ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)…")
                    last_pct = pct
    partial.replace(destination)
    print(f"Download complete: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=ROOT / "static" / "persona-01.jpg")
    parser.add_argument("--target", type=Path, default=ROOT / "static" / "persona-02.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "swapped_result.jpg")
    options = parser.parse_args()

    preload_cuda()

    if not options.model.exists() or options.model.stat().st_size < 100_000_000:
        download_model(options.model)

    import cv2
    from app.engines.inswapper import InSwapperEngine

    print(f"Initializing InSwapperEngine with model: {options.model}…")
    started = time.perf_counter()
    engine = InSwapperEngine(str(options.model), threshold=0.34)
    print(f"Engine initialized in {(time.perf_counter() - started) * 1000:.1f} ms")
    print(f"Provider in use: {engine.provider} (accelerated: {engine.accelerated})")

    if not options.source.exists() or not options.target.exists():
        print(f"Error: source ({options.source}) or target ({options.target}) not found.")
        return 1

    src_bgr = cv2.imread(str(options.source))
    dst_bgr = cv2.imread(str(options.target))
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    dst_rgb = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)

    print("Detecting and enrolling source identity…")
    enrollment = engine.enroll(src_rgb)

    print("Running neural face swap onto target portrait…")
    t0 = time.perf_counter()
    result = engine.process(dst_rgb, enrollment.identity, verified=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(options.output), out_bgr)

    print("\n" + "=" * 50)
    print("  REAL NEURAL FACE SWAP COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Execution time : {elapsed_ms:.1f} ms")
    print(f"  Face found     : {result.face_found}")
    print(f"  Saved result to: {options.output}")
    print("=" * 50 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
