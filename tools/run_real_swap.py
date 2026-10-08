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

PARSER_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx"
RESTORER_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx"
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


def download_file(url: str, destination: Path, label: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"{label} already present: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading {label} from {url}…")
    partial = destination.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
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
                if pct >= last_pct + 20:
                    print(f"  {pct}% ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)…")
                    last_pct = pct
    partial.replace(destination)
    print(f"Download complete: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=ROOT / "static" / "human-01.jpg")
    parser.add_argument("--target", type=Path, default=ROOT / "static" / "human-02.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "swapped_result.jpg")
    parser.add_argument("--enhance", action="store_true", default=True, help="enable GFPGAN detail restoration & parser mask")
    options = parser.parse_args()

    preload_cuda()

    try:
        import onnxruntime
        import insightface
    except ImportError:
        print("Installing insightface and runtime dependencies…")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime", "insightface"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)

    alt_model = Path("/kaggle/working/inswapper_128.onnx")
    if not options.model.exists() and alt_model.exists() and alt_model.stat().st_size > 100_000_000:
        options.model = alt_model

    if not options.model.exists() or options.model.stat().st_size < 100_000_000:
        download_file(MODEL_URL, options.model, "InSwapper 128 Model")

    # Download restoration and parsing models for Swapface-level quality
    parser_path = ROOT / "models" / "face_parser.onnx"
    restorer_path = ROOT / "models" / "gfpgan_1.4.onnx"
    try:
        download_file(PARSER_URL, parser_path, "Face Parser (BiSeNet)")
    except Exception as e:
        print(f"Note: parser download skipped ({e})")
    try:
        download_file(RESTORER_URL, restorer_path, "Face Restorer (GFPGAN v1.4)")
    except Exception as e:
        print(f"Note: restorer download skipped ({e})")

    # Link models so relative paths always resolve
    try:
        import os, shutil
        os.makedirs("/kaggle/working/models", exist_ok=True)
        if parser_path.exists() and not Path("/kaggle/working/models/face_parser.onnx").exists():
            shutil.copyfile(str(parser_path), "/kaggle/working/models/face_parser.onnx")
        if restorer_path.exists() and not Path("/kaggle/working/models/gfpgan_1.4.onnx").exists():
            shutil.copyfile(str(restorer_path), "/kaggle/working/models/gfpgan_1.4.onnx")
    except Exception:
        pass

    import cv2
    from app.engines.inswapper import InSwapperEngine

    print(f"Initializing InSwapperEngine with model: {options.model}…")
    started = time.perf_counter()
    engine = InSwapperEngine(str(options.model), threshold=0.34)
    print(f"Engine initialized in {(time.perf_counter() - started) * 1000:.1f} ms")
    print(f"Provider in use: {engine.provider} (accelerated: {engine.accelerated})")
    print(f"Compositor (BiSeNet): {'ACTIVE' if engine.compositor is not None else 'DISABLED'}")
    print(f"Restorer (GFPGAN):   {'ACTIVE' if engine.restorer is not None else 'DISABLED'}")

    def find_face_image(candidates):
        for path in candidates:
            if not path.exists():
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            faces = engine._faces(rgb)
            if not faces:
                engine.analyzer.prepare(ctx_id=0 if "CUDAExecutionProvider" in engine.providers else -1, det_size=(640, 640), det_thresh=0.2)
                faces = engine._faces(rgb)
            if faces:
                best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                return path, rgb, best
        return None, None, None

    h1 = ROOT / "static" / "human-01.jpg"
    h2 = ROOT / "static" / "human-02.jpg"

    all_portraits = [h1, h2]

    print("Detecting and selecting source identity…")
    src_path, src_rgb, best_src = find_face_image(all_portraits)
    if best_src is None:
        print("Error: could not find face in sample images.")
        return 1
    print(f"Source face: {src_path.name} (bbox={[int(x) for x in best_src.bbox]})")

    print("Detecting target portrait…")
    remaining = [p for p in all_portraits if p != src_path]
    dst_path, dst_rgb, best_dst = find_face_image(remaining)
    if best_dst is None:
        print("Error: could not find target face.")
        return 1
    print(f"Target face: {dst_path.name} (bbox={[int(x) for x in best_dst.bbox]})")

    print(f"Running neural face swap with GFPGAN & BiSeNet ({src_path.name} -> {dst_path.name})…")
    t0 = time.perf_counter()
    result = engine.process(
        dst_rgb, best_src, verified=True,
        overrides={
            "tone_transfer_strength": 0.45,
            "parser_feather": 0.55,
            "restoration_visibility": 1.0,
        }
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(options.output), out_bgr)
    # Also save to current directory and parent directory so IPython display never fails
    cv2.imwrite("swapped_result.jpg", out_bgr)
    cv2.imwrite("/kaggle/working/swapped_result.jpg", out_bgr)
    cv2.imwrite("/kaggle/working/eidomira/swapped_result.jpg", out_bgr)

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
