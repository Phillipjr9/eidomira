#!/usr/bin/env python3
"""tools/launch_cloud_studio.py — Launch Cloud GPU WebRTC Studio with Public HTTPS Tunnel.

Runs the Eidomira Neural WebRTC Studio on a Cloud GPU (e.g. Kaggle Tesla T4),
exposes it via an instant, secure Cloudflare Tunnel (HTTPS with full camera access),
and prints the clickable live studio URL for your Mac browser once fully ready.

Usage:
    python tools/launch_cloud_studio.py
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Ensure immediate line flushing in Jupyter/Kaggle environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_BIN = ROOT / "cloudflared"


def download_cloudflared():
    for p in [Path("/kaggle/working/cloudflared"), CLOUDFLARED_BIN, Path("/usr/local/bin/cloudflared")]:
        if p.exists() and p.stat().st_size > 10_000_000:
            return p
    target = Path("/kaggle/working/cloudflared") if Path("/kaggle/working").exists() else CLOUDFLARED_BIN
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching Cloudflare tunnel binary into {target}…", flush=True)
    for cmd in [
        ["curl", "-fSL", "--connect-timeout", "15", "--max-time", "45", CLOUDFLARED_URL, "-o", str(target)],
        ["wget", "-q", "--timeout=30", CLOUDFLARED_URL, "-O", str(target)],
    ]:
        try:
            subprocess.run(cmd, check=True)
            if target.exists() and target.stat().st_size > 10_000_000:
                target.chmod(0o755)
                print("Cloudflare Tunnel ready.", flush=True)
                return target
        except Exception:
            continue
    return None


def ensure_file_from_urls(target_path: Path, urls: list[str], min_size: int, label: str):
    if target_path.exists() and target_path.stat().st_size >= min_size:
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Model Setup]: Downloading {label}…", flush=True)
    for url in urls:
        try:
            print(f"Fetching from {url}…", flush=True)
            subprocess.run(["curl", "-fSL", "--progress-bar", url, "-o", str(target_path)], check=True)
            if target_path.exists() and target_path.stat().st_size >= min_size:
                print(f"✅ {label} ready ({target_path.stat().st_size / 1_000_000:.1f} MB)", flush=True)
                return target_path
        except Exception as exc:
            print(f"Notice on {url}: {exc}, trying next mirror…", flush=True)
            continue
    return target_path


def ensure_models_for_swap():
    ensure_buffalo_l()
    m_swap = ensure_file_from_urls(
        ROOT / "models" / "inswapper_128.onnx",
        [
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
            "https://huggingface.co/xingren23/comfyflow-models/resolve/main/insightface/inswapper_128.onnx",
        ],
        500_000_000,
        "InSwapper face swap weights (~554 MB)",
    )
    ensure_file_from_urls(
        ROOT / "models" / "face_parser.onnx",
        [
            "https://huggingface.co/facefusion/models-3.3.0/resolve/main/face_parser.onnx",
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/face_parser.onnx",
        ],
        50_000_000,
        "BiSeNet face parser model (~50 MB)",
    )
    ensure_file_from_urls(
        ROOT / "models" / "gfpgan_1.4.onnx",
        [
            "https://huggingface.co/facefusion/models-3.3.0/resolve/main/gfpgan_1.4.onnx",
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
        ],
        300_000_000,
        "GFPGAN v1.4 face restoration model (~332 MB)",
    )
    return m_swap


def ensure_buffalo_l():
    user_bf = Path.home() / ".insightface" / "models" / "buffalo_l"
    if user_bf.exists() and any(user_bf.glob("*.onnx")):
        return user_bf

    # Check local caches first
    for cand in [
        Path("/kaggle/working/LivePortrait/pretrained_weights/insightface/models/buffalo_l"),
        ROOT / "pretrained_weights" / "insightface" / "models" / "buffalo_l",
    ]:
        if cand.exists() and any(cand.glob("*.onnx")):
            user_bf.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(str(cand), str(user_bf))
                print("Linked InsightFace buffalo_l models from cache.", flush=True)
                return user_bf
            except Exception:
                pass

    user_bf.parent.mkdir(parents=True, exist_ok=True)
    zip_path = user_bf.parent / "buffalo_l.zip"
    bf_urls = [
        "https://huggingface.co/techwavelaps/aceplus-face/resolve/main/insightface/models/buffalo_l.zip",
        "https://huggingface.co/public-data/insightface/resolve/main/buffalo_l.zip",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    ]
    print("\n[InsightFace Setup]: Downloading buffalo_l landmark models (~275 MB)…", flush=True)
    for url in bf_urls:
        try:
            print(f"Fetching buffalo_l from {url}…", flush=True)
            subprocess.run(["curl", "-fSL", "--progress-bar", url, "-o", str(zip_path)], check=True)
            if zip_path.exists() and zip_path.stat().st_size > 50_000_000:
                print("Extracting buffalo_l models…", flush=True)
                import zipfile
                with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
                    zip_ref.extractall(str(user_bf))
                zip_path.unlink(missing_ok=True)
                # Flatten nested folder if the zip contained a root directory
                if not any(user_bf.glob("*.onnx")):
                    for sub in user_bf.glob("*"):
                        if sub.is_dir() and any(sub.glob("*.onnx")):
                            for f in sub.glob("*.onnx"):
                                shutil.move(str(f), str(user_bf / f.name))
                print("✅ InsightFace buffalo_l ready!", flush=True)
                return user_bf
        except Exception as exc:
            print(f"Notice on buffalo_l mirror ({exc}), trying next…", flush=True)
            continue
    return user_bf


def main():
    print("=" * 72, flush=True)
    print("  EIDOMIRA CLOUD GPU LIVE WEBRTC STUDIO", flush=True)
    print(f"  Python: {sys.version.split()[0]} | Working dir: {os.getcwd()}")
    print("=" * 72, flush=True)

    # 0. Ensure working directory is repo root and environment is configured
    os.chdir(str(ROOT))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Free any leftover processes from previous runs to guarantee clean port 8000 and full GPU VRAM
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
    subprocess.run(["pkill", "-9", "-f", "LivePortrait/app.py"], check=False)
    subprocess.run(["pkill", "-9", "-f", "uvicorn"], check=False)
    subprocess.run(["fuser", "-k", "8000/tcp"], check=False)

    import argparse
    parser = argparse.ArgumentParser(description="Launch Eidomira Studio on Cloud GPU")
    parser.add_argument(
        "--mode",
        choices=["swap", "portrait"],
        default="swap",
        help="Studio mode: 'swap' (InSwapper Live Webcam Face Replacement) or 'portrait' (LivePortrait Neural Motion Avatar)"
    )
    args, _ = parser.parse_known_args()

    os.environ["STUDIO_ALLOWED_ORIGINS"] = "*"
    os.environ["STUDIO_DEMO_LOGIN"] = "true"
    os.environ["STUDIO_DEMO_PASSWORD"] = "eidomira-demo-2026"
    os.environ["STUDIO_REQUIRE_SELF_VERIFICATION"] = "false"
    os.environ["STUDIO_RESTORATION_VISIBILITY"] = "0.6"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Auto-detect NVIDIA GPU and ensure PyTorch with CUDA is installed
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available() and torch.version.cuda)
    except Exception:
        cuda_ok = False

    if not cuda_ok and shutil.which("nvidia-smi"):
        print("Detected NVIDIA GPU hardware. Ensuring PyTorch with CUDA acceleration is active…", flush=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"
        ], check=False)

    # 1. Verify WebRTC & Studio dependencies
    pkgs = [
        "aiortc", "av", "pydantic_settings", "argon2", "onnxruntime",
        "rich", "pykalman", "albumentations", "tyro"
    ]
    missing = []
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing studio dependencies: {' '.join(missing)}…", flush=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "onnxruntime", "pydantic-settings", "argon2-cffi", "aiortc", "av",
            "python-multipart", "PyJWT", "rich", "pykalman", "albumentations", "tyro",
        ], check=True)
        print("Studio dependencies installed.", flush=True)
    else:
        print("Studio dependencies verified.", flush=True)

    if args.mode == "swap":
        os.environ["STUDIO_BACKEND"] = "inswapper"
        model_p = ensure_models_for_swap()
        os.environ["STUDIO_MODEL_PATH"] = str(model_p)
        try:
            import insightface
        except ImportError:
            print("Installing insightface…", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "insightface"], check=True)
        print("\nTarget Neural Backend: INSWAPPER (DeepFaceLive / FaceFusion Live Stream Face Swap · 0s Delay)", flush=True)
    else:
        os.environ["STUDIO_BACKEND"] = "liveportrait"
        # 2. Ensure LivePortrait neural models are present
        liveportrait_dir = Path("/kaggle/working/LivePortrait")
        if not (liveportrait_dir / "src" / "live_portrait_wrapper.py").exists() and not (ROOT / "third_party" / "LivePortrait" / "src").exists():
            print("LivePortrait not found. Initializing setup…", flush=True)
            from tools.setup_liveportrait import main as setup_lp
            setup_lp()

        # Ensure ROOT is first in sys.path so 'app' always refers to eidomira/app, not LivePortrait/app.py
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
        sys.path.insert(0, str(ROOT))

        if liveportrait_dir.exists() and str(liveportrait_dir) not in sys.path:
            sys.path.append(str(liveportrait_dir))

        # Apply hardware fallback patch so LivePortrait runs smoothly on both CPU and GPU
        wrapper_file = liveportrait_dir / "src" / "live_portrait_wrapper.py"
        if wrapper_file.exists():
            try:
                w = wrapper_file.read_text()
                old_cond = "if inference_cfg.flag_force_cpu:"
                new_cond = "if inference_cfg.flag_force_cpu or not (hasattr(torch, 'cuda') and torch.cuda.is_available() and getattr(torch.version, 'cuda', None)):\n            inference_cfg.flag_use_half_precision = False\n            inference_cfg.flag_force_cpu = True"
                if old_cond in w and "torch.cuda.is_available()" not in w:
                    w = w.replace(old_cond, new_cond)
                    wrapper_file.write_text(w)
            except Exception as exc:
                print(f"Notice on wrapper patch: {exc}", flush=True)

        cropper_file = liveportrait_dir / "src" / "utils" / "cropper.py"
        if cropper_file.exists():
            try:
                c = cropper_file.read_text()
                old_crop = "if flag_force_cpu:"
                new_crop = "if flag_force_cpu or not (hasattr(torch, 'cuda') and torch.cuda.is_available() and getattr(torch.version, 'cuda', None)):"
                if old_crop in c and "torch.cuda.is_available()" not in c:
                    c = c.replace(old_crop, new_crop)
                    cropper_file.write_text(c)
            except Exception as exc:
                print(f"Notice on cropper patch: {exc}", flush=True)

        print("\nTarget Neural Backend: LIVEPORTRAIT (Kling AI Photo Animation)", flush=True)

    # 3. Setup Cloudflare Tunnel binary
    cloudflared_path = download_cloudflared()
    if cloudflared_path is None or not Path(cloudflared_path).exists():
        for cand in [Path("/kaggle/working/cloudflared"), ROOT / "cloudflared", Path("/usr/local/bin/cloudflared")]:
            if cand.exists():
                cloudflared_path = cand
                break

    if cloudflared_path is None or not Path(cloudflared_path).exists():
        print("Error: Cloudflare tunnel binary not found.", file=sys.stderr)
        return 1

    # 4. Open Cloudflare Tunnel
    print("Opening secure HTTPS tunnel for Eidomira Studio…", flush=True)
    log_file = Path("/tmp/cloudflared_studio.log")
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass

    tunnel_cmd = [
        str(cloudflared_path), "tunnel",
        "--no-autoupdate",
        "--url", "http://127.0.0.1:8000",
        "--logfile", str(log_file),
    ]
    tunnel_proc = subprocess.Popen(tunnel_cmd)

    public_url = None
    start_time = time.time()
    while time.time() - start_time < 30:
        if log_file.exists():
            try:
                content = log_file.read_text(errors="ignore")
                match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", content)
                if match:
                    public_url = match.group(0)
                    print(f"[Tunnel Active]: {public_url}", flush=True)
                    break
            except Exception:
                pass
        time.sleep(0.5)

    # 5. Launch Uvicorn in background thread and wait for 200 OK before announcing
    import uvicorn
    import traceback

    try:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
        sys.path.insert(0, str(ROOT))

        print(f"\n--- Initializing {args.mode.upper()} Neural Engine ---", flush=True)
        from app.engines.factory import create_engine
        warmup_engine = create_engine()
        if warmup_engine.name == "inswapper":
            print("🚀 INSWAPPER LIVE WEBCAM SWAP READY ON GPU (0s Latency / TikTok Style)!\n", flush=True)
        elif getattr(warmup_engine, "_initialized", False):
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available() and getattr(torch.version, "cuda", None):
                print("🚀 LIVEPORTRAIT NEURAL ENGINE READY ON GPU (CUDA)!\n", flush=True)
            else:
                print("⚡ LIVEPORTRAIT NEURAL ENGINE READY ON CPU!\n", flush=True)
        else:
            err = getattr(warmup_engine, "_init_error", "Unknown initialization failure")
            print(f"⚠️ Engine running in fallback mode: {err}\n", flush=True)

        from app.main import app as fastapi_app

        config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        print("Waiting for Eidomira web server to respond on port 8000…", flush=True)
        server_ready = False
        start_wait = time.time()
        while time.time() - start_wait < 60:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/app", timeout=1.0) as resp:
                    if resp.status == 200:
                        server_ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if server_ready and public_url:
            print("\n" + "#" * 72, flush=True)
            print("  🎉 EIDOMIRA LIVE WEBCAM STUDIO IS 100% READY!", flush=True)
            print("#" * 72, flush=True)
            print(f"\n  👉 OPEN THIS LINK ON YOUR MAC BROWSER (Chrome/Safari):", flush=True)
            print(f"     {public_url}/app\n", flush=True)
            print(f"  🔑 DEMO PASSWORD: eidomira-demo-2026\n", flush=True)
            print(f"  📸 TO SHARE LIVE OUTPUT FOR AGENT INSPECTION:", flush=True)
            print(f"     1. In Eidomira Studio, click '📸 Snapshot' (or press S) to save", flush=True)
            print(f"        a 3-panel comparison PNG directly to your computer, then drop it into Arena chat.", flush=True)
            print(f"     2. Or in a new Kaggle notebook cell, run:", flush=True)
            print(f"        from IPython.display import Image, display; display(Image('captures/monitor_side_by_side.jpg'))", flush=True)
            print("#" * 72 + "\n", flush=True)
        elif not server_ready:
            print("⚠️ Warning: Server did not respond within 60 seconds on port 8000.", flush=True)

        server_thread.join()

    except Exception as exc:
        print(f"\n[Server Error]: {exc}", flush=True)
        traceback.print_exc()
        time.sleep(30)
    finally:
        tunnel_proc.terminate()


if __name__ == "__main__":
    main()
