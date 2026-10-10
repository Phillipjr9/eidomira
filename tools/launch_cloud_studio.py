#!/usr/bin/env python3
"""tools/launch_cloud_studio.py — Launch Cloud GPU WebRTC Studio with Public HTTPS Tunnel.

Runs the Eidomira Neural WebRTC Studio on a Cloud GPU (e.g. Kaggle Tesla T4),
exposes it via an instant, secure Cloudflare Tunnel (HTTPS with full camera access),
and prints the clickable live studio URL for your Mac browser.

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
import time
import urllib.request
from pathlib import Path

# Ensure immediate line flushing in Jupyter/Kaggle environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
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


def main():
    print("=" * 72, flush=True)
    print("  EIDOMIRA CLOUD GPU LIVE WEBRTC STUDIO", flush=True)
    print("=" * 72, flush=True)

    # 0. Verify WebRTC & Studio dependencies
    try:
        import aiortc
        import av
        import pydantic_settings
        import argon2
        print("Studio dependencies verified.", flush=True)
    except ImportError:
        print("Installing studio dependencies (pydantic-settings, argon2, aiortc, av)…", flush=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "pydantic-settings", "argon2-cffi", "aiortc", "av", "python-multipart", "PyJWT",
        ], check=True)
        print("Studio dependencies installed.", flush=True)

    # Determine neural backend
    liveportrait_dir = Path("/kaggle/working/LivePortrait")
    selected_backend = os.environ.get("STUDIO_BACKEND")
    if not selected_backend:
        if (liveportrait_dir / "src" / "live_portrait_wrapper.py").exists() or (ROOT / "third_party" / "LivePortrait" / "src").exists():
            selected_backend = "liveportrait"
        else:
            selected_backend = "inswapper"

    print(f"\nTarget Neural Backend: {selected_backend.upper()}", flush=True)

    # 1. Preload CUDA only for legacy inswapper
    if selected_backend == "inswapper":
        try:
            from tools.run_real_swap import preload_cuda
            preload_cuda()
        except Exception:
            pass

    # 2. Only download legacy inswapper models if inswapper backend is explicitly chosen
    model_path = ROOT / "models" / "inswapper_128.onnx"
    if selected_backend == "inswapper":
        from tools.run_real_swap import download_file, MODEL_URL, RESTORER_URL
        restorer_path = ROOT / "models" / "gfpgan_1.4.onnx"
        try:
            download_file(MODEL_URL, model_path, "InSwapper-128")
            download_file(RESTORER_URL, restorer_path, "GFPGAN-1.4")
        except Exception as exc:
            print(f"Model notice: {exc}")

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

    # 4. Open Cloudflare Tunnel and acquire live public URL
    print("Opening secure HTTPS tunnel for camera access…", flush=True)
    tunnel_cmd = [
        str(cloudflared_path), "tunnel",
        "--no-autoupdate",
        "--url", "http://127.0.0.1:8000",
    ]
    tunnel_proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    public_url = None
    start_time = time.time()
    while time.time() - start_time < 25:
        line = tunnel_proc.stdout.readline()
        if not line:
            continue
        match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
        if match:
            public_url = match.group(0)
            break

    # Keep tunnel output drained in background so the pipe buffer never blocks
    import threading
    def _keep_tunnel_alive(proc):
        try:
            while proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    break
        except Exception:
            pass

    threading.Thread(target=_keep_tunnel_alive, args=(tunnel_proc,), daemon=True).start()

    if public_url:
        print("\n" + "#" * 72, flush=True)
        print("  🎉 EIDOMIRA LIVE WEBCAM STUDIO IS LIVE!", flush=True)
        print("#" * 72, flush=True)
        print(f"\n  👉 OPEN THIS LINK ON YOUR MAC BROWSER (Chrome/Safari):", flush=True)
        print(f"     {public_url}/app\n", flush=True)
        print(f"  🔑 DEMO PASSWORD: eidomira-demo-2026", flush=True)
        print("#" * 72, flush=True)
        print("\nStarting LivePortrait neural server on port 8000…\n", flush=True)
    else:
        print("Starting LivePortrait neural server on port 8000…", flush=True)

    # 5. Configure environment and launch Uvicorn directly in foreground
    os.environ["STUDIO_BACKEND"] = selected_backend
    os.environ["STUDIO_ALLOWED_ORIGINS"] = "*"
    os.environ["STUDIO_DEMO_LOGIN"] = "true"
    os.environ["STUDIO_DEMO_PASSWORD"] = "eidomira-demo-2026"
    os.environ["STUDIO_REQUIRE_SELF_VERIFICATION"] = "false"
    os.environ["STUDIO_RESTORATION_VISIBILITY"] = "0.6"
    os.environ["STUDIO_MODEL_PATH"] = str(model_path)
    os.environ["PYTHONUNBUFFERED"] = "1"

    if liveportrait_dir.exists() and str(liveportrait_dir) not in sys.path:
        sys.path.insert(0, str(liveportrait_dir))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import uvicorn
    import traceback
    try:
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
    except Exception as exc:
        print(f"\n[Server Error]: {exc}", flush=True)
        traceback.print_exc()
        time.sleep(10)
    finally:
        tunnel_proc.terminate()


if __name__ == "__main__":
    main()
