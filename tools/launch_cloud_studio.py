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
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_BIN = ROOT / "cloudflared"


def download_cloudflared():
    for p in [Path("/kaggle/working/cloudflared"), CLOUDFLARED_BIN, Path("/usr/local/bin/cloudflared")]:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    print("Downloading Cloudflare Tunnel binary…", flush=True)
    target = Path("/kaggle/working/cloudflared") if Path("/kaggle/working").exists() else CLOUDFLARED_BIN
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["wget", "-q", "-c", CLOUDFLARED_URL, "-O", str(target)], check=True)
        target.chmod(0o755)
        print("Cloudflare Tunnel ready.", flush=True)
        return target
    except Exception:
        pass
    try:
        subprocess.run(["curl", "-sL", CLOUDFLARED_URL, "-o", str(target)], check=True)
        target.chmod(0o755)
        print("Cloudflare Tunnel ready.", flush=True)
        return target
    except Exception as exc:
        print(f"Cloudflare download notice: {exc}", flush=True)
        return None


def main():
    print("=" * 72, flush=True)
    print("  EIDOMIRA CLOUD GPU LIVE WEBRTC STUDIO", flush=True)
    print("=" * 72, flush=True)

    # 0. Verify WebRTC dependencies
    try:
        import aiortc
        import av
        print("WebRTC streaming dependencies verified.", flush=True)
    except ImportError:
        print("Notice: aiortc/av streaming dependencies not imported.", flush=True)

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

    # 4. Start Uvicorn Server in background
    print(f"\nStarting Eidomira Neural Studio on port 8000 (Backend: {selected_backend.upper()})…")
    env = os.environ.copy()
    env["STUDIO_BACKEND"] = selected_backend
    env["STUDIO_ALLOWED_ORIGINS"] = "*"
    env["STUDIO_DEMO_LOGIN"] = "true"
    env["STUDIO_DEMO_PASSWORD"] = "eidomira-demo-2026"
    env["STUDIO_REQUIRE_SELF_VERIFICATION"] = "false"
    env["STUDIO_RESTORATION_VISIBILITY"] = "0.6"
    env["STUDIO_MODEL_PATH"] = str(model_path)
    python_paths = [str(ROOT)]
    if liveportrait_dir.exists():
        python_paths.insert(0, str(liveportrait_dir))
    if (ROOT / "third_party" / "LivePortrait").exists():
        python_paths.insert(0, str(ROOT / "third_party" / "LivePortrait"))
    env["PYTHONPATH"] = ":".join(python_paths) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"

    # Pre-seed demo accounts in the database
    try:
        import app.demo as demo
        demo.ensure_accounts()
        print("  Demo accounts ready (demo@eidomira.test / eidomira-demo-2026)")
    except Exception as exc:
        print(f"  Demo account notice: {exc}")

    server_cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "1",
    ]
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for server to finish initializing and start listening
    server_healthy = False
    start_wait = time.time()
    while time.time() - start_wait < 35:
        if server_proc.poll() is not None:
            rest = server_proc.stdout.read()
            print(f"\nBackend process exited with code {server_proc.returncode}:\n{rest}")
            return 1
        line = server_proc.stdout.readline()
        if line:
            print(f"  [Backend] {line.strip()}")
            if "Application startup complete" in line or "Uvicorn running on" in line:
                server_healthy = True
                break
        time.sleep(0.2)

    if not server_healthy:
        print("Warning: Backend took longer than expected to initialize. Check logs above.")
    else:
        print("Backend server verified and listening on port 8000.")

    # 5. Start Cloudflare Tunnel
    print("Opening secure HTTPS tunnel for camera access…")
    tunnel_cmd = [
        str(cloudflared_path), "tunnel",
        "--url", "http://127.0.0.1:8000",
    ]
    tunnel_proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    public_url = None
    # Parse tunnel output to find the https://xxxx.trycloudflare.com link
    start_time = time.time()
    while time.time() - start_time < 30:
        line = tunnel_proc.stdout.readline()
        if not line:
            continue
        match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        print("\n" + "#" * 72)
        print("  🎉 EIDOMIRA LIVE WEBCAM STUDIO IS LIVE!")
        print("#" * 72)
        print(f"\n  👉 OPEN THIS LINK ON YOUR MAC BROWSER (Chrome/Safari):")
        print(f"     {public_url}/app\n")
        print(f"  🔑 DEMO PASSWORD: eidomira-demo-2026")
        print("#" * 72)
        print("\nStreaming live WebRTC GPU pipeline at 30+ FPS… Keep this cell running!")
    else:
        print("Warning: Could not automatically detect tunnel URL. Check tunnel logs.")

    # Stream logs
    try:
        while True:
            line = tunnel_proc.stdout.readline()
            if not line:
                break
    except KeyboardInterrupt:
        print("\nStopping studio…")
        tunnel_proc.terminate()
        server_proc.terminate()


if __name__ == "__main__":
    main()
