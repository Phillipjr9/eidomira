#!/usr/bin/env python3
"""tools/launch_kling_control.py — Official Kling AI LivePortrait Reference Motion Studio.

Launches the official Kling AI LivePortrait WebUI on Cloud GPU (Tesla T4) with
instant Cloudflare HTTPS tunnel.

Features:
- Character Photo to Reference Motion video driving (head pose, eye gaze, mouth, expressions).
- Direct real-time 3D keypoint stitching.
- Full interactive web studio with sliders and preview.
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
from pathlib import Path

# Force unbuffered streaming output in Jupyter / Kaggle
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
LP_DIR = Path("/kaggle/working/LivePortrait") if Path("/kaggle/working").exists() else (ROOT / "third_party" / "LivePortrait")
WEIGHTS_DIR = LP_DIR / "pretrained_weights"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def download_cloudflared() -> Path:
    for cand in [Path("/kaggle/working/cloudflared"), ROOT / "cloudflared", Path("/usr/local/bin/cloudflared")]:
        if cand.exists() and cand.stat().st_size > 10_000_000:
            return cand

    target = Path("/kaggle/working/cloudflared") if Path("/kaggle/working").exists() else (ROOT / "cloudflared")
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
                return target
        except Exception:
            continue
    return target


def is_port_open(port: int = 7860) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def main():
    print("=" * 72)
    print("  LAUNCHING KLING AI LIVEPORTRAIT (REFERENCE MOTION CONTROL)")
    print(f"  Python: {sys.version.split()[0]} | Directory: {LP_DIR}")
    print("=" * 72, flush=True)

    # 1. Clone Kling AI LivePortrait
    if not (LP_DIR / "app.py").exists():
        LP_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning Kling AI LivePortrait repository into {LP_DIR}…", flush=True)
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/KlingAIResearch/LivePortrait.git", str(LP_DIR)])
    else:
        print(f"Kling AI LivePortrait already present at {LP_DIR}", flush=True)

    # 2. Check complete dependencies (including rich, pykalman, tyro)
    print("\nChecking LivePortrait dependencies…", flush=True)
    pkgs = [
        "gradio", "pyyaml", "scipy", "imageio", "imageio-ffmpeg",
        "albumentations", "tyro", "huggingface_hub", "onnxruntime",
        "rich", "pykalman", "lmdb"
    ]
    missing = []
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {' '.join(missing)}…", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing], check=True)
        print("✅ Dependencies ready!", flush=True)
    else:
        print("All dependencies already verified.", flush=True)

    # 3. Download official pretrained weights from Hugging Face
    expected_pth = WEIGHTS_DIR / "liveportrait" / "base_models" / "spade_generator.pth"
    if not expected_pth.exists():
        print(f"\nDownloading Kling AI LivePortrait model weights into {WEIGHTS_DIR}…", flush=True)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="KlingTeam/LivePortrait",
                local_dir=str(WEIGHTS_DIR),
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "liveportrait/base_models/*",
                    "liveportrait/retargeting_models/*",
                    "liveportrait/landmark.onnx",
                    "insightface/models/buffalo_l/*",
                ],
            )
            print("✅ Model weights downloaded successfully!", flush=True)
        except Exception as exc:
            print(f"Notice: Python download error ({exc}). Retrying via CLI…", flush=True)
            subprocess.run([
                "huggingface-cli", "download", "KlingTeam/LivePortrait",
                "--local-dir", str(WEIGHTS_DIR),
                "--exclude", "*.git*", "README.md", "docs",
            ], check=True)
    else:
        print("\nPretrained model weights already present.", flush=True)

    # 4. Open Cloudflare HTTPS tunnel for port 7860
    cf_bin = download_cloudflared()
    log_file = Path("/tmp/cloudflared_kling.log")
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass

    print("\nOpening secure Cloudflare HTTPS tunnel…", flush=True)
    tunnel_cmd = [
        str(cf_bin), "tunnel",
        "--no-autoupdate",
        "--url", "http://127.0.0.1:7860",
        "--logfile", str(log_file),
    ]
    tunnel_proc = subprocess.Popen(tunnel_cmd)

    public_url = None
    start_time = time.time()
    while time.time() - start_time < 20:
        if log_file.exists():
            try:
                txt = log_file.read_text(errors="ignore")
                match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", txt)
                if match:
                    public_url = match.group(0)
                    break
            except Exception:
                pass
        time.sleep(0.5)

    # 5. Start Kling AI LivePortrait in background and wait for port 7860
    os.chdir(str(LP_DIR))
    print("\nLoading Kling AI neural weights into GPU memory…", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "app.py", "--server_port", "7860", "--server_name", "0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Background thread to stream server logs
    def stream_logs():
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)

    log_thread = threading.Thread(target=stream_logs, daemon=True)
    log_thread.start()

    # Wait until port 7860 is listening before announcing ready
    print("Waiting for server to bind to port 7860...", flush=True)
    server_ready = False
    start_wait = time.time()
    while time.time() - start_wait < 60:
        if is_port_open(7860):
            server_ready = True
            break
        if proc.poll() is not None:
            print(f"\nError: LivePortrait app exited with code {proc.returncode}", flush=True)
            tunnel_proc.terminate()
            return 1
        time.sleep(1.0)

    if server_ready and public_url:
        print("\n" + "#" * 72, flush=True)
        print("  🎉 KLING AI LIVEPORTRAIT IS 100% READY AND LISTENING!", flush=True)
        print("#" * 72)
        print(f"\n  👉 CLICK THIS LINK IN YOUR MAC BROWSER (Chrome/Safari):", flush=True)
        print(f"     {public_url}\n", flush=True)
        print("  HOW TO USE:")
        print("  1. Upload your character portrait (e.g. Elon Musk photo)")
        print("  2. Upload your reference action/driving video (or webcam)")
        print("  3. Click 'Animate' / 'Generate'!")
        print("  It will transfer all expressions, head movement, eye gaze, and speech!")
        print("#" * 72 + "\n", flush=True)
    elif not server_ready:
        print("⚠️ Warning: Port 7860 timed out. Inspecting output above.", flush=True)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping Kling AI LivePortrait…")
    finally:
        tunnel_proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
