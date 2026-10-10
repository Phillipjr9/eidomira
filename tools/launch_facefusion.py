#!/usr/bin/env python3
"""tools/launch_facefusion.py — 1-Click FaceFusion WebUI Cloud GPU Runner.

Clones the official FaceFusion repository (https://github.com/facefusion/facefusion),
configures CUDA execution providers on the Tesla T4 GPU, opens a secure Cloudflare
HTTPS tunnel, and provides the complete FaceFusion video-to-video swapping platform
in your browser.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FF_DIR = Path("/kaggle/working/facefusion") if Path("/kaggle/working").exists() else (ROOT / "third_party" / "facefusion")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def download_cloudflared() -> Path:
    cf = Path("/kaggle/working/cloudflared") if Path("/kaggle/working").exists() else (ROOT / "cloudflared")
    if cf.exists() and cf.stat().st_size > 10_000_000:
        return cf

    cf.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    for tool in ["curl", "wget"]:
        if shutil.which(tool):
            try:
                if tool == "curl":
                    run_cmd(["curl", "-fSL", "--connect-timeout", "15", url, "-o", str(cf)])
                else:
                    run_cmd(["wget", "-q", "--timeout=30", url, "-O", str(cf)])
                if cf.exists() and cf.stat().st_size > 10_000_000:
                    cf.chmod(0o755)
                    return cf
            except Exception:
                continue
    return cf


def main():
    print("=" * 72)
    print("  LAUNCHING OFFICIAL FACEFUSION PLATFORM ON CLOUD GPU")
    print("=" * 72, flush=True)

    # 1. Clone official FaceFusion repository
    if not (FF_DIR / "facefusion.py").exists():
        FF_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning official FaceFusion repository into {FF_DIR}…", flush=True)
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/facefusion/facefusion.git", str(FF_DIR)])
    else:
        print(f"FaceFusion already present at {FF_DIR}", flush=True)

    # 2. Install FaceFusion requirements
    print("\nChecking FaceFusion dependencies…", flush=True)
    pkgs = ["gradio", "gradio_rangeslider", "cv2", "onnx", "onnxruntime", "tqdm", "scipy", "psutil"]
    missing = []
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            if pkg == "cv2":
                missing.append("opencv-python-headless")
            elif pkg == "gradio_rangeslider":
                missing.append("gradio-rangeslider")
            elif pkg == "onnxruntime":
                missing.append("onnxruntime-gpu")
            else:
                missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {' '.join(missing)}…", flush=True)
        for pkg in missing:
            print(f"  Installing {pkg}…", flush=True)
            if pkg in ("gradio", "gradio-rangeslider"):
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", pkg], check=True)
            else:
                subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
        print("✅ Dependencies successfully installed!", flush=True)
    else:
        print("All dependencies already verified.", flush=True)

    # 3. Setup Cloudflare tunnel
    cf_path = download_cloudflared()
    if not cf_path.exists():
        print("Error: Could not obtain cloudflared binary.", file=sys.stderr)
        return 1

    log_file = Path("/tmp/cloudflared_ff.log")
    if log_file.exists():
        log_file.unlink()

    print("\nOpening secure HTTPS tunnel for FaceFusion UI…", flush=True)
    tunnel_proc = subprocess.Popen(
        [str(cf_path), "tunnel", "--url", "http://127.0.0.1:7860", "--logfile", str(log_file), "--no-autoupdate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    public_url = None
    start_time = time.time()
    while time.time() - start_time < 30:
        if log_file.exists():
            try:
                content = log_file.read_text(errors="ignore")
                match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", content)
                if match:
                    public_url = match.group(0)
                    break
            except Exception:
                pass
        time.sleep(0.5)

    if public_url:
        print("\n" + "#" * 72)
        print("  🎉 FACEFUSION PLATFORM IS LIVE ON CLOUD GPU!")
        print("#" * 72)
        print(f"\n  👉 OPEN THIS LINK IN YOUR MAC BROWSER:")
        print(f"     {public_url}\n")
        print("  1. Drop your Source Face image (e.g. Elon)")
        print("  2. Drop your Target Video (any real moving video clip)")
        print("  3. Select processors (Face Swapper + Face Enhancer)")
        print("  4. Click START to render your full moving video!")
        print("#" * 72 + "\n", flush=True)
    else:
        print("Starting FaceFusion on port 7860…", flush=True)

    # 4. Run FaceFusion WebUI
    try:
        os.chdir(str(FF_DIR))
        subprocess.run([sys.executable, "facefusion.py", "run", "--execution-providers", "cuda", "cpu"], check=True)
    except KeyboardInterrupt:
        print("\nStopping FaceFusion…")
    finally:
        tunnel_proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
