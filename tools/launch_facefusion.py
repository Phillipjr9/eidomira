#!/usr/bin/env python3
"""tools/launch_facefusion.py — 1-Click FaceFusion WebUI Cloud GPU Runner.

Clones the official FaceFusion repository (https://github.com/facefusion/facefusion),
enables public sharing, and launches the FaceFusion video-to-video swapping platform
directly on the Tesla T4 GPU with an instant public link.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FF_DIR = Path("/kaggle/working/facefusion") if Path("/kaggle/working").exists() else (ROOT / "third_party" / "facefusion")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


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

    # 3. Enable Gradio public share in FaceFusion layouts
    layout_file = FF_DIR / "facefusion" / "uis" / "layouts" / "default.py"
    if layout_file.exists():
        try:
            content = layout_file.read_text()
            if "share = True" not in content:
                content = content.replace("ui.launch(", "ui.launch(share = True, ")
                layout_file.write_text(content)
                print("Enabled public Gradio sharing in FaceFusion.", flush=True)
        except Exception as exc:
            print(f"Note on layout patch: {exc}", flush=True)

    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SHARE"] = "True"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # 4. Optional Cloudflare tunnel if binary already present
    tunnel_proc = None
    cf_cand = Path("/kaggle/working/cloudflared")
    if cf_cand.exists() and cf_cand.stat().st_size > 10_000_000:
        try:
            tunnel_proc = subprocess.Popen(
                [str(cf_cand), "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:7860"],
            )
        except Exception:
            pass

    # 5. Launch FaceFusion WebUI
    print("\n" + "#" * 72)
    print("  🚀 STARTING OFFICIAL FACEFUSION ON TESLA T4 GPU...")
    print("  Look for the public link below (https://xxxx.gradio.live)")
    print("#" * 72 + "\n", flush=True)

    try:
        os.chdir(str(FF_DIR))
        subprocess.run([sys.executable, "facefusion.py", "run"], check=True)
    except KeyboardInterrupt:
        print("\nStopping FaceFusion…")
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
