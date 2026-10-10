#!/usr/bin/env python3
"""Setup script for LivePortrait neural portrait reenactment engine.

Downloads repository and pretrained weights from Hugging Face for zero-friction
Cloud GPU execution on Kaggle / Colab / RunPod.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = Path("/kaggle/working/LivePortrait") if Path("/kaggle/working").exists() else (ROOT / "third_party" / "LivePortrait")
WEIGHTS_DIR = TARGET_DIR / "pretrained_weights"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main():
    print("=" * 72)
    print("  SETTING UP LIVEPORTRAIT NEURAL REENACTMENT ENGINE")
    print("=" * 72)

    # 1. Clone repository if not present
    if not (TARGET_DIR / "src" / "live_portrait_wrapper.py").exists():
        TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning LivePortrait into {TARGET_DIR}…")
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/KlingAIResearch/LivePortrait.git", str(TARGET_DIR)])
    else:
        print(f"LivePortrait repository already present at {TARGET_DIR}")

    # 2. Install required packages
    print("\nInstalling LivePortrait and Studio dependencies…")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "pyyaml", "scipy", "imageio", "imageio-ffmpeg", "albumentations", "tyro",
        "fastapi", "uvicorn", "pydantic-settings", "python-multipart", "PyJWT", "argon2-cffi",
        "aiortc", "av",
    ], check=True)

    # 3. Download pretrained weights from Hugging Face using official Python SDK
    expected_checkpoint = WEIGHTS_DIR / "liveportrait" / "base_models" / "spade_generator.pth"
    if not expected_checkpoint.exists():
        print(f"\nDownloading LivePortrait human pretrained weights into {WEIGHTS_DIR}…", flush=True)
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
            print("Pretrained weights downloaded successfully.", flush=True)
        except Exception as exc:
            print(f"Python SDK download notice: {exc}. Trying fallback…", flush=True)
            try:
                run_cmd(["hf", "download", "KlingTeam/LivePortrait", "--local-dir", str(WEIGHTS_DIR)])
            except Exception:
                run_cmd(["huggingface-cli", "download", "KlingTeam/LivePortrait", "--local-dir", str(WEIGHTS_DIR)])
    else:
        print(f"\nPretrained weights already present at {WEIGHTS_DIR}", flush=True)

    # 4. Setup Cloudflare tunnel binary
    cf = Path("/kaggle/working/cloudflared") if Path("/kaggle/working").exists() else (ROOT / "cloudflared")
    if not (cf.exists() and cf.stat().st_size > 10_000_000):
        print(f"\nFetching Cloudflare tunnel binary into {cf}…", flush=True)
        cf.parent.mkdir(parents=True, exist_ok=True)
        success = False
        for cmd in [
            ["curl", "-fSL", "--connect-timeout", "15", "--max-time", "45",
             "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
             "-o", str(cf)],
            ["wget", "-q", "--timeout=30",
             "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
             "-O", str(cf)],
        ]:
            try:
                subprocess.run(cmd, check=True)
                if cf.exists() and cf.stat().st_size > 10_000_000:
                    cf.chmod(0o755)
                    print("Cloudflare tunnel binary ready.", flush=True)
                    success = True
                    break
            except Exception:
                continue
        if not success:
            print("Notice: Cloudflare binary download will be retried at launch.", flush=True)
    else:
        print(f"Cloudflare tunnel binary already present at {cf}", flush=True)

    print("\n✅ LivePortrait setup completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
