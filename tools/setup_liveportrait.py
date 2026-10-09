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
    print("\nInstalling LivePortrait dependencies…")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "pyyaml", "scipy", "imageio", "imageio-ffmpeg", "albumentations", "tyro",
        "huggingface_hub[cli]",
    ], check=True)

    # 3. Download pretrained weights from Hugging Face using official Python SDK
    expected_checkpoint = WEIGHTS_DIR / "liveportrait" / "base_models" / "spade_generator.pth"
    if not expected_checkpoint.exists():
        print(f"\nDownloading LivePortrait pretrained weights into {WEIGHTS_DIR}…")
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="KlingTeam/LivePortrait",
                local_dir=str(WEIGHTS_DIR),
                ignore_patterns=["*.git*", "README.md", "docs/*"],
            )
        except Exception as exc:
            print(f"Python SDK download notice: {exc}. Trying hf command…")
            try:
                run_cmd(["hf", "download", "KlingTeam/LivePortrait", "--local-dir", str(WEIGHTS_DIR)])
            except Exception:
                run_cmd(["huggingface-cli", "download", "KlingTeam/LivePortrait", "--local-dir", str(WEIGHTS_DIR)])
    else:
        print(f"\nPretrained weights already present at {WEIGHTS_DIR}")

    print("\n✅ LivePortrait setup completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
