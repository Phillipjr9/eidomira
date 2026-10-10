#!/usr/bin/env python3
"""tools/generate_video.py — Offline Face Swap + Voice Lip Sync HD Video Generator.

Takes a source identity (e.g. Elon Musk), swaps it onto a target portrait with
bone adaptation and GFPGAN HD restoration, then synchronizes speech audio
(e.g. speech-elon.mp3) with neural lip-sync to generate a full talking video.

Usage:
    python tools/generate_video.py
    python tools/generate_video.py --source static/celebrity-01.jpg --target static/businessman-neutral.jpg --audio static/speech-elon.mp3 --output my_video.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_dependencies():
    print("Verifying pipeline dependencies…", flush=True)
    required = ["onnxruntime", "librosa", "soundfile", "cv2"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    try:
        import insightface
    except ImportError:
        missing.append("insightface")

    if missing:
        print(f"Installing missing dependencies: {missing}…", flush=True)
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"]
        for m in missing:
            if m == "onnxruntime":
                cmd.extend(["onnxruntime-gpu", "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12"])
            elif m == "cv2":
                cmd.append("opencv-python-headless")
            elif m == "insightface":
                cmd.extend(["--no-deps", "insightface"])
            else:
                cmd.append(m)
        subprocess.run(cmd, check=True)
        print("Dependencies installed successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Full HD Face Swap + Audio Lip-Sync Video Pipeline")
    parser.add_argument("--source", type=Path, default=ROOT / "static" / "celebrity-01.jpg", help="Source face image")
    parser.add_argument("--target", type=Path, default=ROOT / "static" / "businessman-neutral.jpg", help="Target body/portrait")
    parser.add_argument("--audio", type=Path, default=ROOT / "static" / "speech-elon.mp3", help="Driving speech audio")
    parser.add_argument("--output", type=Path, default=ROOT / "final_swapped_lipsync.mp4", help="Output MP4 video")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate")
    args = parser.parse_args()

    print("=" * 72)
    print("  EIDOMIRA OFFLINE VIDEO PIPELINE: SWAP + HD RESTORATION + LIP SYNC")
    print("=" * 72)
    print(f"  Source face: {args.source}")
    print(f"  Target body: {args.target}")
    print(f"  Speech audio: {args.audio}")
    print(f"  Final video: {args.output}\n")

    check_dependencies()

    # Step 1: Run High-Definition Neural Face Swap & GFPGAN Restoration
    print("\n" + "=" * 72)
    print("  STAGE 1: HIGH-DEFINITION NEURAL FACE SWAP & DETAIL RESTORATION")
    print("=" * 72)
    swapped_image = ROOT / "swapped_result.jpg"
    swap_cmd = [
        sys.executable, str(ROOT / "tools" / "run_real_swap.py"),
        "--source", str(args.source),
        "--target", str(args.target),
        "--output", str(swapped_image),
    ]
    print(f"Executing: {' '.join(swap_cmd)}\n", flush=True)
    subprocess.run(swap_cmd, check=True)

    if not swapped_image.exists():
        print("Error: Swapped face was not produced.", file=sys.stderr)
        return 1

    print(f"\n✅ Stage 1 complete! High-definition swap saved to: {swapped_image}")

    # Step 2: Run Audio Lip-Sync with Mouth GFPGAN Enhancement
    print("\n" + "=" * 72)
    print("  STAGE 2: PHONEME-ALIGNED LIP-SYNC & MOUTH DETAIL RESTORATION")
    print("=" * 72)
    lipsync_cmd = [
        sys.executable, str(ROOT / "tools" / "run_lipsync.py"),
        "--face", str(swapped_image),
        "--audio", str(args.audio),
        "--output", str(args.output),
        "--fps", str(args.fps),
    ]
    print(f"Executing: {' '.join(lipsync_cmd)}\n", flush=True)
    subprocess.run(lipsync_cmd, check=True)

    if not args.output.exists():
        print("Error: Output video was not generated.", file=sys.stderr)
        return 1

    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    print("\n" + "#" * 72)
    print("  🎉 PIPELINE COMPLETE! YOUR VIDEO IS READY:")
    print(f"     File: {args.output.resolve()}")
    print(f"     Size: {file_size_mb:.2f} MB")
    print("#" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
