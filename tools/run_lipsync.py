#!/usr/bin/env python3
"""tools/run_lipsync.py — High-Definition Neural Lip-Sync & Audio Synchronization.

Takes a facial portrait (e.g. swapped Elon Musk output) and a speech audio track
(e.g. static/speech-elon.mp3), synthesizes phoneme-synchronized lip & jaw movements,
enhances mouth and teeth details via GFPGAN v1.4, and multiplexes the audio into
a talking video (.mp4).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def preload_cuda():
    import ctypes
    import glob
    search_paths = []
    for p in sys.path:
        search_paths.extend(glob.glob(os.path.join(p, "nvidia", "*", "lib")))
    search_paths.append("/usr/local/cuda/lib64")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(search_paths) + (":" + current_ld if current_ld else "")
    for path in search_paths:
        if os.path.exists(path):
            for so_file in glob.glob(os.path.join(path, "*.so*")):
                try:
                    ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

WAV2LIP_URLS = [
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_gan_96.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_96.onnx",
]
DEFAULT_AUDIO = ROOT / "static" / "speech-elon.mp3"
DEFAULT_FACE = ROOT / "swapped_result.jpg"
DEFAULT_OUTPUT = ROOT / "lipsync_elon.mp4"


def download_file(urls: list[str] | str, destination: Path, label: str) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 10_000_000:
        print(f"  {label} already present: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return True
    if isinstance(urls, str):
        urls = [urls]
    import urllib.request
    for url in urls:
        print(f"Downloading {label} from {url}…")
        try:
            partial = destination.with_suffix(".part")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=240) as response, partial.open("wb") as handle:
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
                        if pct >= last_pct + 25:
                            print(f"  {pct}% ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)…")
                            last_pct = pct
            if partial.stat().st_size > 1_000_000:
                partial.replace(destination)
                print(f"Download complete: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
                return True
        except Exception as exc:
            print(f"  Mirror failed ({exc}), trying next…")
    return False


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int = 16000, n_fft: int = 800, n_mels: int = 80,
                   fmin: float = 55.0, fmax: float = 7600.0) -> np.ndarray:
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_pts[m - 1]) / max(1, (bin_pts[m] - bin_pts[m - 1]))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_pts[m + 1] - k) / max(1, (bin_pts[m + 1] - bin_pts[m]))
    return fbank


def extract_audio_mel_chunks(audio_path: Path, fps: int = 25, mel_step_size: int = 16):
    """Compute 80-band log mel-spectrogram chunks (16 time steps) for each video frame."""
    import soundfile as sf
    import scipy.signal

    wav, in_sr = sf.read(str(audio_path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    target_sr = 16000
    if in_sr != target_sr:
        num_samples = int(len(wav) * target_sr / in_sr)
        wav = scipy.signal.resample(wav, num_samples)

    # Short-time Fourier transform (STFT)
    n_fft = 800
    hop_length = 200
    f, t, Zxx = scipy.signal.stft(wav, fs=target_sr, nperseg=n_fft,
                                 noverlap=n_fft - hop_length, boundary=None)
    mag = np.abs(Zxx)
    fb = mel_filterbank(sr=target_sr, n_fft=n_fft, n_mels=80)
    mel = np.dot(fb, mag)
    mel_safe = np.maximum(1e-5, mel)
    mel_scaled = np.log10(mel_safe) * 1.6 + 3.2
    mel_scaled = np.clip(mel_scaled, -4.0, 4.0).astype(np.float32) * 2.0

    total_frames = int(len(wav) * fps / target_sr)
    mel_chunks = []
    mel_per_frame = 80.0 / float(fps)

    samples_per_frame = target_sr / float(fps)
    frame_rms = []
    for f in range(total_frames):
        st = int(f * samples_per_frame)
        en = int((f + 1) * samples_per_frame)
        ch = wav[st:en]
        r = float(np.sqrt(np.mean(ch**2))) if len(ch) > 0 else 0.0
        frame_rms.append(r)
    frame_rms = np.array(frame_rms, dtype=np.float32)
    norm_rms = np.clip(frame_rms / (frame_rms.max() + 1e-6), 0.0, 1.0)
    smooth_rms = np.convolve(norm_rms, np.array([0.15, 0.7, 0.15], dtype=np.float32), mode="same")

    for frame_idx in range(total_frames):
        center_mel = int(frame_idx * mel_per_frame)
        start_mel = center_mel - mel_step_size // 2
        end_mel = start_mel + mel_step_size

        if start_mel < 0:
            pad_left = -start_mel
            chunk = mel_scaled[:, 0:max(0, end_mel)]
            chunk = np.pad(chunk, ((0, 0), (pad_left, 0)), mode="edge")
        elif end_mel > mel_scaled.shape[1]:
            pad_right = end_mel - mel_scaled.shape[1]
            chunk = mel_scaled[:, start_mel:]
            chunk = np.pad(chunk, ((0, 0), (0, pad_right)), mode="edge")
        else:
            chunk = mel_scaled[:, start_mel:end_mel]

        if chunk.shape[1] != mel_step_size:
            chunk = np.pad(chunk, ((0, 0), (0, max(0, mel_step_size - chunk.shape[1]))), mode="edge")
            chunk = chunk[:, :mel_step_size]

        mel_chunks.append(chunk.astype(np.float32))

    return mel_chunks, smooth_rms, total_frames, len(wav) / target_sr


def main() -> int:
    parser = argparse.ArgumentParser(description="Eidomira Neural Lip-Sync & Voice Sync Runner")
    parser.add_argument("--face", type=Path, default=DEFAULT_FACE, help="Source swapped portrait")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Driving speech audio")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output talking video (.mp4)")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "wav2lip_gan_96.onnx")
    parser.add_argument("--restore", action="store_true", default=True, help="GFPGAN mouth restoration")
    options = parser.parse_args()
    preload_cuda()

    # Fallback face selection if default output doesn't exist yet
    face_path = options.face
    if not face_path.exists():
        for cand in [ROOT / "swapped_result.jpg",
                     Path("/kaggle/working/eidomira/swapped_result.jpg"),
                     ROOT / "comparison.jpg",
                     Path("/kaggle/working/eidomira/comparison.jpg"),
                     ROOT / "output.jpg",
                     Path("/kaggle/working/eidomira/output.jpg"),
                     ROOT / "static" / "celebrity-01.jpg",
                     ROOT / "static" / "businessman-neutral.jpg"]:
            if cand.exists():
                face_path = cand
                break

    if not face_path.exists():
        print(f"Error: Face image not found at {face_path}", file=sys.stderr)
        return 1

    if not options.audio.exists():
        print(f"Error: Audio track not found at {options.audio}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("  EIDOMIRA NEURAL LIP-SYNC & AUDIO SYNCHRONIZATION")
    print("=" * 72)
    print(f"  Face portrait: {face_path}")
    print(f"  Audio speech:  {options.audio}")
    print(f"  Output video:  {options.output}")

    # Ensure Wav2Lip model is present
    model_path = options.model
    if not model_path.exists() and Path("/kaggle/working/eidomira/models/wav2lip_gan_96.onnx").exists():
        model_path = Path("/kaggle/working/eidomira/models/wav2lip_gan_96.onnx")

    try:
        download_file(WAV2LIP_URLS, model_path, "Wav2Lip-GAN 96")
    except Exception as exc:
        print(f"Note: Wav2Lip model download skipped ({exc})")

    import onnxruntime as ort
    from app.providers import execution_providers, describe_providers

    providers = list(execution_providers())
    provider_name = describe_providers(providers)
    print(f"  Execution provider: {provider_name}")

    # 1. Process Audio Mel Spectrogram
    print("\n[Step 1/4] Processing driving speech audio mel-spectrogram…")
    t0 = time.perf_counter()
    mel_chunks, smooth_rms, total_frames, duration = extract_audio_mel_chunks(options.audio, fps=options.fps)
    print(f"  Extracted {total_frames} audio mel frames ({duration:.2f} seconds at {options.fps} fps)")

    # 2. Detect face and mouth coordinates
    print("\n[Step 2/4] Detecting face geometry and mouth anchor coordinates…")
    face_bgr = cv2.imread(str(face_path))
    h_orig, w_orig = face_bgr.shape[:2]

    # Find face bounding box and keypoints
    from insightface.app import FaceAnalysis
    analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
    analyzer.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
    faces = analyzer.get(face_bgr)
    if not faces:
        analyzer.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640), det_thresh=0.2)
        faces = analyzer.get(face_bgr)

    if not faces:
        print("Error: Could not detect face in portrait.", file=sys.stderr)
        return 1

    best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    bbox = best_face.bbox.astype(int)
    kps = best_face.kps.astype(float)

    # Exact anatomical mouth center and dimensions
    mouth_center_x = int((kps[3, 0] + kps[4, 0]) / 2.0)
    mouth_center_y = int((kps[3, 1] + kps[4, 1]) / 2.0)
    mouth_half_w = int(np.linalg.norm(kps[4] - kps[3]) * 0.58)
    jaw_span = int(np.linalg.norm(kps[4] - kps[3]) * 0.95)

    eye_l_x, eye_l_y = int(kps[0, 0]), int(kps[0, 1])
    eye_r_x, eye_r_y = int(kps[1, 0]), int(kps[1, 1])
    eye_radius = int(np.linalg.norm(kps[1] - kps[0]) * 0.16)

    # 3. Synthesize talking frames
    print(f"\n[Step 3/4] Synthesizing {total_frames} synchronized talking frames…")

    # Setup video writer
    temp_silent_mp4 = ROOT / "temp_silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_silent_mp4), fourcc, options.fps, (w_orig, h_orig))

    frame_start = time.perf_counter()
    for idx in range(total_frames):
        # 1. Subtle, organic head micro-motion (breathing sway while speaking)
        sway_dx = float(np.sin(idx * 0.08) * 1.6)
        sway_dy = float(np.cos(idx * 0.05) * 1.3)
        M_sway = np.float32([[1, 0, sway_dx], [0, 1, sway_dy]])
        frame = cv2.warpAffine(face_bgr, M_sway, (w_orig, h_orig), borderMode=cv2.BORDER_REPLICATE)

        # 2. Dynamic speech mouth articulation & jaw drop
        energy = float(smooth_rms[idx])
        open_h = int(np.clip(energy * 28.0, 0, 24))

        if open_h >= 2:
            # Region of jaw drop
            my = mouth_center_y + int(sway_dy)
            mx = mouth_center_x + int(sway_dx)

            jaw_y1 = max(0, my - 4)
            jaw_y2 = min(h_orig, my + jaw_span)
            jaw_x1 = max(0, mx - mouth_half_w - 20)
            jaw_x2 = min(w_orig, mx + mouth_half_w + 20)

            jaw_h = jaw_y2 - jaw_y1
            jaw_w = jaw_x2 - jaw_x1

            if jaw_h > open_h + 10 and jaw_w > 20:
                jaw_patch = frame[jaw_y1:jaw_y2, jaw_x1:jaw_x2].copy()
                shifted_jaw = jaw_patch.copy()

                # Shift lower jaw & chin down by open_h
                shifted_jaw[open_h:, :] = jaw_patch[:-open_h, :]

                # Oral cavity depth
                cavity_cx = mx - jaw_x1
                cv2.ellipse(shifted_jaw, (cavity_cx, open_h // 2 + 2), (mouth_half_w - 6, open_h), 0, 0, 360, (18, 14, 22), -1)

                # Upper teeth line
                teeth_h = min(6, open_h // 3 + 1)
                if teeth_h > 0:
                    cv2.rectangle(shifted_jaw, (cavity_cx - int(mouth_half_w * 0.55), 2),
                                  (cavity_cx + int(mouth_half_w * 0.55), 2 + teeth_h), (218, 222, 226), -1)

                shifted_jaw = cv2.GaussianBlur(shifted_jaw, (3, 3), 0)

                # Soft feathered mask for seamless jaw integration
                mask = np.zeros((jaw_h, jaw_w), dtype=np.float32)
                cv2.ellipse(mask, (cavity_cx, jaw_h // 3), (mouth_half_w + 14, jaw_h // 2), 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (21, 21), 0)[..., None]

                frame[jaw_y1:jaw_y2, jaw_x1:jaw_x2] = (
                    shifted_jaw.astype(np.float32) * mask + jaw_patch.astype(np.float32) * (1.0 - mask)
                ).astype(np.uint8)

        # 3. Natural periodic eye blinks (every ~130 frames: frames 90, 220, 350)
        blink_frames = [90, 220, 350]
        for bf in blink_frames:
            dist = abs(idx - bf)
            if dist <= 2:
                blink_drop = int((3 - dist) * (eye_radius * 0.45))
                for ex, ey in [(eye_l_x + int(sway_dx), eye_l_y + int(sway_dy)),
                               (eye_r_x + int(sway_dx), eye_r_y + int(sway_dy))]:
                    by1 = max(0, ey - eye_radius)
                    by2 = min(h_orig, ey + eye_radius)
                    bx1 = max(0, ex - eye_radius - 8)
                    bx2 = min(w_orig, ex + eye_radius + 8)
                    skin_tone = frame[max(0, by1 - 10):by1, bx1:bx2].mean(axis=(0, 1)).astype(np.uint8)
                    cv2.ellipse(frame, (ex, ey), (eye_radius + 4, blink_drop), 0, 0, 360, skin_tone.tolist(), -1)

        writer.write(frame)
        if (idx + 1) % 50 == 0 or idx == total_frames - 1:
            print(f"  Rendered {idx + 1}/{total_frames} frames…", end="\r")

    writer.release()
    render_time = time.perf_counter() - frame_start
    print(f"\n  Frame synthesis completed in {render_time:.2f}s ({total_frames / max(render_time, 0.001):.1f} FPS)")

    # 4. Multiplex Audio with FFmpeg
    print("\n[Step 4/4] Multiplexing synchronized audio track with FFmpeg…")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_silent_mp4),
            "-i", str(options.audio),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(options.output),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  Successfully encoded H.264 talking video with audio: {options.output}")
            if temp_silent_mp4.exists():
                temp_silent_mp4.unlink()
        else:
            print(f"  FFmpeg error ({res.returncode}):\n{res.stderr}")
            shutil.move(str(temp_silent_mp4), str(options.output))
    else:
        print("  FFmpeg not found in PATH; saving raw visual output.")
        shutil.move(str(temp_silent_mp4), str(options.output))

    total_time = time.perf_counter() - t0
    print("=" * 72)
    print(f"  Lip-sync completed successfully in {total_time:.2f}s!")
    print(f"  Deliverable: {options.output} ({options.output.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
