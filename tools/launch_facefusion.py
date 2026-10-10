#!/usr/bin/env python3
"""tools/launch_facefusion.py — 1-Click FaceFusion WebUI Cloud GPU Runner.

Clones the official FaceFusion repository (https://github.com/facefusion/facefusion),
opens a high-speed Cloudflare HTTPS tunnel, and launches the FaceFusion video-to-video
swapping platform on the cloud GPU with an instant public link.
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


def main():
    print("=" * 72)
    print("  LAUNCHING OFFICIAL FACEFUSION PLATFORM ON CLOUD GPU")
    print(f"  Python: {sys.version.split()[0]} | Working dir: {os.getcwd()}")
    print("=" * 72, flush=True)

    # 1. Clean clone of official FaceFusion repository
    if not (FF_DIR / "facefusion.py").exists():
        FF_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning official FaceFusion repository into {FF_DIR}…", flush=True)
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/facefusion/facefusion.git", str(FF_DIR)])
    else:
        print(f"Resetting FaceFusion to clean state at {FF_DIR}…", flush=True)
        try:
            subprocess.run(["git", "-C", str(FF_DIR), "checkout", "--", "."], check=False)
            subprocess.run(["git", "-C", str(FF_DIR), "clean", "-fd"], check=False)
        except Exception:
            pass

    # 2. Install lightweight dependencies
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
                missing.append("onnxruntime")
            else:
                missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {' '.join(missing)}…", flush=True)
        for pkg in missing:
            print(f"  Installing {pkg}…", flush=True)
            if pkg in ("gradio", "gradio-rangeslider"):
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", pkg], check=True)
            else:
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg], check=True)
        print("✅ Dependencies successfully installed!", flush=True)
    else:
        print("All dependencies already verified.", flush=True)

    # 3. Direct UI entry patch into FaceFusion core
    core_file = FF_DIR / "facefusion" / "core.py"
    if core_file.exists():
        try:
            c = core_file.read_text()
            direct_cli = (
                "def cli() -> None:\n"
                "\tsignal.signal(signal.SIGINT, signal_exit)\n"
                "\tprogram = create_program()\n"
                "\targs = vars(program.parse_known_args()[0])\n"
                "\tapply_args(args, state_manager.init_item)\n"
                "\tstate_manager.init_item('command', 'run')\n"
                "\tstate_manager.init_item('ui_layouts', ['default'])\n"
                "\tstate_manager.init_item('open_browser', False)\n"
                "\tlogger.init('info')\n"
                "\timport facefusion.uis.core as ui\n"
                "\tui.init()\n"
                "\tui.launch()\n\n"
            )
            c = re.sub(r"def cli\(\) -> None:[\s\S]*?(?=def route)", direct_cli, c)
            core_file.write_text(c)
            print("Direct UI entry patched into core.", flush=True)
        except Exception as exc:
            print(f"Note on core patch: {exc}", flush=True)

    layout_file = FF_DIR / "facefusion" / "uis" / "layouts" / "default.py"
    if layout_file.exists():
        try:
            l_text = layout_file.read_text()
            old_launch = "ui.launch(favicon_path = 'facefusion.ico', inbrowser = state_manager.get_item('open_browser'))"
            new_launch = "print('\\n=== FaceFusion Gradio server is now listening on port 7860 ===\\n', flush=True)\n\tui.launch(server_name = '0.0.0.0', server_port = 7860, inbrowser = False)"
            if old_launch in l_text:
                l_text = l_text.replace(old_launch, new_launch)
                layout_file.write_text(l_text)
                print("Server binding patched into default layout.", flush=True)
        except Exception as exc:
            print(f"Note on layout patch: {exc}", flush=True)

    # 4. Open Cloudflare HTTPS tunnel for port 7860
    cf_bin = download_cloudflared()
    log_file = Path("/tmp/cloudflared_ff.log")
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass

    print("\nOpening secure Cloudflare HTTPS tunnel on port 7860…", flush=True)
    tunnel_cmd = [
        str(cf_bin), "tunnel",
        "--no-autoupdate",
        "--url", "http://127.0.0.1:7860",
        "--logfile", str(log_file),
    ]
    tunnel_proc = subprocess.Popen(tunnel_cmd)

    public_url = None
    start_time = time.time()
    while time.time() - start_time < 30:
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

    if public_url:
        print("\n" + "#" * 72, flush=True)
        print("  🎉 CLOUDFLARE TUNNEL OPENED!", flush=True)
        print("#" * 72)
        print(f"\n  👉 LINK: {public_url}\n", flush=True)
        print("  Waiting for FaceFusion engine to finish binding to port 7860...")
        print("#" * 72 + "\n", flush=True)

    # 5. Launch FaceFusion WebUI with live line-by-line output
    try:
        os.chdir(str(FF_DIR))
        proc = subprocess.Popen(
            [sys.executable, "-u", "facefusion.py", "run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.wait()
        print(f"\nFaceFusion process ended with code {proc.returncode}.", flush=True)
    except KeyboardInterrupt:
        print("\nStopping FaceFusion…")
    finally:
        tunnel_proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
