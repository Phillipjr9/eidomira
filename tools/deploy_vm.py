"""Turn a fresh GPU machine into a running Eidomira studio, in one command.

The steps are the ones in `docs/elastic-compute-plan.md`, in their order, with the checks that
catch the expensive mistakes before the invoice:

    1. preflight   — Docker, a GPU, NVIDIA runtime, the licensed models, a free port
    2. configure   — a real STUDIO_AUTH_SECRET, the public URL, origins, model paths
    3. TLS         — a Caddyfile for the domain, because camera access needs https anywhere
                     but localhost, and the whole point is a live test from a phone
    4. start       — docker compose up -d --build
    5. verify      — poll /api/health until the engine is the neural one and accelerated=true,
                     and *fail* if the host fell back to CPU rather than letting it look fine
    6. prove       — the UDP check and the real-device test, printed with the exact commands

Design rules, because this runs on somebody's paid machine:

* **It never installs system packages.** Docker, the NVIDIA driver and the container toolkit
  are the host's business; when one is missing it says precisely what to run and stops. A
  deploy script that edits a machine it does not understand is worse than no script.
* **It never overwrites a secret.** Re-running keeps the existing `STUDIO_AUTH_SECRET`, because
  replacing it signs everybody out and looks like a bug.
* **`--print` does nothing.** The plan is printed first, always, so the dry run and the real
  run share one code path.
* **A CPU fallback is a failure, not a warning.** `accelerated: false` after starting means the
  GPU is not wired to the container, which is the single most common deployment mistake and it
  is silent — the studio works, just many times slower.
"""
from __future__ import annotations

import argparse
import json
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.config import DEVELOPMENT_AUTH_SECRET

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"
COMPOSE_TLS = ROOT / "docker-compose.tls.yml"
ENV_FILE = ROOT / ".env"
CADDYFILE = ROOT / "Caddyfile"

#: Long enough that the HMAC is comfortably over the 32-byte minimum PyJWT warns about, which
#: the running server does complain about when the shipped default is in use.
SECRET_BYTES = 48

DEFAULT_PORT = 8000
HEALTH_PATH = "/api/health"


@dataclass
class Check:
    """One preflight result: what was asked, what was found, and what to do about it."""

    name: str
    ok: bool
    detail: str = ""
    remedy: str = ""
    fatal: bool = True

    def line(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.fatal else "warn")
        text = f"  [{mark}] {self.name}"
        if self.detail:
            text += f" — {self.detail}"
        if not self.ok and self.remedy:
            text += f"\n         fix: {self.remedy}"
        return text


@dataclass
class Result:
    code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0


Runner = Callable[..., Result]


def first_line(text: str, fallback: str = "") -> str:
    """The opening line of a command's output, or a fallback.

    `docker compose version` answering with nothing on stdout is not a reason to raise an
    IndexError out of a deployment tool: the exit code already said whether it worked.
    """
    lines = (text or "").splitlines()
    return lines[0].strip() if lines else fallback


def shell(args: list[str], timeout: float = 120.0) -> Result:
    """Run a command and capture everything, never raising on a non-zero exit.

    A deploy script that dies on the first non-zero exit is a deploy script nobody can
    diagnose: `docker compose version` on a host without compose should produce a message
    about compose, not a traceback.
    """
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return Result(127, "", f"{args[0]}: not found")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"{args[0]}: timed out after {timeout:.0f}s")
    return Result(completed.returncode, completed.stdout.strip(), completed.stderr.strip())


# ── configuration ───────────────────────────────────────────────────────────────

def generate_secret() -> str:
    return secrets.token_urlsafe(SECRET_BYTES)


def read_env(path: Path) -> dict[str, str]:
    """Parse the subset of dotenv this project writes: `KEY=value`, `#` comments, blanks."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def render_env(values: dict[str, str], origin: str) -> str:
    """The .env compose reads. Generated, not hand-written, and safe to re-read."""
    lines = [
        "# Written by `python -m tools.deploy_vm`. Re-running keeps the secret and updates",
        "# everything else; edit by hand if you prefer, and the tool will respect it.",
        "",
        f"# The public origin. https here is what makes the camera work away from localhost.",
        f"STUDIO_PUBLIC_URL={origin}",
        "",
        "# Sessions and verification tokens are signed with this. It is generated once and",
        "# never rewritten: replacing it signs out every existing session.",
        f"STUDIO_AUTH_SECRET={values['STUDIO_AUTH_SECRET']}",
        "",
        "# Browsers at these origins may call the API.",
        f"STUDIO_ALLOWED_ORIGINS={values['STUDIO_ALLOWED_ORIGINS']}",
        "",
        "# The neural engine and where the licensed artefacts are, inside the container.",
        "STUDIO_BACKEND=inswapper",
        f"STUDIO_MODEL_PATH={values['STUDIO_MODEL_PATH']}",
        f"STUDIO_PARSER_MODEL_PATH={values['STUDIO_PARSER_MODEL_PATH']}",
        f"STUDIO_RESTORATION_MODEL_PATH={values['STUDIO_RESTORATION_MODEL_PATH']}",
        "",
        "# Set these once a TURN server exists. Without one, a viewer behind a symmetric NAT",
        "# cannot connect — the signaling completes and no media arrives.",
        f"STUDIO_TURN_URLS={values.get('STUDIO_TURN_URLS', '')}",
        f"STUDIO_TURN_SECRET={values.get('STUDIO_TURN_SECRET', '')}",
        "",
        "# Behind a proxy, set this to the proxy's network or every caller shares one",
        "# rate-limit bucket. Caddy on the same host arrives on loopback, which is already",
        "# trusted, so the default is correct for the layout this tool sets up.",
        f"# FORWARDED_ALLOW_IPS={values.get('FORWARDED_ALLOW_IPS', '172.16.0.0/12')}",
        "",
    ]
    return "\n".join(lines)


def planned_values(origin: str, existing: dict[str, str],
                   models: dict[str, str] | None = None) -> dict[str, str]:
    """The values to write, keeping a secret that is already there and not the shipped one."""
    values = dict(existing)
    secret = existing.get("STUDIO_AUTH_SECRET", "")
    if not secret or secret == DEVELOPMENT_AUTH_SECRET:
        values["STUDIO_AUTH_SECRET"] = generate_secret()
    values["STUDIO_ALLOWED_ORIGINS"] = ",".join(
        part for part in {origin,
                          *(existing.get("STUDIO_ALLOWED_ORIGINS", "").split(","))}
        if part.strip())
    paths = models or {}
    values["STUDIO_MODEL_PATH"] = paths.get("swap", values.get("STUDIO_MODEL_PATH")
                                            or "/models/inswapper_128.onnx")
    values["STUDIO_PARSER_MODEL_PATH"] = paths.get("parser",
                                                   values.get("STUDIO_PARSER_MODEL_PATH")
                                                   or "/models/face_parser.onnx")
    values["STUDIO_RESTORATION_MODEL_PATH"] = paths.get(
        "restorer", values.get("STUDIO_RESTORATION_MODEL_PATH") or "/models/gfpgan_1.4.onnx")
    return values


def write_env(path: Path, values: dict[str, str], origin: str) -> str:
    text = render_env(values, origin)
    path.write_text(text, encoding="utf-8")
    return text


def caddyfile(domain: str, email: str = "", upstream: str = f"127.0.0.1:{DEFAULT_PORT}") -> str:
    """A Caddyfile for `domain`, proxying to the studio.

    Caddy terminates TLS with a real certificate and proxies the HTTP and WebSocket signaling.
    The **media does not go through it**: WebRTC media is UDP straight between the browser and
    the studio, which is why the studio is on host networking and why `tools/udp_probe.py`
    exists. A reverse proxy that looks healthy can still be in front of a host that cannot
    carry a single frame.
    """
    global_block = f"{{\n\temail {email}\n}}\n\n" if email else ""
    return f"""{global_block}{domain} {{
	# The studio answers on the host's own loopback because of host networking; see
	# docker-compose.yml for why that is not a preference.
	reverse_proxy {upstream}
}}
"""


# ── preflight ───────────────────────────────────────────────────────────────────

def check_docker(runner: Runner = shell) -> Check:
    version = runner(["docker", "--version"])
    if not version.ok:
        return Check("docker", False, version.stderr or version.stdout,
                     "install Docker Engine: https://docs.docker.com/engine/install/")
    compose = runner(["docker", "compose", "version"])
    if not compose.ok:
        return Check("docker compose (v2)", False, compose.stderr or compose.stdout,
                     "install the compose plugin: https://docs.docker.com/compose/install/linux/")
    return Check("docker compose (v2)", True,
                 first_line(compose.stdout, "compose is available"))


def check_gpu(runner: Runner = shell) -> Check:
    smi = runner(["nvidia-smi", "--query-gpu=name,memory.total",
                  "--format=csv,noheader"])
    if not smi.ok:
        return Check("nvidia-smi", False, smi.stderr or smi.stdout,
                     "this host has no visible GPU or no driver; the neural engine will fall "
                     "back to CPU and be far too slow to be useful")
    return Check("nvidia-smi", True, first_line(smi.stdout, "a GPU is present"))


def check_nvidia_runtime(runner: Runner = shell) -> Check:
    """Whether the *container* can see the GPU, which is a different question from whether the
    host has one. This is the check whose absence produces `accelerated: false` after a
    successful-looking deploy."""
    info = runner(["docker", "info", "--format", "{{json .Runtimes}}"])
    if not info.ok:
        return Check("docker sees the GPU", False, info.stderr or info.stdout,
                     "start the docker daemon and check `docker info` by hand")
    if "nvidia" not in info.stdout:
        # Fatal rather than a warning, and deliberately: the container cannot see the GPU, so
        # the health check below is certain to fail — a minute of building later. Stopping here
        # is the same answer sooner, which is the whole point of a preflight.
        return Check("docker sees the GPU", False, "no nvidia runtime registered",
                     "install the NVIDIA Container Toolkit and restart docker: "
                     "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")
    return Check("docker sees the GPU", True, "nvidia runtime registered")


def check_models(directory: Path | None = None,
                 runner: Runner = shell) -> Check:
    """The licensed artefacts, checked with the tool that knows their licences."""
    result = runner([sys.executable, "-m", "tools.fetch_models", "--check"],
                    timeout=60.0)
    if result.code != 0:
        return Check("licensed models", False, "tools.fetch_models --check failed",
                     "run `python -m tools.fetch_models --check` and read the report")
    present = [line for line in result.stdout.splitlines() if "present" in line
               and "missing" not in line]
    missing = [line for line in result.stdout.splitlines() if "missing" in line]
    if missing and not present:
        return Check("licensed models", False, f"{len(missing)} of 3 missing",
                     "fetch them on a host with internet: "
                     "`python -m tools.fetch_models --fetch parser` and `--fetch restorer`; "
                     "the swap weights need a licence first (docs/insightface-licence-request.md)")
    if missing:
        return Check("licensed models", True,
                     f"{len(present)} present, {len(missing)} missing — the stack runs without "
                     f"the optional ones, with a softer swap", fatal=False)
    return Check("licensed models", True, "all present")


def check_port(port: int = DEFAULT_PORT) -> Check:
    """A live studio already on the port is not an error — it may be the previous run."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        in_use = probe.connect_ex(("127.0.0.1", port)) == 0
    finally:
        probe.close()
    if in_use:
        return Check(f"port {port}", False, "already in use",
                     "if it is a previous studio, `docker compose down` first; if it is "
                     "something else, set a different port in docker-compose.yml", fatal=False)
    return Check(f"port {port}", True, "free")


def preflight(port: int = DEFAULT_PORT, runner: Runner = shell) -> list[Check]:
    return [check_docker(runner), check_gpu(runner), check_nvidia_runtime(runner),
            check_models(runner=runner), check_port(port)]


def blocking(checks: list[Check]) -> list[Check]:
    return [check for check in checks if not check.ok and check.fatal]


# ── health ──────────────────────────────────────────────────────────────────────

def health_verdict(payload: dict) -> tuple[bool, str]:
    """Read `/api/health` the way the deployment cares about it.

    Three things must be true, and each one has been false on a host that looked fine:

    * the backend is the neural engine, not the diagnostic one (a wrong `STUDIO_BACKEND` value
      does not raise — `create_engine` falls through and runs the diagnostic engine);
    * ONNX Runtime resolved an accelerated provider, not the CPU fallback;
    * `accelerated` agrees, because that is what the studio displays.
    """
    backend = payload.get("backend")
    if backend != "inswapper":
        return False, (f"backend is {backend!r}, not 'inswapper' — the neural engine is not "
                       f"selected, so the studio runs the diagnostic path")
    if not payload.get("accelerated"):
        provider = payload.get("provider") or "none"
        return False, (f"the engine loaded but ONNX Runtime resolved {provider!r} — the "
                       f"container cannot see the GPU. Check the NVIDIA Container Toolkit "
                       f"and the compose `deploy.resources` block")
    return True, (f"engine {backend}, provider {payload.get('provider')}, "
                  f"peers {payload.get('active_peers')}/{payload.get('peer_capacity')}")


def wait_for_health(url: str, timeout: float = 240.0, interval: float = 2.0,
                    opener=urllib.request.urlopen) -> tuple[bool, str]:
    """Poll until the studio answers, then judge the answer.

    A container that starts and reports a CPU engine is a *failure* here, not a warning: it is
    the mistake that costs a whole deployment and it hides behind a working-looking page.
    """
    deadline = time.monotonic() + timeout
    last = "no answer yet"
    while time.monotonic() < deadline:
        try:
            with opener(url, timeout=5) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError,
                TimeoutError) as error:
            last = f"{type(error).__name__}: {error}"
            time.sleep(interval)
            continue
        return health_verdict(payload)
    return False, f"the studio did not answer within {timeout:.0f}s ({last})"


# ── compose ─────────────────────────────────────────────────────────────────────

def compose_command(tls: bool = False, *extra: str) -> list[str]:
    command = ["docker", "compose"]
    if tls:
        command += ["-f", COMPOSE.name, "-f", COMPOSE_TLS.name]
    command += ["up", "-d", "--build", *extra]
    return command


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy the studio to a GPU host, and verify it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --print first: it takes no action and shows every step.")
    parser.add_argument("--domain", help="public hostname, e.g. swap.example.com")
    parser.add_argument("--email", default="", help="email for the TLS certificate (optional)")
    parser.add_argument("--origin", help="public origin (default: https://DOMAIN, else http://127.0.0.1:8000)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tls", action="store_true",
                        help="also start Caddy for TLS (needs --domain and ports 80/443 free)")
    parser.add_argument("--print", dest="dry_run", action="store_true",
                        help="show the plan and change nothing")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--udp-wait", type=float, default=0.0,
                        help="seconds to listen for the UDP check from another machine")
    parser.add_argument("--health-timeout", type=float, default=240.0)
    parser.add_argument("--directory", type=Path, default=ROOT)
    return parser.parse_args(argv)


def report_ok(namespace: argparse.Namespace, demo_login: bool = False) -> int:
    """The closing summary: what to do now, with the commands in the order to run them."""
    domain = namespace.domain or "127.0.0.1"
    scheme = "https" if namespace.tls else "http"
    origin = namespace.origin or f"{scheme}://{domain}"
    print("\n  The studio is up. To prove it end to end:")
    print(f"\n    open {origin}/app and sign in"
          + (" (or press Demo access)" if demo_login else ""))
    print("\n  1. A live session from a phone on mobile data — not from this machine, and not")
    print("     on this host's network. Loopback hides every NAT and firewall problem there is.")
    print("\n  2. Watch the log while it runs:")
    print("       docker compose logs -f studio")
    print("\n  3. If no frames arrive though signaling worked, it is the transport, not the app:")
    print("       python -m tools.udp_probe serve --port 34789 --seconds 60   # here")
    print("       python -m tools.udp_probe send --host %s --port 34789      # elsewhere" % domain)
    print("\n  GPU is confirmed accelerated, so any slowness after this is bandwidth or NAT.")
    return 0


def main(argv: list[str] | None = None, runner: Runner = shell,
         opener=urllib.request.urlopen, env_path: Path | None = None) -> int:
    namespace = parse_args(argv)
    directory: Path = namespace.directory
    env_path = env_path or (directory / ENV_FILE.name)
    origin = namespace.origin or (f"https://{namespace.domain}" if namespace.tls or namespace.domain
                                 else f"http://127.0.0.1:{namespace.port}")

    existing = read_env(env_path)
    values = planned_values(origin, existing)
    keep_secret = bool(existing.get("STUDIO_AUTH_SECRET")) and \
        existing.get("STUDIO_AUTH_SECRET") != DEVELOPMENT_AUTH_SECRET

    print(f"  Eidomira deploy — {'PLAN ONLY (nothing will change)' if namespace.dry_run else 'running'}")
    print(f"\n  origin         {origin}")
    print(f"  auth secret    {'kept from .env' if keep_secret else 'will be generated'}")
    print(f"  models         {values['STUDIO_MODEL_PATH']}")
    print(f"  TLS            {'Caddy on ports 80/443' if namespace.tls else 'none — the camera will not work away from localhost'}")
    print(f"\n  start with     {' '.join(compose_command(namespace.tls))}")

    if namespace.dry_run:
        print("\n  Then it would poll", f"http://127.0.0.1:{namespace.port}{HEALTH_PATH}",
              "until the engine is `inswapper` and accelerated.")
        return 0

    if not namespace.skip_preflight:
        print("\n  preflight")
        checks = preflight(namespace.port, runner=runner)
        for check in checks:
            print(check.line())
        blockers = blocking(checks)
        if blockers:
            # Flush first: the checks above went to stdout and this goes to stderr, so a
            # terminal that separates them would otherwise print the verdict before the
            # evidence that produced it.
            sys.stdout.flush()
            print(f"\n  {len(blockers)} blocking problem(s) — nothing was started or written.",
                  file=sys.stderr)
            return 2

    write_env(env_path, values, origin)
    print(f"\n  wrote {env_path.name}")

    if namespace.tls:
        if not namespace.domain:
            print("  --tls needs --domain: Caddy cannot certify an address", file=sys.stderr)
            return 2
        CADDYFILE.write_text(caddyfile(namespace.domain, namespace.email), encoding="utf-8")
        print(f"  wrote {CADDYFILE.name} for {namespace.domain}")

    command = compose_command(namespace.tls)
    print(f"\n  $ {' '.join(command)}")
    started = runner(command, timeout=1800.0)
    if not started.ok:
        print(started.stdout)
        print(started.stderr or "compose failed", file=sys.stderr)
        return 3

    url = f"http://127.0.0.1:{namespace.port}{HEALTH_PATH}"
    print(f"\n  waiting for {url}")
    healthy, detail = wait_for_health(url, timeout=namespace.health_timeout, opener=opener)
    print(("  [ok  ] " if healthy else "  [FAIL] ") + detail)
    if not healthy:
        print("\n  the studio is running but not usable as configured. Look at:",
              file=sys.stderr)
        print("    docker compose logs studio | tail -40", file=sys.stderr)
        return 4

    if namespace.udp_wait > 0:
        print(f"\n  UDP check: listening for {namespace.udp_wait:.0f}s. From another machine:")
        print(f"    python -m tools.udp_probe send --host {namespace.domain or '<this host>'} "
              f"--port 34789")
        from tools import udp_probe
        echoed = udp_probe.serve(34789, seconds=namespace.udp_wait)
        if echoed:
            print(f"  [ok  ] a datagram reached this host and came back ({echoed})")
        else:
            print("  [warn] nothing arrived. Either the check was not run, or inbound UDP is "
                  "blocked — in which case the signaling will succeed and no media will.")

    return report_ok(namespace)


if __name__ == "__main__":
    raise SystemExit(main())
