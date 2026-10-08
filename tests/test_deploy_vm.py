"""The deployment tool: it must configure correctly, refuse correctly, and never half-install.

Two properties matter more than any individual feature, and both have a test here:

* **It refuses before it writes.** A tool that starts a deployment and then discovers there is
  no GPU leaves a machine in a state somebody has to reason about. Preflight runs first and
  nothing touches the disk until it passes.
* **It never accepts a forgeable signing key.** `STUDIO_AUTH_SECRET` signs every session; the
  shipped default is in a public repository, so anyone who has read it can mint a token for any
  account. The tool generates one and never rewrites it, because replacing it signs out every
  existing session and looks exactly like a bug.

The last test is the one that protects the whole layout: compose only passes the variables it
lists into a container, so a key that `.env` sets and `docker-compose.yml` does not mention is
a setting that silently does nothing. Both files are scaned as text — no YAML dependency — and
the pin is exact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import DEVELOPMENT_AUTH_SECRET
from tools import deploy_vm

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"
COMPOSE_TLS = ROOT / "docker-compose.tls.yml"


def runner_from(table: dict[tuple[str, ...], deploy_vm.Result]):
    """A fake shell: match on the arguments, so each test states only what it cares about."""
    def run(args, timeout=120.0):
        for key, result in table.items():
            if tuple(args[:len(key)]) == key:
                return result
        return deploy_vm.Result(1, "", f"unexpected command: {args}")
    return run


def healthy_runner(overrides: dict | None = None):
    """A ready host: docker, compose, a GPU the daemon can see, and the models in place.

    Matching is on the *text* of the command rather than its exact argv, because the model
    check runs through `sys.executable`, whose path is not something a test should know.
    """
    table = dict(READY_HOST)
    partial = (overrides or {}).pop("models", None)
    result = partial or deploy_vm.Result(
        0, "swap      present\nparser    present\nrestorer  present", "")
    overrides = overrides or {}

    def run(args, timeout=120.0):
        for needle, override in overrides.items():
            if needle in " ".join(args):
                return override
        if "fetch_models" in " ".join(args):
            return result
        return runner_from(table)(args, timeout)

    return run


HEALTHY = deploy_vm.Result(0, json.dumps(
    {"ok": True, "backend": "inswapper", "provider": "CUDAExecutionProvider",
     "accelerated": True, "active_peers": 0, "peer_capacity": 4}), "")

READY_HOST = {
    ("docker", "--version"): deploy_vm.Result(0, "Docker version 27.3.1, build ce12230", ""),
    ("docker", "compose", "version"): deploy_vm.Result(0, "Docker Compose version v2.29.7", ""),
    ("nvidia-smi", "--query-gpu=name,memory.total"): deploy_vm.Result(
        0, "NVIDIA A10G, 23028 MiB", ""),
    ("docker", "info", "--format", "{{json .Runtimes}}"): deploy_vm.Result(
        0, '{"runc":{"path":"runc"},"nvidia":{"path":"nvidia-container-runtime"}}', ""),
}


# ── the secret ──────────────────────────────────────────────────────────────────

def test_it_generates_a_real_secret():
    secret = deploy_vm.generate_secret()
    assert secret != DEVELOPMENT_AUTH_SECRET
    assert len(secret) >= 32, "too short for the HMAC this signs"
    assert deploy_vm.generate_secret() != secret, "it is not generating anything random"


def test_it_never_rewrites_an_existing_secret(tmp_path):
    """Replacing it signs out every session, which reads as a login bug rather than a deploy."""
    path = tmp_path / ".env"
    first = deploy_vm.planned_values("https://swap.example.com", {})
    deploy_vm.write_env(path, first, "https://swap.example.com")
    original = first["STUDIO_AUTH_SECRET"]

    # a second run, reading what the first wrote
    second = deploy_vm.planned_values("https://swap.example.com",
                                      deploy_vm.read_env(path))
    deploy_vm.write_env(path, second, "https://swap.example.com")

    assert deploy_vm.read_env(path)["STUDIO_AUTH_SECRET"] == original


def test_it_replaces_the_shipped_development_secret(tmp_path):
    """The one value that must not survive a deployment, even if somebody put it there."""
    values = deploy_vm.planned_values("https://swap.example.com",
                                      {"STUDIO_AUTH_SECRET": DEVELOPMENT_AUTH_SECRET})
    assert values["STUDIO_AUTH_SECRET"] != DEVELOPMENT_AUTH_SECRET
    assert len(values["STUDIO_AUTH_SECRET"]) >= 32


# ── configuration ───────────────────────────────────────────────────────────────

def test_the_public_origin_is_accepted_by_the_api():
    """A public origin missing from the allow-list is CORS failures that look like an outage."""
    values = deploy_vm.planned_values("https://swap.example.com",
                                      {"STUDIO_ALLOWED_ORIGINS": "http://127.0.0.1:8000"})
    allowed = values["STUDIO_ALLOWED_ORIGINS"].split(",")
    assert "https://swap.example.com" in allowed
    assert "http://127.0.0.1:8000" in allowed, "the existing entry was dropped"


def test_an_origin_is_not_added_twice():
    values = deploy_vm.planned_values("https://swap.example.com",
                                      {"STUDIO_ALLOWED_ORIGINS": "https://swap.example.com"})
    assert values["STUDIO_ALLOWED_ORIGINS"].count("https://swap.example.com") == 1


def test_the_env_file_is_re_readable(tmp_path):
    """`read_env` has to read what `write_env` writes, or idempotency is a coincidence."""
    path = tmp_path / ".env"
    values = deploy_vm.planned_values("https://swap.example.com", {})
    deploy_vm.write_env(path, values, "https://swap.example.com")

    parsed = deploy_vm.read_env(path)
    assert parsed["STUDIO_AUTH_SECRET"] == values["STUDIO_AUTH_SECRET"]
    assert parsed["STUDIO_PUBLIC_URL"] == "https://swap.example.com"
    assert parsed["STUDIO_BACKEND"] == "inswapper"


def test_the_caddyfile_certifies_the_domain_and_reaches_the_studio():
    text = deploy_vm.caddyfile("swap.example.com", "ops@example.com")
    assert "swap.example.com {" in text
    assert "reverse_proxy 127.0.0.1:8000" in text
    assert "ops@example.com" in text


def test_the_caddyfile_does_not_invent_an_email():
    text = deploy_vm.caddyfile("swap.example.com")
    assert "email" not in text.split("swap.example.com")[0], "an email block with no address"


# ── preflight ───────────────────────────────────────────────────────────────────

def test_a_host_without_docker_is_refused_with_a_fix():
    check = deploy_vm.check_docker(runner_from({}))
    assert not check.ok and check.fatal
    assert "install Docker" in check.remedy


def test_a_host_without_a_gpu_is_refused():
    """Not a warning: a CPU host cannot run this at a usable speed, and finding out after the
    deployment means the first live test is a mystery."""
    table = dict(READY_HOST)
    del table[("nvidia-smi", "--query-gpu=name,memory.total")]
    checks = deploy_vm.preflight(runner=runner_from(table))
    gpu = next(c for c in checks if c.name == "nvidia-smi")
    assert not gpu.ok and gpu.fatal
    assert "CPU" in gpu.remedy


def test_a_host_that_has_a_gpu_but_docker_cannot_see_it_is_flagged():
    """This is the mistake that produces `accelerated: false` after a deploy that looked
    successful, so it is called out by name and left as a warning for the health check to
    confirm."""
    checks = deploy_vm.preflight(runner=healthy_runner({
        "docker info": deploy_vm.Result(0, '{"runc":{"path":"runc"}}', "")}))
    runtime = next(c for c in checks if c.name == "docker sees the GPU")

    assert not runtime.ok
    assert "Container Toolkit" in runtime.remedy
    # Fatal on purpose: without the runtime the container cannot see the GPU, so the health
    # check is certain to fail — a minute of building later. Same answer, sooner.
    assert runtime.fatal
    assert runtime in deploy_vm.blocking(checks)


def test_a_ready_host_has_no_blockers():
    checks = deploy_vm.preflight(port=59999, runner=healthy_runner())
    assert not deploy_vm.blocking(checks), [c.line() for c in deploy_vm.blocking(checks)]


def test_missing_models_are_a_blocker_but_one_of_three_is_not():
    """None of the three is required for the process to start, so "missing" alone is not the
    question. The question is whether the neural engine has anything to load: with all three
    absent the studio would run the diagnostic path and call it a deployment."""
    all_missing = deploy_vm.Result(0, "swap      missing\nparser    missing\nrestorer  missing", "")
    checks = deploy_vm.preflight(port=59999, runner=healthy_runner({"models": all_missing}))
    models = next(c for c in checks if c.name == "licensed models")
    assert not models.ok and models.fatal
    assert "fetch_models" in models.remedy

    some_missing = deploy_vm.Result(0, "swap      present\nparser    missing\nrestorer  present", "")
    checks = deploy_vm.preflight(port=59999, runner=healthy_runner({"models": some_missing}))
    models = next(c for c in checks if c.name == "licensed models")
    assert models.ok, "a missing optional model stopped a deploy it should not stop"
    assert "softer swap" in models.detail


# ── the health verdict ──────────────────────────────────────────────────────────

def test_a_cpu_fallback_is_a_failure_not_a_warning():
    ok, detail = deploy_vm.health_verdict(
        {"backend": "inswapper", "provider": "CPUExecutionProvider", "accelerated": False})
    assert not ok
    assert "CPUExecutionProvider" in detail
    assert "GPU" in detail or "NVIDIA" in detail


def test_the_diagnostic_backend_is_a_failure():
    ok, detail = deploy_vm.health_verdict(
        {"backend": "diagnostic", "provider": None, "accelerated": False})
    assert not ok
    assert "diagnostic" in detail


def test_a_healthy_host_passes():
    ok, detail = deploy_vm.health_verdict(json.loads(HEALTHY.stdout))
    assert ok and "inswapper" in detail


def test_waiting_survives_a_studio_that_is_not_up_yet():
    """The container is up before the model is loaded; a first poll that fails is normal."""
    calls = {"n": 0}

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.payload

    def opener(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection refused")
        return Response(HEALTHY.stdout.encode())

    ok, detail = deploy_vm.wait_for_health("http://127.0.0.1:8000/api/health", timeout=10,
                                           interval=0.01, opener=opener)
    assert ok, detail
    assert calls["n"] == 3


def test_waiting_gives_up_with_the_last_error():
    def opener(url, timeout=None):
        raise OSError("connection refused")

    ok, detail = deploy_vm.wait_for_health("http://127.0.0.1:8000/api/health", timeout=0.2,
                                           interval=0.05, opener=opener)
    assert not ok
    assert "did not answer" in detail
    assert "connection refused" in detail, "the reason was thrown away"


# ── the plan, and the refusal to half-deploy ────────────────────────────────────

def test_print_changes_nothing(tmp_path, capsys):
    env_path = tmp_path / ".env"
    code = deploy_vm.main(["--print", "--domain", "swap.example.com", "--tls",
                           "--directory", str(tmp_path)], env_path=env_path)

    assert code == 0
    printed = capsys.readouterr().out
    assert "PLAN ONLY" in printed
    assert "docker compose -f docker-compose.yml -f docker-compose.tls.yml up" in printed
    assert not env_path.exists(), "the dry run wrote a file"


def test_a_failed_preflight_writes_nothing(tmp_path, capsys):
    env_path = tmp_path / ".env"
    code = deploy_vm.main(["--domain", "swap.example.com", "--directory", str(tmp_path)],
                          runner=runner_from({}), env_path=env_path)

    assert code == 2
    assert not env_path.exists(), "a deploy started on a host that cannot run it"
    printed = capsys.readouterr()
    assert "blocking problem" in printed.err


def test_a_healthy_deploy_writes_configuration_and_starts_compose(tmp_path, capsys):
    started: list[list[str]] = []
    ready = healthy_runner()

    def run(args, timeout=120.0):
        if "compose" in " ".join(args) and "up" in args:
            started.append(list(args))
            return deploy_vm.Result(0, "", "")
        return ready(args, timeout)

    env_path = tmp_path / ".env"
    code = deploy_vm.main(["--domain", "swap.example.com", "--directory", str(tmp_path),
                           "--health-timeout", "5"],
                          runner=run, opener=lambda url, timeout=None: _response(HEALTHY.stdout),
                          env_path=env_path)

    assert code == 0, capsys.readouterr()
    assert started, "compose was never run"
    written = deploy_vm.read_env(env_path)
    assert written["STUDIO_PUBLIC_URL"] == "https://swap.example.com"
    assert written["STUDIO_AUTH_SECRET"] not in ("", DEVELOPMENT_AUTH_SECRET)
    assert written["STUDIO_BACKEND"] == "inswapper"
    out = capsys.readouterr().out
    assert "mobile data" in out, "the real-device test is not in the instructions"


def _response(payload: str):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload.encode()

    return Response()


def test_a_cpu_engine_fails_the_deploy(tmp_path, capsys):
    """The studio would run and look fine, and be many times too slow. The deploy says so."""
    cpu = json.dumps({"backend": "inswapper", "provider": "CPUExecutionProvider",
                      "accelerated": False})
    ready = healthy_runner()

    def run(args, timeout=120.0):
        if "compose" in " ".join(args) and "up" in args:
            return deploy_vm.Result(0, "", "")
        return ready(args, timeout)

    code = deploy_vm.main(["--domain", "swap.example.com", "--directory", str(tmp_path),
                           "--health-timeout", "5"],
                          runner=run, opener=lambda url, timeout=None: _response(cpu),
                          env_path=tmp_path / ".env")

    assert code == 4
    assert "CPUExecutionProvider" in capsys.readouterr().out


# ── the files have to agree ─────────────────────────────────────────────────────

def test_every_setting_the_env_sets_is_passed_into_the_container():
    """Compose passes only the variables listed in the service, so a key that `.env` sets and
    `docker-compose.yml` does not mention is a setting that silently does nothing — including
    the one that decides whether the neural engine runs at all. Scanned as text: the pin is
    about the literal reference existing."""
    rendered = deploy_vm.render_env(deploy_vm.planned_values("https://swap.example.com", {}),
                                    "https://swap.example.com")
    compose = COMPOSE.read_text(encoding="utf-8")

    keys = [line.split("=", 1)[0] for line in rendered.splitlines()
            if line and not line.startswith("#") and "=" in line]
    assert keys, "the env renderer produced nothing"
    for key in keys:
        assert f"${{{key}" in compose, f"{key} is set in .env but never passed to the container"


def test_the_signing_key_is_required_not_defaulted():
    """Fail closed: the shipped default is in a public repository, so compose must refuse to
    start the GPU deployment without a real one."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "${STUDIO_AUTH_SECRET:?" in compose, "compose would start with a forgeable key"


def test_both_files_use_host_networking():
    """The live media is SRTP over UDP on an ephemeral port, so a `ports:` mapping cannot
    carry it. If either service loses host networking the media path breaks, and it breaks
    silently — signaling succeeds and no frames arrive."""
    for path in (COMPOSE, COMPOSE_TLS):
        text = path.read_text(encoding="utf-8")
        assert "network_mode: host" in text, f"{path.name} cannot carry WebRTC media"
        body = "\n".join(line for line in text.splitlines()
                         if not line.strip().startswith("#"))
        assert not re.search(r"^\s{4}ports:", body, re.M), \
            f"{path.name} maps ports, which cannot cover an ephemeral UDP socket"
