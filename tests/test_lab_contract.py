"""The on-device lab: one page, two languages, and nothing to catch a disagreement.

The lab runs the boost in JavaScript, which this suite cannot execute in a browser. So the
things that *can* be checked are checked here — that every id the script reaches for exists,
that the runtime version it loads is the one it pins, that the CDN it loads is one the
Content-Security-Policy actually permits — and the algorithm itself is held to the Python
implementation by `tests/test_boost_parity.py`.

The CSP check matters more than it looks: a page that loads a 28 MB wasm runtime from a host
the policy does not allow fails silently in the browser console, which surfaces as "your
feature does not work on my phone" weeks later.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
LAB_HTML_PATH = STATIC / "lab" / "boost-lab.html"
LAB_JS_PATH = STATIC / "lab" / "boost-lab.js"
BOOST_JS_PATH = STATIC / "boost.js"
MODEL_PATH = STATIC / "lab" / "swapper-standin.onnx"

LAB_HTML = LAB_HTML_PATH.read_text(encoding="utf-8")
LAB_JS = LAB_JS_PATH.read_text(encoding="utf-8")
BOOST_JS = BOOST_JS_PATH.read_text(encoding="utf-8")


def element_ids() -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', LAB_HTML))


def ids_created_by_the_script() -> set[str]:
    """Rows `reportDevice()` writes into the list. They are not in the markup on purpose."""
    return set(re.findall(r'\["(d[A-Za-z]+)",\s*"', LAB_JS))


def ids_the_script_reaches_for() -> set[str]:
    return set(re.findall(r'\$\("([^"]+)"\)', LAB_JS)) | set(
        re.findall(r'readout\("([^"]+)"', LAB_JS)
    )


def test_every_id_the_lab_script_uses_exists():
    """The failure mode this catches: a renamed id throws nothing until a visitor presses the
    button, and the visitor is on a phone in another country."""
    missing = sorted(ids_the_script_reaches_for() - element_ids() - ids_created_by_the_script())
    assert missing == [], f"the script reaches for ids that are not there: {missing}"


def test_the_page_loads_its_own_module_and_the_shared_boost():
    assert 'src="/static/lab/boost-lab.js"' in LAB_HTML
    assert 'from "../boost.js"' in LAB_JS, "the page must use the tested boost, not a copy"
    assert 'type="module"' in LAB_HTML, "an untyped script tag would fail to import anything"
    assert "/static/tokens.css" in LAB_HTML, "the lab should not be a different-looking site"


def test_the_boost_module_exports_what_the_page_imports():
    """A rename in one file and not the other is a blank page, not an error message."""
    imported = re.search(r'import\s*\{([^}]+)\}\s*from\s*"\.\./boost\.js"', LAB_JS)
    assert imported, "the lab no longer imports the boost module"
    for name in [part.strip() for part in imported.group(1).split(",") if part.strip()]:
        assert re.search(rf"export\s+(function|const)\s+{name}\b", BOOST_JS), (
            f"boost-lab.js imports {name}, which static/boost.js does not export"
        )


def test_the_runtime_version_pinned_in_the_code_is_the_one_in_the_url():
    """A version in the label and a different one in the URL is worse than no label.

    Resolved rather than grepped: the URL is built from the constant, so the check is what
    the constant actually produces. The shape is pinned exactly, because the failure this
    guards is a typo in a path — and a wrong path here is a 404 that looks like "WebGPU does
    not work on my phone".
    """
    version = re.search(r'const ORT_VERSION = "([\d.]+)"', LAB_JS).group(1)
    base = re.search(r'const ORT_BASE = `([^`]+)`', LAB_JS).group(1).replace("${ORT_VERSION}", version)
    module = (re.search(r'const ORT_MODULE = `([^`]+)`', LAB_JS).group(1)
              .replace("${ORT_VERSION}", version).replace("${ORT_BASE}", base))

    assert re.fullmatch(
        r"https://[\w.-]+/npm/onnxruntime-web@\d+\.\d+\.\d+/dist/ort\.webgpu\.bundle\.min\.mjs",
        module,
    ), f"the runtime URL is not the shape it should be: {module}"
    assert module.startswith(base)
    # The wasm binary is fetched relative to this, so the base has to be the dist directory.
    assert base.endswith("/dist/"), f"ORT_BASE must point at the dist directory: {base}"
    # Shown on the page, not in the markup: the row is rendered at run time, so the check is
    # that the reader is told which runtime version produced the numbers.
    assert re.search(
        r'\["dRuntime",\s*"Runtime",\s*`onnxruntime-web \$\{ORT_VERSION\}`', LAB_JS
    ), "the page must show which runtime version was measured"


def test_the_policy_permits_the_host_the_lab_loads_from():
    """Checked through the real middleware rather than by reading the string."""
    with TestClient(app) as client:
        response = client.get("/lab")

    assert response.status_code == 200
    policy = response.headers["content-security-policy"]
    module_url = re.search(r'https://([\w.-]+)/npm/onnxruntime-web', LAB_JS)
    assert module_url, "the lab no longer loads the runtime from a CDN"
    host = module_url.group(1)
    assert f"script-src 'self' https://{host}" in policy, (
        f"script-src does not allow https://{host}, so the runtime cannot load: {policy}"
    )
    assert f"connect-src 'self' https://{host}" in policy, (
        "the wasm binary is fetched at run time, so connect-src has to allow it too"
    )
    # wasm cannot be compiled without this, and the failure is a console error, not a visible one.
    assert "'wasm-unsafe-eval'" in policy


def test_the_lab_is_served_but_not_framed():
    """No session, no identity, nothing to protect — but it is still not a page to embed in
    someone else's site, because its whole point is what the visitor's own device does."""
    with TestClient(app) as client:
        response = client.get("/lab")

    assert response.status_code == 200
    assert response.headers["x-frame-options"] == "SAMEORIGIN"
    assert "'self'" in response.headers["content-security-policy"]


def test_the_stand_in_model_is_present_and_is_the_one_the_page_expects():
    """The page cannot run without it, and a silent 404 here looks like "WebGPU is broken"."""
    assert MODEL_PATH.is_file(), "the stand-in model is missing"
    payload = MODEL_PATH.read_bytes()
    assert len(payload) < 64 * 1024, (
        "the stand-in is meant to be tiny; anything large would be a real model, and shipping "
        "one of those is a licensing decision, not a prototype one"
    )
    for name in (b"swapper-standin", b"crop", b"face"):
        assert name in payload, f"the model does not mention {name.decode()}"
    assert str(MODEL_PATH.relative_to(ROOT)) in LAB_JS


@pytest.mark.parametrize("claim", [
    "No swap weights here",
    "stand-in",
    "554",
])
def test_the_page_states_what_it_is_not(claim):
    """This project's standard is that a page which cannot do the thing says so, on the page.

    The 554 MB figure is the real size of `inswapper_128.onnx`: without it, a reader could
    reasonably assume the timings describe a face swap.
    """
    assert claim in LAB_HTML
