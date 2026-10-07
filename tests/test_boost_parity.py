"""The browser boost and the Python boost must be the same algorithm, not similar ones.

Two implementations of one idea drift. The one that drifts is the one nobody runs, which
here is the browser: it cannot be exercised from this test suite, and by the time a user
sees a scrambled face the cause is three changes back.

So the fixtures are generated *by the Python implementation* — the four pass buffers it
sampled, the canvas it interleaved them into, and the source rectangles it sampled — and
`tests/boost_parity.mjs` asserts the JavaScript reproduces both the rectangles and the
canvas. Byte for byte, not approximately.

Skipped, not failed, when node is absent: this is a parity check, not a runtime dependency.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.boost import ALIGN, boosted_face, phases, shifted_transform

REPO = Path(__file__).resolve().parent.parent
NODE_SCRIPT = Path(__file__).resolve().parent / "boost_parity.mjs"

#: The aligned region inside the source frame: 200 pixels across, so the 128 crop has to
#: shrink it and the phases have real detail to recover.
REGION = {"x": 100, "y": 60, "w": 200, "h": 200}
SCALE = 2


def detailed_source() -> np.ndarray:
    """Detail at the resolution limit, plus structure a misaligned merge cannot fake."""
    size = 512
    image = np.full((size, size, 3), 90, np.uint8)
    face = np.full((REGION["h"], REGION["w"], 3), 170, np.uint8)
    checkerboard = np.indices((REGION["h"], REGION["w"])).sum(0) % 2
    face[checkerboard == 1] = np.array([120, 150, 190], np.uint8)
    face[::7, :] = np.array([40, 60, 80], np.uint8)
    face[:, ::11] = np.array([200, 190, 60], np.uint8)
    image[REGION["y"]:REGION["y"] + REGION["h"], REGION["x"]:REGION["x"] + REGION["w"]] = face
    return image


def build_fixtures(directory: Path) -> dict:
    source = detailed_source()
    #: Crop pixels are this many source pixels, because the crop is `step` times smaller.
    step = REGION["w"] / ALIGN
    base = np.array(
        [[ALIGN / REGION["w"], 0, -REGION["x"] * ALIGN / REGION["w"]],
         [0, ALIGN / REGION["h"], -REGION["y"] * ALIGN / REGION["h"]]],
        np.float64,
    )

    def warp_crop(dx, dy):
        return cv2.warpAffine(source, shifted_transform(base, dx, dy), (ALIGN, ALIGN),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # The geometry the browser's `sourceRectForPhase` claims: phase p of `scale` samples
    # from `region + (p / scale) * step`. Asserted against the transform itself here, so the
    # fixture cannot encode a mistake both implementations agree on.
    #
    # A linear ramp, not the pattern above: `step` is 1.5625 source pixels, so half a crop
    # pixel is 0.78 of a source pixel and cubic interpolation is unavoidable — comparing the
    # pattern's pixels would only say that *some* interpolation happened. Cubic
    # interpolation reproduces a linear signal exactly, so a ramp measures the sampling
    # position rather than the filter.
    ramp = np.zeros((512, 512, 3), np.uint8)
    ramp[:, :, 0] = np.clip(np.arange(512) * .4, 0, 255).astype(np.uint8)
    for offset in (0.0, 0.5, 1.0):
        sampled = cv2.warpAffine(ramp, shifted_transform(base, offset, 0), (ALIGN, ALIGN),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        expected = .4 * (REGION["x"] + offset * step)
        assert abs(float(sampled[0, 0, 0]) - expected) < 1.5, (
            f"warp_crop({offset}) sampled {sampled[0, 0, 0]}, which is not "
            f"region.x + {offset} * step (about {expected:.2f})"
        )

    canvas = boosted_face(warp_crop, lambda crop: crop, SCALE)

    rects = []
    passes = []
    # The order comes from the Python implementation, so a disagreement about ordering fails
    # here rather than silently producing a canvas nobody can explain.
    for x_phase, y_phase in phases(SCALE):
        rects.append({
            "x": REGION["x"] + (x_phase / SCALE) * step,
            "y": REGION["y"] + (y_phase / SCALE) * step,
            "w": REGION["w"],
            "h": REGION["h"],
        })
        passes.append(warp_crop(x_phase / SCALE, y_phase / SCALE))

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rects.json").write_text(json.dumps(
        {"scale": SCALE, "region": REGION, "rects": rects}
    ))
    (directory / "passes.bin").write_bytes(
        np.stack(passes).astype(np.uint8).tobytes()
    )
    (directory / "expected.bin").write_bytes(canvas.astype(np.uint8).tobytes())
    return {"canvas": canvas, "passes": passes}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_browser_boost_reproduces_the_python_boost(tmp_path):
    fixtures = build_fixtures(tmp_path / "fixtures")

    assert fixtures["canvas"].shape == (ALIGN * SCALE, ALIGN * SCALE, 3)
    assert fixtures["canvas"].std() > 20, (
        "the fixture should contain real detail, or parity is trivially satisfied"
    )

    result = subprocess.run(
        ["node", str(NODE_SCRIPT), str(tmp_path / "fixtures")],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, (
        f"the JavaScript boost disagrees with the Python one:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "PARITY OK" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_parity_check_fails_when_the_javascript_is_wrong(tmp_path):
    """A parity test that cannot fail is a parity test that proves nothing.

    The fixture is built, then one pass buffer is nudged by a single byte — which is what a
    wrong phase or a transposed interleave would look like — and the checker must reject it.
    """
    build_fixtures(tmp_path / "fixtures")
    passes = tmp_path / "fixtures" / "passes.bin"
    damaged = bytearray(passes.read_bytes())
    damaged[0] = (damaged[0] + 1) % 256
    passes.write_bytes(bytes(damaged))

    result = subprocess.run(
        ["node", str(NODE_SCRIPT), str(tmp_path / "fixtures")],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode != 0, "a corrupted pass was accepted"
    assert "PARITY FAILED" in result.stderr
