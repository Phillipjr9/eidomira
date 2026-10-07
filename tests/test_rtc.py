"""What the live pipeline measures, and what it feeds the adaptive controller.

`LatestFrameProcessor` owns the per-frame path: engine, then stabiliser, then conversion
to an output frame. It used to hand the controller `result.latency_ms`, which covers only
the engine — so the controller was tuning against a number that omitted stabilisation and
conversion, and could sit well over budget while believing it was comfortably inside.

These tests drive the real processor with a stand-in engine and stabiliser whose costs are
known, so the accounting is checked arithmetically rather than by inspection.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import av
import numpy as np
import pytest

import app.rtc as rtc
from app.engines.base import FrameResult
from app.sessions import Session

ENGINE_MS = 20.0
STABILIZER_MS = 40.0
TOLERANCE = 5.0


class FakeEngine:
    """A neural adapter that reports only its own work, like the real ones do."""

    name = "inswapper"

    def process(self, rgb, identity, verified):
        time.sleep(ENGINE_MS / 1000)
        return FrameResult(rgb.copy(), True, verified, ENGINE_MS)

    def verify_self(self, rgb, identity):
        time.sleep(ENGINE_MS / 1000)
        return True, 0.9

    def observe_liveness(self, rgb):
        time.sleep(ENGINE_MS / 1000)
        return True, 0.0


class DiagnosticLikeEngine(FakeEngine):
    name = "diagnostic"


class SlowStabilizer:
    """Replaces the real stabiliser so its share of the frame has a known size."""

    def __init__(self, *args, **kwargs):
        pass

    def apply(self, input_rgb, output_rgb):
        time.sleep(STABILIZER_MS / 1000)
        return output_rgb


class FakeTrack:
    def __init__(self, count):
        self.count = count
        self.sent = 0

    async def recv(self):
        if self.sent >= self.count:
            await asyncio.sleep(3600)  # a real track block; the processor cancels it
        frame = av.VideoFrame.from_ndarray(
            np.zeros((180, 320, 3), dtype=np.uint8), format="rgb24"
        )
        frame.pts = self.sent
        self.sent += 1
        return frame


class RecordingChannel:
    readyState = "open"

    def __init__(self):
        self.messages = []

    def send(self, payload):
        self.messages.append(json.loads(payload))

    def of_type(self, kind):
        return [m for m in self.messages if m.get("type") == kind]


async def run_frames(engine, monkeypatch, verified=True, count=1, stabilizer=SlowStabilizer):
    """Drive the processor until the requested frames come out, and record what it saw."""
    monkeypatch.setattr(rtc, "MotionAwareStabilizer", stabilizer)
    session = Session("session-1", identity=object(), verified=verified)
    channel = RecordingChannel()
    processor = rtc.LatestFrameProcessor(
        FakeTrack(count), engine, session, telemetry=channel
    )
    observed = []
    original = processor.quality.observe
    processor.quality.observe = lambda ms: (observed.append(ms), original(ms))[1]
    try:
        for _ in range(count):
            await asyncio.wait_for(processor.recv(), timeout=10)
    finally:
        await processor.stop()
        await asyncio.sleep(0)  # let the cancelled tasks unwind inside this loop
    return observed, channel


def test_controller_sees_the_whole_frame_not_just_the_engine(monkeypatch):
    observed, channel = asyncio.run(run_frames(FakeEngine(), monkeypatch))

    metrics = channel.of_type("metrics")[0]
    assert metrics["inference_ms"] == pytest.approx(ENGINE_MS, abs=TOLERANCE)

    full_frame_ms = ENGINE_MS + STABILIZER_MS
    assert metrics["frame_ms"] >= full_frame_ms - TOLERANCE, (
        f"frame_ms reported {metrics['frame_ms']} ms but the frame cost at least "
        f"{full_frame_ms:.0f} ms"
    )

    assert len(observed) == 1
    assert observed[0] >= full_frame_ms - TOLERANCE, (
        f"controller was told {observed[0]:.1f} ms while the frame cost at least "
        f"{full_frame_ms:.0f} ms — stabilisation is invisible to it again"
    )
    assert observed[0] == pytest.approx(metrics["frame_ms"], abs=1.0)


def test_the_two_reported_numbers_are_actually_different(monkeypatch):
    """If these ever collapse into one value, the distinction has been lost again."""
    _, channel = asyncio.run(run_frames(FakeEngine(), monkeypatch))
    metrics = channel.of_type("metrics")[0]
    assert metrics["frame_ms"] - metrics["inference_ms"] >= STABILIZER_MS - TOLERANCE


def test_the_controller_can_still_lower_the_resolution(monkeypatch):
    """The reported cost has to reach the controller's decision, not just its log."""
    async def scenario():
        monkeypatch.setattr(rtc, "MotionAwareStabilizer", SlowStabilizer)
        session = Session("s", identity=object(), verified=True)
        processor = rtc.LatestFrameProcessor(FakeTrack(1), FakeEngine(), session)
        processor.quality.minimum = 128
        before = processor.quality.width
        try:
            for _ in range(processor.quality.interval):
                processor.quality.observe(ENGINE_MS + STABILIZER_MS)
        finally:
            await processor.stop()
            await asyncio.sleep(0)
        return before, processor.quality.width

    before, after = asyncio.run(scenario())
    assert after < before, "an over-budget frame cost did not move the resolution"


# ─────────────── which frames the controller is allowed to learn from ───────────────

def controller_view(engine_name: str) -> SimpleNamespace:
    return SimpleNamespace(engine=SimpleNamespace(name=engine_name))


@pytest.mark.parametrize("engine_name", ["inswapper", "diagnostic", "base"])
def test_steady_state_frames_report_their_cost(engine_name):
    view = controller_view(engine_name)
    expected = 60.0 if engine_name != "diagnostic" else 0.0
    assert rtc.LatestFrameProcessor._observed_latency(view, True, 60.0) == expected


def test_liveness_and_verification_frames_are_not_sampled():
    """A one-off phase must not shrink quality for the rest of the session."""
    view = controller_view("inswapper")
    assert rtc.LatestFrameProcessor._observed_latency(view, False, 400.0) == 0.0


def test_a_zero_sample_leaves_the_controller_untouched():
    from app.adaptive import AdaptiveQualityController

    controller = AdaptiveQualityController("balanced", 960, 384, 8)
    before = controller.width
    for _ in range(20):
        controller.observe(0.0)
    assert controller.width == before
    assert not controller.samples


def test_liveness_phase_still_emits_frames_but_is_not_measured(monkeypatch):
    """The challenge phase must keep the video flowing while being excluded from tuning."""
    observed, channel = asyncio.run(run_frames(FakeEngine(), monkeypatch, verified=False))

    assert channel.of_type("liveness"), "no liveness challenge was sent"
    assert observed == [0.0], "a liveness frame was fed to the controller"
    assert channel.of_type("metrics"), "the frame was not emitted at all"
