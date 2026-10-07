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
from app.config import settings
from app.engines.base import FrameResult
from app.sessions import Session

ENGINE_MS = 20.0
STABILIZER_MS = 40.0
TOLERANCE = 5.0


class FakeEngine:
    """A neural adapter that reports only its own work, like the real ones do."""

    name = "inswapper"

    def __init__(self):
        self.overrides_seen = []
        self.faults = {}

    def process(self, rgb, identity, verified, overrides=None):
        time.sleep(ENGINE_MS / 1000)
        self.overrides_seen.append(dict(overrides or {}))
        return FrameResult(rgb.copy(), True, verified, ENGINE_MS)

    def verify_self(self, rgb, identity):
        time.sleep(ENGINE_MS / 1000)
        return True, 0.9

    def observe_liveness(self, rgb):
        time.sleep(ENGINE_MS / 1000)
        return True, 0.0

    def stage_faults(self):
        return dict(self.faults)

    def active_stages(self):
        return {"restoration"}


class DiagnosticLikeEngine(FakeEngine):
    name = "diagnostic"


class NoFaceEngine(FakeEngine):
    """A frame where the engine reports no face: not evidence about swap quality."""

    def process(self, rgb, identity, verified, overrides=None):
        super().process(rgb, identity, verified, overrides)
        return FrameResult(rgb.copy(), False, verified, ENGINE_MS)


class SlowStabilizer:
    """Replaces the real stabiliser so its share of the frame has a known size."""

    def __init__(self, *args, **kwargs):
        self.resets = 0

    def apply(self, input_rgb, output_rgb):
        time.sleep(STABILIZER_MS / 1000)
        return output_rgb

    def reset(self):
        self.resets += 1


class FakeTrack:
    def __init__(self, count, fill=0):
        self.count = count
        self.fill = fill
        self.sent = 0

    async def recv(self):
        if self.sent >= self.count:
            await asyncio.sleep(3600)  # a real track block; the processor cancels it
        frame = av.VideoFrame.from_ndarray(
            np.full((180, 320, 3), self.fill, dtype=np.uint8), format="rgb24"
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
    return observed, channel, processor


def test_controller_sees_the_whole_frame_not_just_the_engine(monkeypatch):
    observed, channel, _ = asyncio.run(run_frames(FakeEngine(), monkeypatch))

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
    _, channel, _ = asyncio.run(run_frames(FakeEngine(), monkeypatch))
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
    observed, channel, _ = asyncio.run(run_frames(FakeEngine(), monkeypatch, verified=False))

    assert channel.of_type("liveness"), "no liveness challenge was sent"
    assert observed == [0.0], "a liveness frame was fed to the controller"
    assert channel.of_type("metrics"), "the frame was not emitted at all"


# ───────────── a frame that swapped nothing must not inherit the last face ─────────────

SWAP_PATCH = (slice(40, 60), slice(40, 60))
FILL = 255


class SequenceEngine:
    """Swaps on the frames its plan says, and reports no face on the others."""

    name = "inswapper"

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0
        self.swaps = 0

    def stage_faults(self):
        return {}

    def active_stages(self):
        return set()

    def process(self, rgb, identity, verified, overrides=None):
        swap, found = self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        image = rgb.copy()
        if swap:
            image[SWAP_PATCH] = 0            # an unmistakable swapped face
            self.swaps += 1
        return FrameResult(image, found, verified, 5.0)

    def verify_self(self, rgb, identity):
        return True, 0.9

    def observe_liveness(self, rgb):
        return True, 0.0


class PacedTrack(FakeTrack):
    """Paces delivery so the one-slot queue drops nothing while we are watching.

    The scene is deliberately constant. A brightness change would move the stabiliser's
    motion map to zero weight and hide the very behaviour under test: the ghost appears
    when the person stays still and a face detection drops out.
    """

    def __init__(self, count, interval=.05, fill=FILL):
        super().__init__(count, fill=fill)
        self.interval = interval

    async def recv(self):
        if self.sent >= self.count:
            await asyncio.sleep(3600)
        await asyncio.sleep(self.interval)
        frame = av.VideoFrame.from_ndarray(
            np.full((180, 320, 3), self.fill, dtype=np.uint8), format="rgb24"
        )
        frame.pts = self.sent
        self.sent += 1
        return frame


async def collect_frames(engine, monkeypatch, count, stabilizer=None):
    """Run the real stabiliser and hand back the frames the client would receive."""
    monkeypatch.setattr(rtc, "MotionAwareStabilizer", stabilizer or rtc.MotionAwareStabilizer)
    session = Session("session-1", identity=object(), verified=True)
    processor = rtc.LatestFrameProcessor(PacedTrack(count), engine, session)
    frames = []
    try:
        for _ in range(count):
            try:
                frame = await asyncio.wait_for(processor.recv(), timeout=3)
            except asyncio.TimeoutError:
                break
            frames.append(frame.to_ndarray(format="rgb24").copy())
    finally:
        await processor.stop()
        await asyncio.sleep(0)
    return frames


def test_a_swapped_frame_really_is_swapped(monkeypatch):
    """Guards the test below: if nothing was swapped, it could not detect a ghost."""
    engine = SequenceEngine([(True, True)])
    frames = asyncio.run(collect_frames(engine, monkeypatch, 1))
    assert engine.swaps >= 1, "the swap path never ran"
    assert frames and frames[-1][SWAP_PATCH].max() == 0, "the fake engine did not swap after all"


def test_a_frame_with_no_face_is_passed_through_untouched(monkeypatch):
    """The bug: the last swapped face was blended back over a frame that had none.

    The blend feeds its own output forward, so the ghost did not fade -- it settled just
    short of the true value and stayed there for every following frame. On a constant
    scene at 22% strength the patch settles around 199 instead of 255.
    """
    engine = SequenceEngine([(True, True), (False, False)])
    frames = asyncio.run(collect_frames(engine, monkeypatch, 2))

    assert engine.swaps >= 1, "the sequence never swapped anything, so nothing was proven"
    assert len(frames) == 2, "the paced track should have delivered both frames"
    assert frames[0][SWAP_PATCH].max() == 0, "the first frame should be the swapped one"

    last = frames[-1]
    assert (last == FILL).all(), (
        f"a frame with no face was altered: patch reads {int(last[SWAP_PATCH].max())}, "
        f"expected {FILL}"
    )


def test_the_stabiliser_is_cleared_when_a_frame_is_not_swapped(monkeypatch):
    """Reset, not just skipped: history would otherwise colour the next swapped frame."""
    engine = SequenceEngine([(True, True), (False, False), (True, True)])
    frames = asyncio.run(collect_frames(engine, monkeypatch, 3))
    assert len(frames) == 3
    assert int(frames[2][SWAP_PATCH].max()) == 0, "the third frame should be swapped afresh"


# ───────────────────────── the trainer on the live path ─────────────────────────

def test_the_trainer_state_reaches_the_studio(monkeypatch):
    """The panel is fed from the same channel the other telemetry uses."""
    _, channel, processor = asyncio.run(run_frames(FakeEngine(), monkeypatch))

    states = channel.of_type("trainer")
    assert states, "no trainer state was ever sent, so the panel would sit empty"
    assert states[0]["monitoring"] is True
    assert "counts" in states[0] and "values" in states[0]


def test_diagnostic_mode_does_not_watch_anything(monkeypatch):
    """There is no swap to judge, so every frame would be reported as a defect."""
    _, channel, processor = asyncio.run(run_frames(DiagnosticLikeEngine(), monkeypatch))

    assert processor.trainer.enabled is False
    assert channel.of_type("trainer")[0]["monitoring"] is False
    assert processor.trainer.counts == {}


def test_a_frame_without_a_face_is_not_measured(monkeypatch):
    """A frame that carried no swapped face is not evidence about swap quality."""
    monkeypatch.setattr(settings, "trainer_sample_every", 1)
    _, _, with_face = asyncio.run(run_frames(FakeEngine(), monkeypatch))
    _, _, without = asyncio.run(run_frames(NoFaceEngine(), monkeypatch))

    assert with_face.trainer.samples >= 1, (
        "the harness never sampled anything, so the assertion below would hold for the "
        "wrong reason"
    )
    assert without.trainer.samples == 0


def test_a_tuned_value_reaches_the_engine_on_the_next_frame(monkeypatch):
    async def scenario():
        monkeypatch.setattr(rtc, "MotionAwareStabilizer", SlowStabilizer)
        engine = FakeEngine()
        session = Session("session-1", identity=object(), verified=True)
        channel = RecordingChannel()
        processor = rtc.LatestFrameProcessor(FakeTrack(2), engine, session, telemetry=channel)
        try:
            processor.trainer.values["tone_transfer_strength"] = .4
            await asyncio.wait_for(processor.recv(), timeout=10)
        finally:
            await processor.stop()
            await asyncio.sleep(0)
        return engine

    engine = asyncio.run(scenario())
    assert engine.overrides_seen[-1].get("tone_transfer_strength") == .4


def test_a_stage_that_reports_a_fault_is_repaired_and_reported(monkeypatch):
    async def scenario():
        monkeypatch.setattr(rtc, "MotionAwareStabilizer", SlowStabilizer)
        engine = FakeEngine()
        engine.faults = {"restoration": "the model rejected an input frame"}
        session = Session("session-1", identity=object(), verified=True)
        channel = RecordingChannel()
        processor = rtc.LatestFrameProcessor(FakeTrack(1), engine, session, telemetry=channel)
        try:
            await asyncio.wait_for(processor.recv(), timeout=10)
        finally:
            await processor.stop()
            await asyncio.sleep(0)
        return channel, processor

    channel, processor = asyncio.run(scenario())
    assert processor.trainer.repairs[0].stage == "restoration"
    assert processor.trainer.count("restoration_fault") == 1
    # restoration gives way rather than being retried on every remaining frame
    assert processor.trainer.overrides["restoration_visibility"] == 0.0
    assert any("repairs" in state for state in channel.of_type("trainer"))


def test_a_report_is_written_when_the_stream_ends(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trainer_report_dir", tmp_path / "reports")
    asyncio.run(run_frames(FakeEngine(), monkeypatch))

    written = list((tmp_path / "reports").glob("*.md"))
    assert len(written) == 1, f"expected one session report, found {written}"
    assert "Session report" in written[0].read_text()


def test_the_report_survives_a_stream_that_ended_after_one_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trainer_report_dir", tmp_path / "reports")
    asyncio.run(run_frames(FakeEngine(), monkeypatch, count=1))
    report = next((tmp_path / "reports").glob("*.md")).read_text()
    assert "cannot retrain the swap model" in report


# ─────────────────── the loop end to end, on a defect it can fix ───────────────────

class NoSleepEngine(FakeEngine):
    """No sleeping: the per-frame cost is the measurement, not a stand-in model."""

    def process(self, rgb, identity, verified, overrides=None):
        import cv2

        from app.enhance import transfer_tone

        tone = float((overrides or {}).get("tone_transfer_strength",
                                           settings.tone_transfer_strength))
        out = rgb.copy()
        out[40:110, 100:190] = (out[40:110, 100:190].astype(int) - 30).clip(0, 255).astype("uint8")
        mask = np.zeros(rgb.shape[:2], np.float32)
        mask[45:105, 105:185] = 1.0
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        if tone > 0:
            out = transfer_tone(rgb, out, mask, tone)
        self.overrides_seen.append(dict(overrides or {}))
        return FrameResult(out, True, verified, 4.0)


class FastStabilizer:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, input_rgb, output_rgb):
        return output_rgb

    def reset(self):
        pass


def test_the_trainer_improves_a_session_that_has_a_defect_it_can_fix(monkeypatch):
    """The whole point, end to end: nobody touches anything and the output gets better.

    The engine leaves the swapped face 30 levels dark, and the tone stage can correct it.
    Starting from a deliberately low value, the trainer has to find its way up, keep the
    steps that helped, and stop when one stops helping.
    """
    monkeypatch.setattr(settings, "trainer_sample_every", 1)
    monkeypatch.setattr(settings, "tone_transfer_strength", .3)

    async def scenario():
        monkeypatch.setattr(rtc, "MotionAwareStabilizer", FastStabilizer)
        engine = NoSleepEngine()
        session = Session("climb", identity=object(), verified=True)
        processor = rtc.LatestFrameProcessor(PacedTrack(60, interval=.01), engine, session)
        seen = 0
        try:
            while seen < 60:
                try:
                    await asyncio.wait_for(processor.recv(), timeout=2)
                    seen += 1
                except asyncio.TimeoutError:
                    break
        finally:
            await processor.stop()
            await asyncio.sleep(0)
        return processor

    trainer = asyncio.run(scenario()).trainer
    kept = [a for a in trainer.adjustments if a.kept is True]

    assert kept, "the trainer never kept a change, so a fixable defect went unfixed"
    assert trainer.values["tone_transfer_strength"] > .3, "the value never moved"
    first, last = kept[0], kept[-1]
    assert last.measured_after < first.measured_before, (
        f"the defect did not improve: {first.measured_before} then {last.measured_after}"
    )
    assert last.measured_after < first.measured_before * .75, (
        "the improvement is too small to call this working"
    )


def test_a_live_session_saves_frames_for_the_offline_search(tmp_path, monkeypatch):
    """The wiring, not the writer: the frame the trainer measures is the frame saved.

    Without this the offline tool can only search generated cases, which is the "guess
    about real frames" it exists to avoid.
    """
    from tools.train_defaults import pairs_from_directory

    monkeypatch.setattr(settings, "trainer_sample_every", 1)
    monkeypatch.setattr(settings, "trainer_capture_limit", 2)
    monkeypatch.setattr(settings, "trainer_capture_dir", tmp_path)

    async def scenario():
        monkeypatch.setattr(rtc, "MotionAwareStabilizer", FastStabilizer)
        session = Session("capture", identity=object(), verified=True)
        processor = rtc.LatestFrameProcessor(
            PacedTrack(24, interval=.01), NoSleepEngine(), session
        )
        seen = 0
        try:
            while seen < 24:
                try:
                    await asyncio.wait_for(processor.recv(), timeout=2)
                    seen += 1
                except asyncio.TimeoutError:
                    break
        finally:
            await processor.stop()
            await asyncio.sleep(0)
        return processor

    trainer = asyncio.run(scenario()).trainer

    assert trainer.captures == 2, f"expected the cap to stop it at 2, got {trainer.captures}"
    pairs = pairs_from_directory(tmp_path)
    assert len(pairs) == 2, f"the offline tool cannot read what was saved: {list(tmp_path.iterdir())}"
    # A pair of the same frame, not two of the same image: the search scores the difference.
    assert not np.array_equal(pairs[0].original, pairs[0].swapped)
