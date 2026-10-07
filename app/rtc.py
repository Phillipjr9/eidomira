from __future__ import annotations

import asyncio
import json
import logging
import time
from fractions import Fraction

import av
import cv2
from aiortc import MediaStreamTrack, RTCPeerConnection

from app.config import settings
from app.temporal import MotionAwareStabilizer
from app.adaptive import AdaptiveQualityController
from app.trainer import Trainer


class LatestFrameProcessor:
    """Decouples capture, inference and output with one-slot queues.

    New camera frames replace stale queued frames. Slow inference increases
    dropped-frame count rather than latency, which is essential for live video.
    """

    def __init__(self, track, engine, session, telemetry=None):
        self.track = track
        self.engine = engine
        self.session = session
        self.telemetry = telemetry
        self.input: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.output: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.closed = False
        self.dropped = 0
        self.processed = 0
        self.started = time.monotonic()
        self.quality = AdaptiveQualityController(
            session.quality, settings.max_frame_width, settings.adaptive_min_width,
            settings.adaptive_interval_frames,
        )
        self.stabilizer = MotionAwareStabilizer(
            self.quality.temporal, settings.temporal_motion_threshold
        )
        # Diagnostic mode has no swap to judge, so there is nothing to watch: an enabled
        # trainer there would report a defect on every frame for doing exactly what
        # diagnostic mode is supposed to do.
        self.trainer = Trainer(
            getattr(session, "session_id", "session"),
            sample_every=settings.trainer_sample_every,
            enabled=settings.trainer_enabled and engine.name != "diagnostic",
            report_dir=settings.trainer_report_dir,
            active_stages=engine.active_stages(),
        )
        self.seen = 0
        self._last_frame_ms: float | None = None
        self._trainer_signature: tuple = ()
        self.capture_task = asyncio.create_task(self._capture())
        self.infer_task = asyncio.create_task(self._infer())

    @staticmethod
    def _replace(queue: asyncio.Queue, value):
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(value)

    async def _capture(self):
        try:
            while not self.closed:
                frame = await self.track.recv()
                if self.input.full():
                    self.dropped += 1
                self._replace(self.input, frame)
        except Exception:
            await self.stop()

    async def _infer(self):
        try:
            while not self.closed:
                frame = await self.input.get()
                self.seen += 1
                started = time.monotonic()
                steady_state = False
                face_found = False
                rgb = frame.to_ndarray(format="rgb24")
                # Diagnostic mode validates transport and must preserve the camera feed.
                # Neural mode may adapt resolution to prevent latency accumulation.
                if self.engine.name != "diagnostic" and rgb.shape[1] > self.quality.width:
                    ratio = self.quality.width / rgb.shape[1]
                    rgb = cv2.resize(rgb, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

                if settings.require_self_verification and not self.session.verified:
                    if not self.session.liveness.complete:
                        if self.engine.name == "diagnostic":
                            yaw = {"center": 0.0, "side_a": -.3, "side_b": .3}.get(
                                self.session.liveness.current, 0.0
                            )
                            face_found = True
                        else:
                            face_found, yaw = await asyncio.to_thread(
                                self.engine.observe_liveness, rgb
                            )
                        challenge = self.session.liveness.observe(yaw, face_found)
                        self._send({"type": "liveness", **challenge})
                        result_rgb, latency = rgb, 0.0
                        steady_state = False
                    else:
                        verified, score = await asyncio.to_thread(
                            self.engine.verify_self, rgb, self.session.identity
                        )
                        self.session.verified = verified
                        self.session.verify_score = score
                        self._send({"type": "verification", "verified": verified,
                                    "score": round(score, 3)})
                        face_found = True
                        if verified:
                            result = await asyncio.to_thread(
                                self.engine.process, rgb, self.session.identity, True,
                                self.trainer.overrides,
                            )
                            result_rgb, latency, face_found = result.image, result.latency_ms, result.face_found
                            steady_state = True
                        else:
                            result_rgb, latency = rgb, 0.0
                else:
                    result = await asyncio.to_thread(
                        self.engine.process, rgb, self.session.identity, self.session.verified,
                        self.trainer.overrides,
                    )
                    result_rgb, latency, face_found = result.image, result.latency_ms, result.face_found
                    steady_state = True

                if self.engine.name != "diagnostic":
                    # Blend only a frame that really produced a swapped face. Blending an
                    # untouched frame with the previous stabilised output paints the last
                    # swapped face back over it, and because the blend feeds its own output
                    # forward, the residue then holds instead of fading.
                    if steady_state and face_found:
                        result_rgb = self.stabilizer.apply(rgb, result_rgb)
                    else:
                        self.stabilizer.reset()
                # Measure before the frame is converted, and only frames the trainer can
                # judge: a frame that carried no swapped face is not evidence about the
                # quality of a swapped face. The metrics cost about 5 ms, so they run off
                # the loop and only on the frames `should_sample` accepts.
                metrics = None
                if steady_state and face_found and self.trainer.enabled:
                    over = (self._last_frame_ms is not None
                            and self._last_frame_ms > self.quality.target)
                    if self.trainer.should_sample(self.seen, over):
                        metrics = await asyncio.to_thread(
                            self.trainer.measure, rgb, result_rgb
                        )
                for stage, detail in self.engine.stage_faults().items():
                    self.trainer.notice_fault(stage, detail)

                output = av.VideoFrame.from_ndarray(result_rgb, format="rgb24")
                output.pts = frame.pts
                output.time_base = frame.time_base or Fraction(1, 90000)
                self.processed += 1
                frame_ms = (time.monotonic() - started) * 1000
                self.quality.observe(self._observed_latency(steady_state, frame_ms))
                self.trainer.observe(self.seen, frame_ms=frame_ms,
                                     target_ms=self.quality.target, face_found=face_found,
                                     metrics=metrics, steady_state=steady_state)
                self._last_frame_ms = frame_ms
                elapsed = max(.001, time.monotonic() - self.started)
                # Two numbers, because they answer different questions: inference_ms is the
                # engine's own work, frame_ms is everything this frame cost including
                # stabilisation and frame conversion.
                self._send({"type": "metrics", "inference_ms": round(latency, 1),
                            "frame_ms": round(frame_ms, 1),
                            "fps": round(self.processed / elapsed, 1), "dropped": self.dropped,
                            "face_found": face_found, "inference_width": rgb.shape[1],
                            "quality": self.quality.preset,
                            "quality_adjustments": self.quality.adjustments})
                self._send_trainer_state()
                self._replace(self.output, output)
        except Exception as exc:
            self._send({"type": "error", "message": str(exc)})
            await self.stop()

    def _send_trainer_state(self):
        """Send the panel its state when something changed, plus a slow heartbeat.

        Sending the whole ledger every frame would be most of the telemetry channel for no
        new information, and sending it only on change leaves a panel that has never
        received one after a session that found nothing.
        """
        signature = (sum(self.trainer.counts.values()), len(self.trainer.adjustments),
                     len(self.trainer.repairs), self.trainer.locked and 1)
        heartbeat = self.seen % 120 == 0
        if signature == self._trainer_signature and not heartbeat:
            return
        self._trainer_signature = signature
        self._send({"type": "trainer", **self.trainer.state()})

    def _observed_latency(self, steady_state: bool, frame_ms: float) -> float:
        """What the adaptive controller should learn from this frame.

        The controller decides the inference resolution, so it needs the whole per-frame
        cost, not the engine's own figure: stabilisation and frame conversion were
        previously invisible to it, which let the pipeline run well over budget while the
        controller believed it was comfortably inside.

        Only frames that ran the swap are representative — it changes the resolution of
        that work, and the liveness and verification phases cost a different, one-off
        amount that would otherwise shrink quality for the whole session. Diagnostic mode
        has no neural work to adapt.

        Returning 0 makes the controller skip the sample without changing the frame.
        """
        if not steady_state or self.engine.name == "diagnostic":
            return 0.0
        return frame_ms

    def _send(self, payload):
        channel = self.telemetry
        if channel and channel.readyState == "open":
            channel.send(json.dumps(payload))

    async def recv(self):
        return await self.output.get()

    async def stop(self):
        if self.closed:
            return
        self.closed = True
        current = asyncio.current_task()
        for task in (self.capture_task, self.infer_task):
            if task is not current and not task.done():
                task.cancel()
        # The report is the part of the trainer that outlives the session, so it is written
        # even if the stream ended badly.
        path = self.trainer.write_report()
        logging.getLogger("eidomira.trainer").info(
            "%s%s", self.trainer.summary_line(), f" — report at {path}" if path else ""
        )


class ProcessedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, processor: LatestFrameProcessor):
        super().__init__()
        self.processor = processor

    async def recv(self):
        return await self.processor.recv()

    def stop(self):
        super().stop()
        asyncio.create_task(self.processor.stop())


class PeerRegistry:
    def __init__(self):
        self.peers: set[RTCPeerConnection] = set()

    def __len__(self):
        return len(self.peers)

    def add(self, peer):
        self.peers.add(peer)

    async def discard(self, peer):
        self.peers.discard(peer)
        await peer.close()

    async def close_all(self):
        await asyncio.gather(*(pc.close() for pc in self.peers), return_exceptions=True)
        self.peers.clear()


peers = PeerRegistry()
