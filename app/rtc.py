from __future__ import annotations

import asyncio
import json
import time
from fractions import Fraction

import av
import cv2
from aiortc import MediaStreamTrack, RTCPeerConnection

from app.config import settings
from app.temporal import MotionAwareStabilizer
from app.adaptive import AdaptiveQualityController


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
                                self.engine.process, rgb, self.session.identity, True
                            )
                            result_rgb, latency, face_found = result.image, result.latency_ms, result.face_found
                        else:
                            result_rgb, latency = rgb, 0.0
                else:
                    result = await asyncio.to_thread(
                        self.engine.process, rgb, self.session.identity, self.session.verified
                    )
                    result_rgb, latency, face_found = result.image, result.latency_ms, result.face_found

                if self.engine.name != "diagnostic":
                    result_rgb = self.stabilizer.apply(rgb, result_rgb)
                output = av.VideoFrame.from_ndarray(result_rgb, format="rgb24")
                output.pts = frame.pts
                output.time_base = frame.time_base or Fraction(1, 90000)
                self.processed += 1
                self.quality.observe(latency)
                elapsed = max(.001, time.monotonic() - self.started)
                self._send({"type": "metrics", "inference_ms": round(latency, 1),
                            "fps": round(self.processed / elapsed, 1), "dropped": self.dropped,
                            "face_found": face_found, "inference_width": rgb.shape[1],
                            "quality": self.quality.preset,
                            "quality_adjustments": self.quality.adjustments})
                self._replace(self.output, output)
        except Exception as exc:
            self._send({"type": "error", "message": str(exc)})
            await self.stop()

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
