from __future__ import annotations
from dataclasses import dataclass, field
import threading, time, uuid
from app.liveness import LivenessChallenge


@dataclass
class Session:
    id: str
    identity: object
    owner_id: str = "local-guest"
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    verified: bool = False
    verify_score: float = 0.0
    quality: str = "balanced"
    liveness: LivenessChallenge = field(default_factory=LivenessChallenge)
    frame_lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(self, ttl: int, maximum: int):
        self.ttl, self.maximum = ttl, maximum
        self.data: dict[str, Session] = {}
        self.lock = threading.Lock()

    def create(self, identity, quality: str = "balanced", owner_id: str = "local-guest"):
        with self.lock:
            self._purge()
            if len(self.data) >= self.maximum:
                oldest = min(self.data.values(), key=lambda s: s.touched)
                self.data.pop(oldest.id, None)
            session = Session(uuid.uuid4().hex, identity, owner_id=owner_id, quality=quality)
            self.data[session.id] = session
            return session

    def get(self, sid):
        with self.lock:
            self._purge()
            session = self.data.get(sid)
            if session: session.touched = time.time()
            return session

    def delete(self, sid):
        with self.lock: self.data.pop(sid, None)

    def _purge(self):
        cutoff = time.time() - self.ttl
        for sid in [k for k, v in self.data.items() if v.touched < cutoff]:
            self.data.pop(sid, None)
