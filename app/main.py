from __future__ import annotations
import asyncio, time
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks, Request, Header
from fastapi.responses import FileResponse, JSONResponse, Response, RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription

from app.config import settings
from app.ice import rtc_configuration
from app.calls import create_room_name, create_call_token
from app.security import register, authenticate, access_token, optional_user, authenticated_user, issue_email_token, verify_email_token
from app.database import database
from app.mailer import send_verification
from app.billing import PLAN, TOOLS, create_trial, account as billing_account, quote as billing_quote
from app.paystack import checkout as paystack_checkout, verify_transaction as paystack_verify, valid_signature as paystack_valid_signature, process_webhook as paystack_process_webhook
from app.limits import rate_limit_middleware
from app.observability import metrics_middleware, ACTIVE_SESSIONS, ACTIVE_PEERS
from app.headers import security_headers_middleware
import json, time, uuid
from app.engines import create_engine
from app.sessions import SessionStore
from app.rtc import LatestFrameProcessor, ProcessedVideoTrack, peers

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Eidomira Live API", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.middleware("http")(metrics_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(security_headers_middleware)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
engine = create_engine()
sessions = SessionStore(settings.session_ttl_seconds, settings.max_sessions)


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


class CallRequest(BaseModel):
    display_name: str


class CallJoinRequest(CallRequest):
    room: str


class AuthRequest(BaseModel):
    email: str
    password: str


class ConsentRequest(BaseModel):
    scope: str = "live-identity-processing"
    policy_version: str = "2026-10-06"


class VerifyEmailRequest(BaseModel):
    token: str


class EmailRequest(BaseModel):
    email: str


class QuoteRequest(BaseModel):
    tool: str
    quantity: float
    api: bool = False


class CheckoutRequest(BaseModel):
    product: str = "live-pro-monthly"


@app.on_event("shutdown")
async def shutdown_peers():
    await peers.close_all()


def decode(data: bytes):
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError("Invalid image")
    if bgr.shape[1] > settings.max_frame_width:
        ratio = settings.max_frame_width / bgr.shape[1]
        bgr = cv2.resize(bgr, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode(rgb):
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
    if not ok: raise ValueError("Frame encoding failed")
    return data.tobytes()


@app.get("/")
def index(): return FileResponse(ROOT / "static" / "index.html")


@app.get("/models/face_landmarker.task", include_in_schema=False)
def landmark_model():
    """Public MediaPipe landmark bundle for the on-device landing demo.

    Deliberately an explicit single-file route rather than a StaticFiles mount over
    `models/`: that directory also holds licensed face-swap weights (`inswapper_128.onnx`)
    which must never be reachable over HTTP.
    """
    path = ROOT / "models" / "face_landmarker.task"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Landmark model not installed")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/app")
def private_app(user=Depends(authenticated_user)):
    return FileResponse(ROOT / "static" / "app.html")


@app.get("/metrics", include_in_schema=False)
def metrics():
    ACTIVE_SESSIONS.set(len(sessions.data)); ACTIVE_PEERS.set(len(peers))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/auth/register")
def auth_register(request: AuthRequest, background_tasks: BackgroundTasks):
    try:
        user=register(request.email,request.password)
        token=issue_email_token(user["id"])
        background_tasks.add_task(send_verification,user["email"],token)
        return {"verification_required":True,"message":"Check your email to activate your 7-day trial."}
    except ValueError as exc:
        return JSONResponse({"error":str(exc)},status_code=400)


@app.post("/api/auth/login")
def auth_login(request: AuthRequest):
    user=authenticate(request.email,request.password)
    if not user:return JSONResponse({"error":"Invalid email or password"},status_code=401)
    token=access_token(user)
    response=JSONResponse({"access_token":token,"token_type":"bearer","expires_in":settings.access_token_ttl,"user":user})
    response.set_cookie("eidomira_access_token",token,max_age=settings.access_token_ttl,httponly=True,secure=settings.public_url.startswith("https://"),samesite="lax",path="/")
    return response


@app.post("/api/auth/logout")
def auth_logout():
    response=JSONResponse({"ok":True})
    response.delete_cookie("eidomira_access_token",path="/")
    return response


@app.post("/api/auth/verify-email")
def auth_verify_email(request: VerifyEmailRequest):
    user=verify_email_token(request.token)
    if not user:return JSONResponse({"error":"Verification link is invalid or expired."},status_code=400)
    create_trial(user["id"])
    token=access_token(user)
    response=JSONResponse({"access_token":token,"token_type":"bearer","expires_in":settings.access_token_ttl,"user":user,"trial":billing_account(user["id"])})
    response.set_cookie("eidomira_access_token",token,max_age=settings.access_token_ttl,httponly=True,secure=settings.public_url.startswith("https://"),samesite="lax",path="/")
    return response


@app.post("/api/auth/resend-verification")
def auth_resend(request: EmailRequest, background_tasks: BackgroundTasks):
    user=database.one("SELECT id,email,email_verified_at FROM users WHERE email=? AND disabled=0",(request.email.strip().lower(),))
    if user and not user["email_verified_at"]:
        background_tasks.add_task(send_verification,user["email"],issue_email_token(user["id"]))
    return {"ok":True,"message":"If an unverified account exists, a new link has been sent."}


@app.get("/api/auth/me")
def auth_me(user=Depends(optional_user)):
    return user


@app.get("/api/plans")
def plans(): return {"plans":[PLAN],"trial":{"days":7,"credits":100},"tools":TOOLS}


@app.get("/api/billing/account")
def get_billing_account(user=Depends(optional_user)):
    if user["id"] == "local-guest": return JSONResponse({"error":"Authentication required"},status_code=401)
    return billing_account(user["id"])


@app.post("/api/billing/quote")
def get_billing_quote(request: QuoteRequest,user=Depends(optional_user)):
    if user["id"] == "local-guest": return JSONResponse({"error":"Authentication required"},status_code=401)
    try:return billing_quote(request.tool,request.quantity,api=request.api)
    except ValueError as exc:return JSONResponse({"error":str(exc)},status_code=400)


@app.post("/api/payments/paystack/checkout")
def create_paystack_checkout(request: CheckoutRequest,user=Depends(optional_user)):
    if user["id"] == "local-guest": return JSONResponse({"error":"Authentication required"},status_code=401)
    try:return paystack_checkout(user,request.product)
    except PermissionError as exc:return JSONResponse({"error":str(exc)},status_code=403)
    except ValueError as exc:return JSONResponse({"error":str(exc)},status_code=400)
    except RuntimeError as exc:return JSONResponse({"error":str(exc)},status_code=503)


@app.get("/api/payments/paystack/verify/{reference}")
def verify_paystack_payment(reference:str,user=Depends(optional_user)):
    if user["id"] == "local-guest": return JSONResponse({"error":"Authentication required"},status_code=401)
    try:return paystack_verify(reference,user["id"])
    except ValueError as exc:return JSONResponse({"error":str(exc)},status_code=404)
    except RuntimeError as exc:return JSONResponse({"error":str(exc)},status_code=503)


@app.post("/api/payments/paystack/webhook",include_in_schema=False)
async def paystack_webhook(request:Request,x_paystack_signature:str|None=Header(default=None)):
    raw=await request.body()
    if not paystack_valid_signature(raw,x_paystack_signature): return Response(status_code=401)
    try:paystack_process_webhook(raw)
    except Exception:return Response(status_code=500)
    return Response(status_code=200)


@app.post("/api/consents")
def grant_consent(request: ConsentRequest,user=Depends(optional_user)):
    if user["id"] == "local-guest":
        return JSONResponse({"error":"Create an account before recording durable consent."},status_code=401)
    consent_id=uuid.uuid4().hex
    database.execute("INSERT INTO consents(id,user_id,scope,policy_version,granted_at,metadata_json) VALUES(?,?,?,?,?,?)",
                     (consent_id,user["id"],request.scope,request.policy_version,int(time.time()),json.dumps({"source":"web"})))
    return {"id":consent_id,"scope":request.scope,"policy_version":request.policy_version}


@app.delete("/api/consents/{consent_id}")
def revoke_consent(consent_id:str,user=Depends(optional_user)):
    database.execute("UPDATE consents SET revoked_at=? WHERE id=? AND user_id=? AND revoked_at IS NULL",
                     (int(time.time()),consent_id,user["id"]))
    return {"ok":True}


@app.get("/api/health")
def health():
    return {"ok": True, "backend": engine.name, "gpu": engine.name != "diagnostic",
            "self_verification": settings.require_self_verification,
            "active_peers": len(peers), "peer_capacity": settings.max_active_peers,
            "turn_configured": bool(settings.turn_url_list and settings.turn_secret),
            "calls_configured": bool(settings.livekit_api_key and settings.livekit_api_secret)}


@app.post("/api/calls/rooms")
def create_call(request: CallRequest, user=Depends(optional_user)):
    try:
        room = create_room_name()
        return create_call_token(room, request.display_name)
    except (RuntimeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/calls/join")
def join_call(request: CallJoinRequest, user=Depends(optional_user)):
    try:
        return create_call_token(request.room, request.display_name)
    except (RuntimeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/sessions")
def enroll(image: UploadFile = File(...), consent: bool = Form(...), quality: str = Form("balanced"), user=Depends(optional_user)):
    if not consent:
        return JSONResponse({"error": "Explicit consent is required."}, status_code=400)
    if quality not in {"speed", "balanced", "quality"}:
        return JSONResponse({"error": "Invalid quality preset."}, status_code=400)
    try:
        enrollment = engine.enroll(decode(image.file.read()))
        session = sessions.create(enrollment.identity, quality, user["id"])
        return {"session_id": session.id, "backend": engine.name, "expires_in": settings.session_ttl_seconds}
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


@app.delete("/api/sessions/{sid}")
def reset(sid: str):
    sessions.delete(sid)
    return {"ok": True}


@app.get("/api/webrtc/{sid}/configuration")
def webrtc_configuration(sid: str):
    if not sessions.get(sid):
        return JSONResponse({"error": "Session expired"}, status_code=404)
    return rtc_configuration(sid[:16])


@app.post("/api/webrtc/{sid}/offer")
async def webrtc_offer(sid: str, offer: WebRTCOffer):
    session = sessions.get(sid)
    if not session:
        return JSONResponse({"error": "Session expired"}, status_code=404)
    if len(peers) >= settings.max_active_peers:
        return JSONResponse({"error": "All live engines are busy. Try again shortly."}, status_code=503)

    pc = RTCPeerConnection()
    peers.add(pc)
    telemetry = pc.createDataChannel("eidomira-telemetry")
    processors = []

    @pc.on("connectionstatechange")
    async def connection_state_changed():
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            for processor in processors:
                await processor.stop()
            await peers.discard(pc)

    @pc.on("track")
    def track_received(track):
        if track.kind != "video":
            return
        processor = LatestFrameProcessor(track, engine, session, telemetry)
        processors.append(processor)
        pc.addTrack(ProcessedVideoTrack(processor))

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    except Exception as exc:
        await peers.discard(pc)
        return JSONResponse({"error": str(exc)}, status_code=400)


# Compatibility fallback for environments where WebRTC/UDP is blocked.
@app.websocket("/api/live/{sid}")
async def live(socket: WebSocket, sid: str):
    await socket.accept()
    session = sessions.get(sid)
    if not session:
        await socket.send_json({"type": "fatal", "message": "Session expired"}); await socket.close(); return
    await socket.send_json({"type": "ready", "backend": engine.name})
    try:
        while True:
            data = await socket.receive_bytes()
            frame = decode(data)
            if settings.require_self_verification and not session.verified:
                session.verified, session.verify_score = await asyncio.to_thread(engine.verify_self, frame, session.identity)
                await socket.send_json({"type":"verification", "verified":session.verified,
                                        "score":round(session.verify_score, 3)})
                if not session.verified:
                    continue
            started = time.perf_counter()
            with session.frame_lock:
                result = await asyncio.to_thread(engine.process, frame, session.identity, session.verified)
            await socket.send_json({"type":"frame", "latency_ms":round(result.latency_ms,1),
                                    "face_found":result.face_found, "server_ms":round((time.perf_counter()-started)*1000,1)})
            await socket.send_bytes(encode(result.image))
    except (WebSocketDisconnect, RuntimeError, ValueError):
        return
