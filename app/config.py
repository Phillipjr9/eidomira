from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.knobs import TUNED_DEFAULTS_PATH, apply_to_settings, read_tuned


#: The value shipped here and in `.env.example`. Not a secret: it is in the repository.
DEVELOPMENT_AUTH_SECRET = "development-only-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env")
    backend: str = "diagnostic"
    model_path: Path = Path("models/inswapper_128.onnx")
    #: Sub-pixel resolution recovery on the swap output. 1 is off, giving the raw 128px
    #: face; `scale` runs scale^2 passes of the swapper to fill a scale*128 canvas, so 2
    #: doubles the effective sampling of the swapped face at four times the swap cost.
    #: What app/boost.py proves about this, and what it does not, is written at the top
    #: of that module.
    swap_pixel_boost: int = 1
    max_sessions: int = 25
    session_ttl_seconds: int = 1800
    max_frame_width: int = 960
    jpeg_quality: int = 82
    verification_threshold: float = 0.34
    require_self_verification: bool = True
    allowed_origins: str = "http://127.0.0.1:8000,http://localhost:8000"
    max_active_peers: int = 4
    turn_urls: str = ""
    turn_secret: str = ""
    turn_realm: str = "eidomira.local"
    turn_credential_ttl: int = 3600
    temporal_strength: float = 0.22
    temporal_motion_threshold: float = 24.0
    parser_model_path: Path = Path("models/face_parser.onnx")
    parser_feather: float = 0.035
    # Post-swap refinement. The restoration model is optional and inactive unless the
    # file exists: see docs/quality-and-licensing.md for which ones may be shipped.
    restoration_model_path: Path = Path("models/gfpgan_1.4.onnx")
    restoration_visibility: float = 0.75
    tone_transfer_strength: float = 1.0
    parser_include_ears: bool = True
    # The trainer watches every sampled frame, records defects, and tunes the knobs above
    # within their bounds. It cannot retrain the swap model: there are no weights here.
    trainer_enabled: bool = True
    trainer_sample_every: int = 30
    #: Save measured frames as pairs the offline search can replay. 0 is off; every saved
    #: frame costs two PNG encodes on top of the measurement.
    trainer_capture_limit: int = 0
    trainer_capture_dir: Path = Path("captures")
    trainer_report_dir: Path = Path("reports")
    # Frame-latency targets live with the presets in app/adaptive.py (32/45/65 ms for
    # speed/balanced/quality). There is deliberately no single global target here: one
    # number cannot express three quality modes, and this field was never read by
    # anything, so setting STUDIO_TARGET_INFERENCE_MS silently did nothing.
    adaptive_min_width: int = 384
    adaptive_interval_frames: int = 24
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    call_token_ttl: int = 3600
    call_room_prefix: str = "eidomira"
    database_path: Path = Path("data/eidomira.db")
    # Signing key for session and verification tokens. Anyone who knows it can mint a
    # valid token for any account, and nothing behaves differently while it is wrong, so
    # app.main warns at boot rather than trusting it to be noticed.
    auth_secret: str = DEVELOPMENT_AUTH_SECRET
    access_token_ttl: int = 3600
    require_auth: bool = False
    #: One-click demo sign-in. Off by default, refused outright over https, and the demo
    #: passwords are random — see app/demo.py for why none of that is negotiable.
    demo_login: bool = False
    #: Origins allowed to frame the studio and the owner console, space or comma separated.
    #: Empty — the default — keeps `frame-ancestors 'self'`, which is what stops another site
    #: from putting the studio in a frame and collecting clicks meant for its buttons. It needs
    #: a value only where the app is deliberately embedded: a hosted preview is served inside a
    #: frame, and there the default refuses to render the page at all. `*` is accepted and
    #: discouraged.
    embed_ancestors: str = ""
    #: Optional, and deliberately blank: setting it *publishes* a credential, because the
    #: login card shows it and the accounts then accept it on the ordinary sign-in form. It
    #: only takes effect while demo_login is on, and it must satisfy the normal password
    #: rules. Left blank, the demo passwords stay random and the buttons are the only way in.
    demo_password: str = ""
    rate_limit_per_minute: int = 120
    enrollment_limit_per_hour: int = 20
    # Failed sign-ins allowed per account per hour, on top of the per-IP limit.
    # This is the limit that still works when the IP key is degraded, e.g. behind
    # a proxy uvicorn does not trust, where every caller shares one bucket.
    login_limit_per_hour: int = 30
    public_url: str = "http://127.0.0.1:8000"
    email_token_ttl: int = 86400
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = "Eidomira <verify@eidomira.invalid>"
    smtp_starttls: bool = True
    paystack_secret_key: str = ""
    paystack_monthly_plan_code: str = ""
    paystack_annual_plan_code: str = ""
    paystack_monthly_amount_kobo: int = 5990000
    paystack_annual_amount_kobo: int = 59900000
    paystack_topup_200_kobo: int = 1200000
    paystack_topup_500_kobo: int = 2750000
    paystack_topup_1500_kobo: int = 7500000
    paystack_topup_5000_kobo: int = 22500000

    @property
    def origin_list(self):
        return [v.strip() for v in self.allowed_origins.split(",") if v.strip()]

    @property
    def turn_url_list(self):
        return [v.strip() for v in self.turn_urls.split(",") if v.strip()]


settings = Settings()

# Values measured by `tools/train_defaults.py`, if it has run. Clamped on the way in, so a
# stale or hand-edited file cannot put the pipeline outside the ranges in app/knobs.py.
# Recorded rather than merely applied, because "why is this number not the one in
# .env.example" is a question that otherwise costs an hour.
TUNED_DEFAULTS_APPLIED: dict = apply_to_settings(
    settings, read_tuned(), source=str(TUNED_DEFAULTS_PATH)
)
