from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env")
    backend: str = "diagnostic"
    model_path: Path = Path("models/inswapper_128.onnx")
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
    parser_include_ears: bool = True
    target_inference_ms: float = 45.0
    adaptive_min_width: int = 384
    adaptive_interval_frames: int = 24
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    call_token_ttl: int = 3600
    call_room_prefix: str = "eidomira"
    database_path: Path = Path("data/eidomira.db")
    auth_secret: str = "development-only-change-me"
    access_token_ttl: int = 3600
    require_auth: bool = False
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
