# Neural Face Studio MVP

A clean production-oriented rebuild for consensual, self-only real-time face transformation.

## What is implemented

- Side-by-side camera and processed output
- Explicit one-click camera control
- Ephemeral identity sessions with TTL eviction
- Self-verification gate supported by the neural backend
- WebRTC send/receive video transport with STUN/TURN negotiation
- Latest-frame-wins scheduler with stale-frame dropping
- WebRTC data-channel telemetry for latency, FPS, tracking, and verification
- Automatic connection recovery and peer cleanup
- Short-lived HMAC TURN REST credentials; no permanent browser password
- Configurable Coturn deployment and active-peer admission control
- Motion-aware temporal stabilization that backs off during movement
- Stable-frame flicker suppression without fixed frame accumulation
- Optional 19-class semantic face parser and face-local compositor
- Preservation of parsed glasses, hair, hats, jewelry, neck, and clothing
- Feathered skin, feature, lip, and optional ear blending
- Speed, Balanced, and Quality live presets
- Automatic latency-driven inference resolution scaling
- Stable one-frame queues instead of latency-producing backlogs
- Randomized center/side/side/center active liveness challenge
- Multi-frame pose holds, challenge timeout, and identity match gating
- Browser-local processed-stream recording with no server upload
- Downloadable timestamped WebM clips and recording timer
- Fullscreen OBS clean-output mode with embedded synthetic label
- Keyboard shortcuts: `R` recording, `O` clean output, `Escape` exit
- Installable mobile PWA with offline shell caching
- Front/rear camera switching and screen wake lock
- Optional microphone-synchronized local recording
- Native mobile share-sheet support for completed clips
- Safe-area-aware iOS and Android responsive layout
- Self-hosted LiveKit room creation and joining
- Short-lived, room-scoped call JWTs issued only by the backend
- Processed Eidomira output published as the call camera track
- Direct microphone publishing that bypasses neural inference
- Consent-first training manifest schema and fail-closed validator
- Subject-disjoint split and duplicate-asset leakage checks
- Argon2id password hashing and short-lived signed access tokens
- SQLite WAL account and consent metadata store
- Durable grant/revoke consent records without raw biometric logging
- Per-route sliding-window rate limits and enrollment throttling
- Privacy-safe Prometheus request, latency, session, and peer metrics
- CSP, permissions policy, anti-framing, no-sniff, and no-store headers
- Restricted cross-origin API policy
- Compatibility WebSocket endpoint for constrained diagnostics
- Mandatory synthetic label from the neural adapter
- Full-resolution diagnostic transport mode without neural downscaling
- H.264-first WebRTC codec preference with VP8 fallback
- Honest diagnostic mode when no GPU model is mounted
- Optional InsightFace/InSwapper adapter
- NVIDIA Docker and RunPod deployment configuration

## Public landing page

`static/index.html` is the marketing page. It is self-contained, dependency-free, and
ships as static files: `landing.css` (design system + sections + motion), `landing.js`
(interactions), `fonts.css` + `static/fonts/*.woff2` (self-hosted Instrument Serif,
Inter Tight, JetBrains Mono — no third-party requests, so `style-src 'self'` holds).

Design notes:

- Dark editorial art direction: 8px baseline rhythm, one accent gradient (violet → cyan),
  serif italic display contrast against a tight grotesk.
- Motion: word-by-word hero reveal, rAF scroll progress, eased cursor spotlight,
  pointer tilt on specimens, image-wipe on hover, sticky scroll-spy steps, sliding
  toolkit tabs, drag/arrow use-case rail, count-up stats, height-animated FAQ,
  magnetic primary buttons, live canvas signal trace in the hero monitor.
- Every animation is disabled or reduced under `prefers-reduced-motion`, the hero
  gallery is illustrative (no real faces), and the page states plainly that it is a
  preview rather than a live feed.
- The hero monitor is a styled illustration of the Studio; telemetry values are
  illustrative, not measured.

`/` and `/static/*` are embeddable (no session state, no credentials) so the marketing
page can be hosted in previews, docs and product embeds. `/app` and `/api/*` keep
`X-Frame-Options: SAMEORIGIN` and `frame-ancestors 'self'`.

## Local UI/transport test

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Diagnostic mode shows the real camera/transport path but intentionally does not fake neural transformation.

## GPU mode

The repository does not redistribute face-swap model weights. Place a legitimately licensed model at `licensed-models/inswapper_128.onnx`, then use:

```bash
docker compose up --build
```

InsightFace code and pretrained weights have different licenses. Pretrained InSwapper weights are non-commercial unless separately licensed by their owner. Confirm licensing before deployment.

## Important MVP limitation

Direct peer-to-worker WebRTC is implemented for the MVP. Before a high-concurrency launch, add a production TURN service and regional SFU/gateway rather than terminating every public peer directly on GPU workers. The current scheduler deliberately drops stale frames instead of accumulating latency, but GPU admission control is still required for multiple simultaneous neural sessions.

## Safety boundary

The default product policy is self-only enrollment. The neural backend compares the live face embedding to the enrolled reference before processing. Reference state is held in memory and expires automatically. Camera frames are not written to disk.

## Accounts, trial, credits, and Live Pro

Eidomira includes verified-email registration, Argon2id passwords, signed access tokens, one verified trial per account, an auditable credit ledger, plan/tool quotes, and customer entitlements.

- 7-day trial after email verification, 100 credits, no card required
- Eidomira Live Pro: $39.99/month or $399/year, 1,500 monthly credits
- Trial excludes Live Swap, Character Swap, Talking Avatar, Voice Cloning, and API access
- Pro removes promotional branding, never mandatory synthetic-media provenance
- Checkout remains disabled until Stripe and/or Paystack signed webhooks are configured

Set `STUDIO_PUBLIC_URL` and SMTP values from `.env.example`. Without SMTP, development verification URLs print to server logs; this must not be used in production.

Endpoints: `POST /api/auth/register`, `/api/auth/login`, `/api/auth/verify-email`, `/api/auth/resend-verification`; `GET /api/plans`, `/api/billing/account`; `POST /api/billing/quote`.
