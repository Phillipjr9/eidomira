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
ships as static files: `landing.css` (landing sections + motion), `landing.js`
(interactions), `fonts.css` + `static/fonts/*.woff2` (self-hosted Instrument Serif,
Inter Tight, JetBrains Mono — no third-party requests, so `style-src 'self'` holds).

### Stylesheet layout

| File | Owns |
| --- | --- |
| `tokens.css` | Design tokens, reset, buttons, pills, reveal utilities, auth modal. Shared. |
| `landing.css` | Public page: nav, hero, gallery, steps, pricing, FAQ, footer. |
| `studio.css` | Private workbench, telemetry, calls dock, OBS clean output. |

`tokens.css` is the only place tokens are declared; both page sheets consume them with
`var()` and tests forbid `:root` in the page sheets, so the public page and the private
studio cannot drift. The studio page class is `body.workspace` (not `.studio`) because
`.studio` was already the two-column video grid inside it.

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

## Studio (`/app`)

The private workbench was previously a second marketing site bolted onto the tool: its own
hero, feature grid, workflow, benefit tabs and pricing section, on the old pale-blue skin.
It is now just the workbench, in the same design system as the public page, with the
duplicated marketing content removed (the landing page owns that job).

Kept intact: every id `app.js` queries — camera, output, liveness, recording, telemetry,
calls, account and billing controls — plus the `.stage` / `.stage.output` / `.viewport`
structure that OBS clean mode depends on, and the `.hasImage`, `.recording` and `.active`
class hooks `app.js` toggles. `tests/test_studio_contract.py` enforces all of it.

Because camera access is granted to top-level pages only, the studio is not embeddable:
`/app` keeps `SAMEORIGIN` and `frame-ancestors 'self'`.

## On-device demo (`/#try`)

The "Your camera. Your own device." section runs a real MediaPipe FaceLandmarker in the
visitor's browser against their own webcam and draws the 478-point mesh, contours, iris
points and a tracking reticle, plus live blink/smile meters from the blendshape output.

- Detection happens entirely client-side; no frame, embedding or landmark leaves the tab.
  Nothing is fetched until the visitor clicks enable, and the camera track is released on
  stop, on engine failure, and on page hide.
- The engine is a pinned CDN build (`@mediapipe/tasks-vision`, see `ENGINE_CANDIDATES` in
  `landing.js`) with a second version as fallback. The model is served from this repo by
  `GET /models/face_landmarker.task`.
- `GET /models/face_landmarker.task` is an explicit single-file route on purpose: a
  `StaticFiles` mount over `models/` would also expose licensed weights such as
  `inswapper_128.onnx`.
- CSP carries `'wasm-unsafe-eval'` (WebAssembly compilation only, not JS `eval`) and the
  CDN in `connect-src`/`worker-src`, because the wasm runtime is fetched at run time.
- Camera permission is a top-level-only capability, so inside an embedded frame the demo
  detects the framing and offers an "open in a new tab" link instead of failing silently.

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

### Execution providers

The adapter does not assume CUDA. At startup `app/engines/providers.py` asks ONNX Runtime
which providers the host exposes, then orders them by expected throughput — TensorRT,
CUDA, ROCm, MIGraphX, DirectML, CoreML, OpenVINO — and keeps CPU last as the fallback.
The resolved list is reported by `GET /api/health` as `provider`, `providers` and
`accelerated`, and shown in the studio status line, so a host that quietly fell back to
CPU is visible rather than merely slow.

| Variable | Meaning |
| --- | --- |
| `EIDOMIRA_PROVIDERS` | Comma-separated override, e.g. `cuda,cpu` or `directml`. Unknown or unavailable names fall back to automatic ordering. |
| `EIDOMIRA_TENSORRT` | Set to `1` to allow TensorRT. Off by default: it builds an engine on first use, which can take minutes and is the wrong default for a request path. |

When CUDA is absent the InsightFace detector is prepared with `ctx_id=-1`, because a
non-negative id asks for a GPU context the host does not have.

### What the live frame budget is spent on

The adaptive controller holds the frame pipeline at a latency target — 32 / 45 / 65 ms for
the speed / balanced / quality presets, in `app/adaptive.py` — by adjusting inference
resolution. It measures the **whole frame**: engine work, stabilisation and frame
conversion. Feeding it only the engine's own figure let the pipeline run about a third over
budget without the controller noticing.

Telemetry carries both numbers, and the studio panel shows them as `Latency` and `Frame`:

| Field | Meaning |
| --- | --- |
| `inference_ms` | The engine's own work: detection, swap, optional parser. |
| `frame_ms` | Everything that frame cost, which is what the controller acts on. |

The gap between them is worth watching. On a 2-vCPU cloud instance the stabiliser alone
measured **14.9 ms at 960×540 — 33% of the 45 ms budget** (8.1 ms at 768×432, 2.5 ms at the
512×288 floor). Frame conversion and colour conversion are negligible by comparison, around
0.1 ms.

Liveness and verification frames are deliberately excluded from the controller's samples.
They cost a different, one-off amount, and letting them in would shrink quality for the rest
of the session.

None of this requires a GPU. The CPU fallback runs the full pipeline; what changes is the
resolution and frame rate it can sustain. Offline photo and batch work is comfortable on
CPU. The live path is budgeted for an accelerator, and "accelerator" need not mean external:
the registry above resolves a local GPU, an integrated one through DirectML, Apple Silicon
through CoreML, or a hosted GPU identically.

Two things to know before sizing hardware. There are **no face-swap weights in this
repository** — `models/` holds only the MediaPipe bundle behind the on-device demo — and the
INSwapper weights are non-commercial, so what may be shipped is a licensing question that
comes before any hardware decision.

### Measuring output quality

Whether a composite looks real is not something to assert in a README. It is something to
measure, and `tools/quality_report.py` does, from frame pairs alone:

```bash
python -m tools.quality_report --pair original.png swapped.png
python -m tools.quality_report --pair a1.png b1.png --pair a2.png b2.png --json
```

| Signal | What it catches |
| --- | --- |
| `seam_ratio` | A discontinuity on the edge of the composited region. 1.0 means the boundary is no sharper than the texture it sits in. |
| `colour` | Distance between the composited region's colour and the skin immediately around it — a face pasted from a differently lit source. |
| `flicker` | Frame-to-frame instability of the region, *net of real motion*, because motion moves the original frames too. Give it two or more pairs to get this. |

The metrics are themselves tested against cases with known answers: a known 25-level colour
offset comes back as a distance of 43.9 (25×√3), a hard cut scores roughly 5× the seam of a
feathered one, and added jitter shows up while a travelling subject does not.

Three findings from building it, all silent in the old code:

- **`PROTECTED_OCCLUDERS` was declared and never read.** The morphological close that removes
  mask speckle also filled straight over parsed glasses, hair and jewellery, compositing the
  swapped face on top of them.
- **`parser_include_ears: false` did nothing** for the same reason: the close refilled the ears.
- **A parser emitting probabilities or single-channel logits produced an empty mask.**
  `astype(np.uint8)` collapses values in `[0, 1]` to zero and wraps negative logits into
  garbage class ids. Nothing raised; `blend()` simply returned the original frame and the
  swap silently did nothing.

Blending also rounded where it used to truncate. Truncation biased every blended pixel down by
up to one level, which the feedback in the temporal stabiliser turned into a residue that
never cleared — a frame returning to its true value settled one level short of it and stayed
there. And the stabiliser is now reset on any frame that did not actually produce a swapped
face, instead of blending the last swapped face back over it.

None of this has been seen running against a real swap, because there are no weights to run
it against. The metrics exist so that when there are, the claim is checked rather than
believed.

## Important MVP limitation

Direct peer-to-worker WebRTC is implemented for the MVP. Before a high-concurrency launch, add a production TURN service and regional SFU/gateway rather than terminating every public peer directly on GPU workers. The current scheduler deliberately drops stale frames instead of accumulating latency, but GPU admission control is still required for multiple simultaneous neural sessions.

## Third-party code review

See [`docs/third-party-review.md`](docs/third-party-review.md) before adding code borrowed
from another repository. It records reviewed repos with a verdict, and includes the checks
to run first — starting with reading build files, since hooks like MSBuild `PreBuild` run
on build and can execute anything.

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

### Sign-in behaviour worth knowing

**Unknown addresses cost the same as wrong passwords.** `authenticate()` used to return
before hashing anything when an address had no account, so the response time alone
revealed whether a given email was registered — 96 ms against 2 ms over HTTP, both
answering an identical 401. It now spends an equivalent Argon2 verification against a
throwaway hash. Disabled accounts take the same path, so they cannot be told apart from
unregistered ones either.

**Failures are throttled per account, not just per address.** The middleware limit keys on
the peer address as seen directly, so behind a proxy uvicorn does not trust every caller
shares one bucket. `login_limit_per_hour` (default 30) is a second limit keyed on a digest
of the email, which keeps working in that case and never holds an address in memory. Only
failures count against it, so signing in normally never locks anyone out.

The cost of that design, recorded rather than discovered later: once an account's failure
budget is spent, **even the correct password is refused until the window passes**, and
someone who knows an address can deliberately trigger that lockout. The budget is checked
before the password because that is the only ordering that slows guessing down. Raise
`login_limit_per_hour` to trade protection for availability, or lower it to do the reverse.

**A proxy must be declared, or rate limiting degrades silently.** uvicorn only rewrites the
peer address from `X-Forwarded-For` for proxies listed in `FORWARDED_ALLOW_IPS`
(loopback by default). Left at the default behind a reverse proxy, every caller shares one
bucket: the whole platform gets 120 requests a minute and one client can lock out sign-in
for everyone. Set it to the proxy address or network — `docker-compose.yml` carries a
commented example — and the app logs a warning at startup when `PUBLIC_URL` is `https://`
while only loopback is trusted.
