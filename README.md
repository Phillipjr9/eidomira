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

The adapter does not assume CUDA. At startup `app/providers.py` asks ONNX Runtime
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

### Closing the realism gap

The gap between Eidomira's raw swap and the best tools is **not the swap model**, and it is
worth being precise about that, because the obvious move — find a stronger swapper — leads
nowhere. `inswapper` already leads the one-shot class on identity retrieval (93.52 against
SimSwap's 92.25 and DeepFaceLab's 89.56) and wins Attribute, Anti-Occlusion and Fidelity
too. Its characteristic failure is that it *drifts toward the target face instead of
producing obvious artifacts*, so a raw result reads as soft and slightly wrong rather than
visibly broken.

The difference is the stack around it. Every production pipeline converges on the same one:

```
detect → swap at 128 → pixel-boost to 256/512 → restoration at 0.7–0.8 visibility
       → occluder-preserving mask → LAB tone transfer → soft-mask paste
```

`inswapper_128` emits a 128×128 face, which is why a raw swap looks soft until restoration
is stacked on top. Two parts of that stack are now implemented:

| Stage | Where | Effect (synthetic frames) |
| --- | --- | --- |
| LAB tone transfer | `app/enhance.py` → `transfer_tone()` | A face lit 25 levels from its room moves from 146 to **165** against a surround of 169; colour distance **32.9 → 7.2**. Symmetric in the other direction, and a face that already matches moves **0.04** levels. |
| Restoration at partial visibility | `app/enhance.py` → `FaceRestorer` | **Inactive.** Starts only when a model file is present. |

Restoration is deliberately a **pluggable file** rather than a hardcoded dependency, and
which file it may be is a licensing question rather than a technical one:

| Model | Licence | Shippable in a paid product |
| --- | --- | --- |
| GFPGAN v1.4 | Apache-2.0 | **yes** |
| GPEN-BFR-512 | Apache-2.0 code | yes, verify the weights |
| CodeFormer | NTU S-Lab License 1.0 | **no** — non-commercial; the strongest on paper, and not usable here |
| inswapper_128 | non-commercial research (code MIT) | not as-is — InsightFace **sells** a commercial licence |

Code and model licences have to be checked separately: InsightFace's code is MIT while its
pre-trained models are not. `docs/quality-and-licensing.md` records the full inventory and
the sources behind it.

Settings, all optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STUDIO_TONE_TRANSFER_STRENGTH` | `1.0` | How far to pull the swapped face into the target's lighting. `0` disables it. |
| `STUDIO_RESTORATION_MODEL_PATH` | `models/gfpgan_1.4.onnx` | Restoration model. Absent file means the stage is skipped, not an error. |
| `STUDIO_RESTORATION_VISIBILITY` | `0.75` | Blend strength for restoration. Full strength looks airbrushed. |

Tone transfer's statistics are taken from inside the feather rather than across it: the
transition ring is part target and part source, so including it pulls both sets of
statistics toward each other and quietly weakens the correction. The gain is clamped, which
matters more than it sounds — an unbounded gain matches the reference's *contrast*, so a
low-contrast face carrying sensor grain gets its grain stretched to full texture strength.
On the synthetic case that is a 9.3× contrast increase without the clamp and 0.93× with it.

#### Pixel-boost: getting back what the 128 crop threw away

`inswapper` works on a fixed 128×128 aligned crop, so a face that is 200 pixels across in
the frame is sampled at 0.64× and everything finer than that is gone before the result is
ever pasted back. That is what "raw swaps look soft" means.

`app/boost.py` recovers it without another model. Nudge the crop by half a pixel, run the
swapper again, and the second pass sees a *different sampling* of the same face. `scale`²
passes at the sub-pixel offsets of a `scale × scale` grid carry exactly as many samples as
a `scale × 128` canvas — four 128-squares are one 256-square, sample for sample — so the
canvas is built by interleaving them. Not averaging: averaging means upscaling and blending
the interpolated results, which measured *worse than a single pass* because the
interpolation cost more detail than the phases added.

Measured through the engine on a 200-pixel face whose detail is at the resolution limit:

| `STUDIO_SWAP_PIXEL_BOOST` | passes | face detail recovered | error against the real face |
|---|---|---|---|
| 1 (default) | 1 | 44 of 7744 | 30.1 |
| 2 | 4 | 1695 | 15.1 |
| 3 | 9 | 3980 | 14.9 |

The tests assert the stronger statement, as an identity rather than a similarity: with a
resolution-limited stand-in, `scale`² passes are *bit-for-bit* the aligned crop rendered at
`scale × 128`. That is a resolution recovery, not a sharpening filter.

**What is not proven:** a real swapper is generative, and its output is not a plain
resampling of its crop, so the gain will be smaller than the table above. Two things are
known to limit it. The interleave gives every output pixel exactly one pass's sample, so it
cannot average a pass's own noise away — measured: output noise equals input noise. And
against a face *smaller* than the crop there is nothing to recover, so the engine skips the
boost entirely rather than pay four passes for it; a test holds that.

It is off by default. Turning it on runs the swap `scale`² times per frame, which is a
different latency decision on a CPU box than on a GPU one, and the path has never executed
against a real model — `insightface` is not installed here and there are no weights. The
three assumptions it makes about that library are listed on `_boosted_swap`, and a failure
falls back to one pass for the rest of the session rather than failing every frame.

Still missing from the stack, in the order they block a launch: **trained swap weights in
`models/`** and the **inswapper commercial licence**. Until both exist, nothing here has
been run against a real swap, no quality claim is a measurement, and no parity with any
competitor is claimed.

### The trainer: watching for defects, and tuning what it may

`app/trainer.py` watches every sampled frame of a live session. **It cannot retrain the
swap model** — there are no weights in `models/`, no training data, and no accelerator in
this loop — so anything claiming otherwise would be inventing numbers. What it does is real:

| | |
| --- | --- |
| **Monitor** | Every sampled frame is scored with the same metrics as `tools/quality_report.py`, and each defect is recorded with the measurement that produced it: a colour mismatch, a visible edge, a frame that claimed a face and changed nothing. |
| **Tune** | When a measured defect has a knob that addresses it, it moves one step, then *verifies* the change against later samples and reverts it if the metric did not improve. A knob reverted twice is locked for the session, and a reverted knob gets a cooldown — otherwise the trainer re-applies the change it just disproved and spends the session oscillating. |
| **Repair** | A stage that reports a fault is switched off rather than run on every remaining frame, and recorded as a defect. A parser that keeps returning no face pixels falls back to the box mask rather than compositing nothing. |
| **Report** | `reports/<session>-<time>.md`: what was wrong, what changed and why, what was tried and reverted with the numbers, what is locked, and what no setting fixed. `reports/example-session-report.md` is a real one, from a session where the engine leaves the swapped face 30 levels dark. |

The panel in the studio (`/app`) shows the same thing live, over the telemetry channel.

#### What it is allowed to touch

Bounds live in `app/knobs.py`, shared by the live trainer and the offline one, because an
unbounded search for "better" always finds a setting that scores well and looks wrong.

| Knob | Range | Why the bound is there |
| --- | --- | --- |
| `tone_transfer_strength` | 0 – 1.0 | Past 1.0 the correction stops matching the target's light and starts replacing the swapped face with it. |
| `restoration_visibility` | 0 – 1.0 | Full strength is reported to look airbrushed. |
| `parser_feather` | .01 – .08 | Lower and the composited edge is a visible line; higher and the mask stops protecting hair and glasses. |

`temporal_strength`, `verification_threshold`, `max_frame_width` and the rest are listed in
`FIXED` with the reason, so "it did not touch this" is a decision on record: temporal
strength already belongs to the adaptive controller, and a second controller moving it is
how the two would fight.

Values travel to the engine **per call**, never written to `settings`, because the engine is
shared by every live session and one session's tuning must not change another's output.

#### Why measurement is sampled

A full-resolution quality sample costs **64 ms at 960×738** on this hardware — more than the
entire 45 ms frame budget. So measurement happens on a frame downscaled to 256 px (5 ms),
only every 30th frame, and never on a frame that was already over budget: spending 5 ms
measuring a frame that is already late makes the lateness worse.

#### The offline trainer

`tools/train_defaults.py` replays frames and searches the same three knobs, writing what it
measured to `data/tuned_defaults.json`, which `app/config.py` reads at startup and clamps.
Delete the file to go back to the configured defaults.

```bash
python -m tools.train_defaults --synthetic 2          # generated cases with known defects
python -m tools.train_defaults --pairs ./captures     # real saved *_original/_swapped pairs
python -m tools.train_defaults --pairs ./captures --json --no-write
```

Generated cases prove the search works. They cannot tell you what *your* sessions get
wrong, which is the question worth asking — so set `STUDIO_TRAINER_CAPTURE_LIMIT=200` and
the trainer writes every measured frame to `captures/` as the pair the tool reads. Then:

```bash
python -m tools.train_defaults --pairs captures       # searches the defects that happened
```

That loop is the one that matters: the live session records its own failures, and the
offline search answers whether a different setting would have avoided them. Nothing is
captured by default — two PNG encodes cost more than the measurement itself, and the
frames contain faces, so turning it on is a deliberate act with a deliberate directory.

It refuses to do two things, both of which would be easy and both of which would be wrong:

- **Score a knob it cannot evaluate.** `restoration_visibility` needs a restoration model to
  re-run; with none installed it says so rather than inventing a value.
- **Report its own noise as an improvement.** A change is only recommended if it beats the
  current value by 15%, and never at a bound: these metrics are one-sided — nothing in them
  penalises over-correcting a face or over-softening a mask — so a preference for the
  extreme is not evidence. Each recommendation prints what that metric cannot see.

Both rules exist because the first version broke both. Scored over a region derived from the
frames being compared, the tone curve came out non-monotonic (0.6 scored 4.9 and 0.7 scored
9.3, which no smooth blend parameter can do); pinning the region per pair made it monotone
again. The margin and the bound rule came from the same failure in miniature: the search
wanted to move `tone_transfer_strength` from 1.0 to 0.9 for a 10% gain, and `parser_feather`
to its maximum.

On generated pairs the search now finds nothing worth changing, which is the correct answer
for data with no unknown in it — and it does find a genuinely wrong default: with
`STUDIO_TONE_TRANSFER_STRENGTH=0.2` it recommends 0.9, an 89% improvement.

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
