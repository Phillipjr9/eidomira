# Where to run the swap: the compute decision

The short answer is **a plain GPU virtual machine with inbound UDP** — not a serverless GPU
platform, and specifically not RunPod Pods, whose documentation says *"Pods do not support UDP
connections"* and *"Docker Compose is not supported"*. Both sentences rule it out for this
product, and the reasons are architectural rather than a matter of taste.

## Why a VM, in two facts about our own code

1. **The server is a WebRTC peer.** `app/rtc.py` runs `aiortc.RTCPeerConnection`; the browser
   posts an offer to `/api/webrtc/{sid}/offer` and the server answers. The media is SRTP over
   **UDP**. A host that forwards only TCP completes the signaling handshake and then delivers
   no frames at all — the failure looks like a hung page, not a firewall, and this is the
   single most expensive thing to discover after paying for a host.
2. **The port is ephemeral.** `aiortc` binds its ICE socket on a random high port and exposes
   no way to pin it. So a fixed `ports:` mapping cannot cover the media path, which is why
   `docker-compose.yml` uses host networking and why the host firewall must allow the
   ephemeral range (32768–60999 by default on Linux), or media must be relayed through coturn
   (`STUDIO_TURN_URLS` / `STUDIO_TURN_SECRET` are already supported).

RunPod Pods fail both facts. Serverless in general (Modal, RunPod Serverless, fal, Replicate)
fails the first: an HTTP function invocation cannot host a long-lived peer connection.

**A caveat before paying for anything: verify UDP.** One minute with
`python -m tools/udp_probe` on the candidate host answers it, and no provider's marketing page
answers it for you:

```bash
# on the candidate host
python -m tools.udp_probe serve --port 34789 --seconds 120
# from anywhere else — your laptop, a phone hotspot, a cheap VPS
python -m tools.udp_probe send --host <the host> --port 34789
```

A provider that cannot complete that exchange cannot run the live studio, however fast its
GPUs are.

## What to pick

| stage | host | what it costs | why |
|---|---|---|---|
| **Now — validation** | any hourly GPU VM with UDP: Vast.ai, DataCrunch, Hyperstack, Lambda, or a hyperscaler | **$0.39–1.50/hr**, billed by the second/minute | one hour answers "does the stack run with real weights", and that is the whole purchase |
| **Live, low volume** | the same VM, started for demo hours | L4 ≈$0.43–0.80/hr · A10G ≈$0.69/hr · A100 80GB ≈$1.39–2.17/hr | a live session needs a warm process for its duration, so you pay per session, not per month |
| **Later — upload tools** | serverless: Modal (`L4 $0.80/hr`, `A10 $1.10`, `A100 $2.50`, `H100 $3.95`, $30/mo free credit) or RunPod Serverless | ≈$7/mo at side-project volume | `face_swap_hd`, `lip_sync_hd` and `talking_avatar_hd` are request/response jobs — a perfect fit |
| **Steady state** | an always-on pod, once utilization passes ≈73% | A100 ≈$4,020/mo | below that crossover, scale-to-zero is cheaper |

Prices gathered **2026-10-08** from vendor pricing pages and comparison write-ups; sources
disagree at the margin (Modal's A100 80GB appears as both $2.10 and $2.50; RunPod's A100 as
$1.49, $2.17 and $3.49 across three pages). Treat the table as shape, not quote — check the
provider's own page before committing.

### About the serverless tools

Those three upload tools are **priced in `app/billing.py` and have no routes yet** — zero
occurrences in `app/main.py`. So serverless is the right home for them *when they exist*, and
paying for a serverless platform today buys nothing. The live swap is the only neural path
that works, and it needs the VM.

### Region, for a Lagos user base

| region | rough RTT from Lagos | note |
|---|---|---|
| AWS `af-south-1` (Cape Town), Azure South Africa North | ~60–90 ms | closest; GPU availability is thinner and pricier |
| EU — Paris, Frankfurt, Amsterdam | ~70–110 ms | the practical default: good latency, wide GPU stock |
| US East | ~180–250 ms | works for upload jobs, poor for live conversation |

## One command

`tools/deploy_vm.py` does the whole of §"Order of operations" below, in the order that catches
the expensive mistakes before they cost anything:

```bash
# on the fresh GPU host, from the repository root
python -m tools.deploy_vm --print --domain swap.example.com --tls   # plan only, changes nothing
python -m tools.deploy_vm --domain swap.example.com --tls
```

What it does, and what it refuses to do:

| step | behaviour |
|---|---|
| preflight | Docker, a GPU the *daemon* can see (not just one `nvidia-smi` finds), the licensed models, a free port. **Nothing is written until this passes.** |
| configure | writes `.env`: a generated `STUDIO_AUTH_SECRET` (47-char urlsafe), the public origin added to the allowed origins, the model paths. The secret is never rewritten on a re-run — that would sign every session out and look like a bug. |
| TLS | writes a `Caddyfile` and starts Caddy from `docker-compose.tls.yml`, because camera access needs a secure context anywhere but localhost and the point of this host is a live test from a phone. |
| start | `docker compose up -d --build`, with host networking (see `docker-compose.yml` for why that is a requirement, not a preference). |
| verify | polls `/api/health` until the engine is `inswapper` **and** `accelerated: true`. A CPU fallback is a **failure** here, not a warning: the studio would run, look fine, and be many times too slow. |
| prove | prints the real-device test, and optionally listens for the UDP probe while you run it from another machine. |

It never installs system packages: Docker, the driver and the NVIDIA Container Toolkit are the
host's business, and when one is missing it names what to install and stops. That is deliberate
— a deploy script that edits a machine it does not understand is worse than no script.

`docker-compose.yml` now requires `STUDIO_AUTH_SECRET` (`${STUDIO_AUTH_SECRET:?...}`), so
compose refuses to start the GPU deployment with the shipped default key rather than leaving a
forgeable signing key in place. The tool writes it; the manual path is
`python -c "import secrets; print(secrets.token_urlsafe(48))"`.

## Order of operations

1. Buy the licence (`docs/insightface-licence-request.md`) — nothing else is worth buying first.
2. Rent **one hour** of a UDP-capable GPU VM, copy the licensed artefacts into
   `licensed-models/`, and run one command:

   ```bash
   python -m tools.deploy_vm --domain swap.example.com --tls
   ```

   The health line is the acceptance test, and the tool reads it rather than printing it: the
   engine must be `inswapper` and the provider accelerated, or the deploy fails and says which
   of the two is wrong. A host that quietly fell back to CPU is the failure this exists to
   catch.
3. Run a real live session from a phone on mobile data — not from the same machine, because
   loopback hides every NAT and firewall problem there is.
4. Only then decide between keeping a VM warm and moving the batch tools to serverless.

## Where to sign up

The live studio is one self-contained box: `PeerRegistry` and `SessionStore` are **in-process**
and `max_active_peers` is **4 per process**, so the application is written for a single
instance. Splitting layers buys nothing until that state moves to Redis.

### The live studio — a dedicated GPU box with a raw public IP

| provider | what you get | price | why it fits |
|---|---|---|---|
| **Hetzner GEX44** — *the default* | RTX 4000 SFF Ada 20 GB, 64 GB RAM, 2×1.92 TB NVMe, **1 Gbit/s unlimited traffic**, Falkenstein/Nuremberg | **€184/mo** (~$211) + €79 setup | a dedicated machine with a real public IP and UDP, flat price, no egress bill. ~70–110 ms from Lagos. 20 GB of VRAM is comfortable: inswapper 4 GB + parser ~1 GB + GFPGAN ~2 GB |
| Hetzner GEX131 | RTX PRO 6000 Blackwell 96 GB, 256 GB RAM | €889/mo | only when one GPU stops being enough |
| Lambda · `gpu_1x_a10` | A10 24 GB, hourly | ≈$440/mo always-on | hourly is the right shape for the **validation hour**, and for demo days |
| Verda (ex-DataCrunch) · Hyperstack | A10/L40S/A100, hourly, public IPs | ≈$1–2.50/hr | neoclouds with self-service GPUs and real public IPs |
| GCP `g2-standard-4` · AWS `g5.xlarge` | L4 24 GB · A10G 24 GB | ≈$516/mo · ≈$734/mo | works, costs more, and egress is metered |

**Bandwidth is not the constraint** — and it is worth doing the sum, because it decides the
egress question. The adaptive controller caps the published frame at 960 px wide
(`app/adaptive.py`); with no explicit bitrate set, aiortc's encoder targets roughly 1.5–2.5
Mbps for that. At ~2 Mbps a session moves about 0.9 GB per hour, so four concurrent sessions
need ~16 Mbps — nothing — and Hetzner's flat unlimited gigabit removes the variable entirely.
On a metered host the same 100 hours of live sessions is ~100 GB, ≈$9 at AWS's $0.09/GB. Real,
but not existential; the flat rate is simply one less thing to watch.

### The batch tools, later

`face_swap_hd`, `lip_sync_hd` and `talking_avatar_hd` are request/response jobs, so they belong
on a serverless GPU once their routes exist (see the stages table above) — and they can share
the same `licensed-models` volume baked into an image.

### What to buy in what order

1. **Domain** — anywhere, ~$10/yr. TLS is free: Caddy gets it automatically from the
   `Caddyfile` `tools/deploy_vm.py` writes.
2. **One hourly GPU VM** (Lambda, Verda, Hyperstack, or a hyperscaler) for the licence-check
   hour in `docs/swap-requirements.md` §2. A few dollars.
3. **Hetzner GEX44** once a real user is going to use it. €184/mo flat, and it is the cheapest
   way to keep a live product online — compare $5,687/mo for a warm H100 pair or $4,020/mo for
   an always-on A100 on a hyperscaler.

## The free trial path — nothing paid until a real user shows up

Rent nothing. Two free routes cover the whole product, and they are different routes because the
two halves need different things.

### The website — Oracle Cloud Always Free

| what | detail |
|---|---|
| compute | **2 Arm OCPUs, 12 GB RAM**, always free (Oracle halved this from 4/24 in June 2026; most write-ups still quote the old figure) |
| storage / transfer | 200 GB block storage, **10 TB/month outbound** |
| network | a **real public IP**, so UDP works — unlike Vast.ai, unlike any PaaS |
| runs what | `docker compose up` as written; the whole site, sign-in, dashboard, credits, and the **on-device camera demo** |
| card | required for identity verification, never charged on Always Free |
| the catches | Arm capacity is often "out of capacity" — try Frankfurt or Singapore; idle instances can be reclaimed; support is forum-only |

Our dependencies all have `aarch64` wheels (checked against PyPI: `numpy`, `av`, `pylibsrtp`,
`opencv-python-headless`, `argon2-cffi-bindings`; `aiortc` is pure Python), so Arm is not a
problem for this stack.

**What it cannot do is the live GPU swap.** No free tier anywhere includes an always-on GPU —
Oracle's own docs say GPU shapes are paid-only. On 2 Arm cores the neural engine would fall
back to CPU and be unusably slow. That is fine: the public thing that impresses a visitor is
the on-device demo, which runs in their browser and needs no server GPU at all.

### The GPU hour — Google Cloud's $300 trial

`$300` for 90 days, card at signup, and the credit pays before the card. Which GPU to ask for:

| machine type | GPU | VRAM | price (us-central1, on-demand) | hours on $300 |
|---|---|---|---|---|
| `n1-standard-4` + 1×T4 | **T4** | 16 GB | **≈$0.55/hr** all-in ($0.35 GPU + ~$0.19 VM) | **≈550** |
| `g2-standard-4` | **L4** (Ada) | 24 GB | ≈$1.00/hr ($0.71 GPU + ~$0.28 VM) | ≈300 |
| A2 `a2-highgpu-1g` | A100 40 GB | 40 GB | ≈$3.67/hr | ≈80 — not needed |

**Pick the T4.** The VRAM math says 16 GB is enough with room to spare: inswapper 4 GB
(fp16 2 GB) + `buffalo_l` ~1–2 GB + the parser ~1 GB + GFPGAN ~2 GB ≈ **8–9 GB peak**. The L4
is the better GPU — newer, faster fp16, 24 GB — and worth it if you want headroom; at these
prices the validation hour costs cents either way, so choose the T4 and stop thinking about it.

**Two setup steps that block people, both easy to miss:**

1. **Upgrade the billing account to the paid tier.** GPUs are not available on a trial-only
   account. The $300 credit still pays first, so nothing is charged.
2. **Request a GPU quota.** New accounts start at **0 GPUs** — the VM will refuse to start until
   you ask for a quota increase (1 GPU, in the region you chose). Approval is usually quick.

**Region, for a Lagos user:** `europe-west1` (Belgium) or `europe-west4` (Netherlands) at
~70–110 ms; `africa-south1` (Johannesburg) if the GPU you want is in stock there. T4 stock
varies by region, so if `us-central1` is full, try the European ones.

Azure's equivalent is `$200` for 30 days, with `NCasT4_v3` (T4 16 GB) or `NVadsA10_v5` (A10).
Google's is the longer runway.

**Free notebooks are not an option, and it is worth saying why.** Colab and Kaggle hand out
free T4s, and neither can run this: they are notebooks, not servers — no inbound UDP, no
persistent process, no public IP a browser can open a WebRTC connection to. Modal's $30/month
free credit is real but it is HTTP request/response, so it is for the batch upload tools once
those routes exist, not for the live studio.

### And right now, for nothing at all

The Arena sandbox preview already runs the full site with the demo accounts. It is the same
`docker compose` deployment concept on a CPU box, and it has been the fast way to show every
change in this project.

### Summary

| phase | where | cost |
|---|---|---|
| show the site to anyone, today | Oracle Always Free (or the sandbox preview) | **€0** |
| validate the neural stack once licensed | Google Cloud trial credit | **$0** from the $300 |
| a real user using it daily | Hetzner GEX44 | €184/mo |

## What not to do

* **Do not put the live studio on RunPod Pods.** UDP is unsupported there, in their words.
* **Do not use Vast.ai for live sessions.** Its instances share public IPs with port mapping;
  a shared address cannot carry our ephemeral UDP socket. Cheap, and wrong for this.
* **Do not deploy to a PaaS** (Vercel, Render, Railway, Fly). None carries raw inbound UDP, and
  the in-process session state means a second instance would not share sessions anyway.
* **Do not put it on a serverless platform** and hope the WebRTC handshake is enough; the
  handshake is TCP and the media is not.
* **Do not buy an always-on A100 for a product with no live traffic yet.** At demo volume the
  same money buys hundreds of hours of an hourly VM.
* **Do not trust the health endpoint alone** that the host is usable. It reports the provider
  ONNX Runtime resolved; whether UDP reaches the machine is answered by the probe, not by the
  engine.
