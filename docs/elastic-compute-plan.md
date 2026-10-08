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

## What not to do

* **Do not put the live studio on RunPod Pods.** UDP is unsupported there, in their words.
* **Do not put it on a serverless platform** and hope the WebRTC handshake is enough; the
  handshake is TCP and the media is not.
* **Do not buy an always-on A100 for a product with no live traffic yet.** At demo volume the
  same money buys hundreds of hours of an hourly VM.
* **Do not trust the health endpoint alone** that the host is usable. It reports the provider
  ONNX Runtime resolved; whether UDP reaches the machine is answered by the probe, not by the
  engine.
