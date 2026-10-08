# What the swap still needs from outside this repository

Written in answer to *"what else do we need externally for a perfect swap"*. The pipeline is
already built: the swap adapter, the semantic compositor with occluder preservation, the
pixel-boost phase interleave, the tone transfer, the temporal smoothing, the trainer and its
knobs. What is missing is not code. It is **two artefacts, one licence, a host with a GPU, and
real material** — in that order of dependency.

## 1. The artefacts

| stage | file | licence | shippable in a paid product? |
|---|---|---|---|
| swap | `models/inswapper_128.onnx` | insightface pre-trained weights — **non-commercial** (the *code* is MIT; the weights are not) | **No.** Requires a purchase. |
| detect / align / embed | `buffalo_l` pack (downloaded by `FaceAnalysis`) | same agreement as the swap weights | **No.** Same purchase. |
| parser | `models/face_parser.onnx` | BiSeNet 19-class, as redistributed by FaceFusion — OpenRAIL-AS | Yes, with use-restrictions |
| restorer | `models/gfpgan_1.4.onnx` | GFPGAN v1.4 — Apache-2.0 (the ONNX conversion is third-party) | Yes |

`inswapper_128.onnx` is ~554 MB and takes ~1 GB resident per process. None of the four exists
in this repository, which is why the neural swap **has never executed here**.

### Fetching them

```bash
python -m tools.fetch_models --check        # what is present, and what each one is
python -m tools.fetch_models --print-urls   # where each comes from
python -m tools.fetch_models --fetch parser
python -m tools.fetch_models --fetch restorer
python -m tools.fetch_models --fetch swap --accept-licence   # only if you hold the licence
```

Two properties worth knowing:

* **It refuses the swap weights without `--accept-licence`.** The tool will not quietly put a
  non-commercial model into a paid product because it appeared in a table. The flag states
  that the operator holds a licence; nothing else does.
* **Hashes are recorded on the first fetch and verified on every one after.** The first fetch
  is trust-on-first-use — the canonical digest cannot be known without having fetched it once
  — but a source that later returns different bytes is a hard failure, and the file is removed
  rather than left in place. The URLs in the table are *unverified from inside this
  repository*: the sandbox cannot reach Hugging Face, so they are the best-known mirrors
  rather than tested ones. The tool prints what it actually received.

### Where the files go

Two conventions exist and both are fine, but they are not the same directory:

* **Local runs** read `models/…`, which is what `app/config.py` defaults to and what
  `tools/fetch_models.py --check` inspects by default.
* **The container** mounts `./licensed-models` read-only at `/models` and points
  `STUDIO_MODEL_PATH` and `STUDIO_PARSER_MODEL_PATH` there (see `docker-compose.yml`). That
  directory is gitignored.

So either fetch into `models/` for a local run, or fetch with
`--directory licensed-models` and let compose mount it. The setting always wins over the
default: the app reads whatever `STUDIO_MODEL_PATH` says.

## 2. Verifying the pipeline before spending anything

This is the part that can be done today, on a CPU, for free, and it is what makes the licence
conversation concrete rather than speculative.

```bash
pip install -r requirements.txt -r requirements-neural.txt
python -m tools.stand_in_models --directory models/standin
python -m pytest tests/test_neural_path.py -q
```

`tools/stand_in_models.py` writes three small ONNX graphs with the **exact shapes, names and
input order** the adapters require — a 128px swapper with the 512×512 embedding map where
`INSwapper` looks for it, a 19-class parser with a plausible face layout, a sharpening
restorer. They contain no trained weights and demonstrate no quality. They do prove:

* the graphs load through the real adapters, on the host's real execution providers;
* the parser produces a mask, and the occluder it labels is **not** covered — an occluder the
  compositor failed to preserve shows up as a non-zero mask over the glasses band;
* pixels the mask leaves at zero survive the blend **byte for byte**;
* the restorer enhances without faulting, and the boost interleaves four real inference passes
  onto one 256px canvas.

That is the plumbing verified end to end. Quality remains untested until the licensed
artefacts exist, and no claim about it should be made before then.

To run the app against the stand-ins, the tool prints the environment it needs:

```
STUDIO_MODEL_PATH=models/standin/inswapper_128.onnx
STUDIO_PARSER_MODEL_PATH=models/standin/face_parser.onnx
STUDIO_RESTORATION_MODEL_PATH=models/standin/gfpgan_1.4.onnx
STUDIO_BACKEND=insightface
```

`models/standin/` rather than `models/` on purpose: a stand-in must never be mistaken for a
licensed artefact, by a person or by `--check`.

There are two stand-in conventions in the repository, and they are for two different runtimes:
`static/lab/swapper-standin.onnx` is a committed 277-byte graph for the **browser** (the
`/lab` page runs ONNX Runtime Web), and `tools/stand_in_models.py` generates the three
**server** graphs, which are not committed (`*.onnx` is gitignored).

### Where the engine stops once the stand-ins are in place

Worth recording, because it is the measurement that says which purchase actually gates what.
With `STUDIO_BACKEND=inswapper` and all three stand-in paths set, `create_engine()` gets all
the way through the model file and stops at:

```
Downloading .../buffalo_l.zip from https://github.com/deepinsight/insightface/releases/...
stopped at: SSLError
```

That is `FaceAnalysis(name="buffalo_l")`, and it is the last thing standing between the
stand-ins and a running neural engine. The pack holds detection, 2D landmarks and ArcFace
embeddings; the swap adapter cannot align a face without it, and **it carries the same
non-commercial licence as the swap weights** — so it is part of the same purchase, not a
separate free download that `pip install` performs on a working host.

The order of the constructor is the order of the plan: swap model, then `buffalo_l`, then the
parser compositor, then the restorer. Each one is optional and reported; `GET /api/health`
says which of them the running process actually has.

## 3. A host with a GPU

**Read `docs/elastic-compute-plan.md` for the decision.** The short of it: the server is a
WebRTC peer (`app/rtc.py`, aiortc) and the media is SRTP over **UDP**, and `aiortc` binds an
ephemeral port with no way to pin it. So the host must be a plain virtual machine with inbound
UDP — not a serverless function, and not RunPod Pods, whose documentation says *"Pods do not
support UDP connections"* and *"Docker Compose is not supported"*. Verify any candidate in one
minute with `python -m tools.udp_probe` before paying for it.

The prices below are for capacity and cost, not for suitability — all of them assume the UDP
requirement is satisfied.

This sandbox is 2 cores, 3 GB, **no GPU** — the diagnostic engine, and a CPU path at best.
Live swap needs acceleration. Prices below were gathered on **2026-10-07** and move; treat
them as shape, not quote.

| host | rate | note |
|---|---|---|
| Modal L4 / A10 / A100 80 GB | $0.80 / $1.10 / $2.50 per hour | per-second billing, scale-to-zero |
| fal.ai H100 / A100 40 GB | $1.89 / $0.99 per hour | |
| RunPod pod A100 80 GB / H100 PCIe | $1.39 / $2.89 per hour | serverless A100 $2.72 |
| always-on A100 | ≈$4,020 per month | only worth it above ≈73% utilization |

VRAM: `inswapper_128` 4 GB (2 GB fp16), `simswap_512` 10 GB. At current volume **an hourly VM started for demo hours
beats an always-on pod by a wide margin** — a `keep_warm=2` H100 pair is ≈$5,687/month against
≈$165 of real inference. Scale-to-zero is the right shape for the upload tools
(`face_swap_hd`, `lip_sync_hd`, `talking_avatar_hd`) once those routes exist; they are priced
in `app/billing.py` today and have none, so a serverless platform bought now would serve
nothing.

`app/providers.py` detects TensorRT, CUDA, ROCm, MIGraphX, DirectML, CoreML, OpenVINO or CPU
and orders by expected throughput, always keeping a CPU path last. TensorRT is opt-in
(`EIDOMIRA_TENSORRT=1`) because it builds an engine on first use, which can take minutes.
`EIDOMIRA_PROVIDERS=cuda,cpu` overrides the order entirely.

## 4. Production configuration

| setting | why it blocks real use |
|---|---|
| `STUDIO_AUTH_SECRET` | still the shipped development default, so sessions can be forged. The server warns about this at every startup. |
| `STUDIO_PUBLIC_URL` (https) + domain + TLS | camera access requires a secure context outside localhost, and demo sign-in refuses to enable itself while this is https. |
| `STUDIO_PAYSTACK_SECRET_KEY` + plan codes | **no live Paystack call has ever been made from this repository.** |
| `STUDIO_SMTP_*` | verification emails currently print to the server log, and verifying is what activates the trial credits. |
| `STUDIO_ALLOWED_ORIGINS`, `STUDIO_EMBED_ANCESTORS` | set per deployment; the embed ancestor must match the frame that hosts the studio. |

## 5. Real material, with consent

The landing page's `.pane__empty` frames stay empty until a genuine capture exists — no
fabricated sample, no stock face. Showing a real result needs a consented before/after pair and
a real device for honest `/lab` latency numbers. This is not a purchase, but it is external and
it is the last visual piece outstanding.

## What is explicitly not needed

* **CodeFormer.** Its licence (NTU S-Lab 1.0, and Replicate's terms) forbids commercial use.
  It is popular and it stays out.
* **A licence bypass.** The non-commercial weights will not be shipped in a paid product. The
  licence is bought, or the stack ships without the neural swap and says so.
* **An always-on GPU pod**, at this volume. See §3.

## Order of operations

1. Verify the pipeline with stand-ins (§2) — free, today, and it is the evidence that the rest
   of the plan is worth paying for.
2. Fetch the two licence-clean artefacts (parser, restorer) and run the stack with them. The
   swap is still absent at this point; the compositor and the restorer are not.
3. Send the licence request (`docs/insightface-licence-request.md`) and buy the swap weights
   plus the `buffalo_l` pack.
4. Put a GPU behind it and measure real latency and real quality on real faces. Every number
   before this step is synthetic, and should be labelled as such.
5. Fill in the landing page with a consented capture.
