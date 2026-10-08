# Draft: commercial licence request to insightface

This is the one purchase that unblocks the neural swap. Send it from a company address if one
exists, and fill in every `[…]`. Nothing here should be sent with a placeholder left in.

The three questions that decide everything are at the bottom — **hosted use vs redistribution**,
**price model**, and **their acceptable-use constraints** — and they are asked as plainly as
possible, because a licensing conversation that goes wrong usually goes wrong at the first
ambiguity, not the last.

---

**To:** (insightface's published contact — currently via their GitHub organisation or the
contact address on insightface.ai; check before sending, it changes)

**Subject:** Commercial licence for inswapper_128 and the buffalo_l model pack — hosted web
application

Hello,

I am [name], [role] at [company / legal entity], based in [country]. We are building
[product name] at [domain], a web application that performs real-time face swapping for
consented users — the user provides their own face, passes a liveness check, and the result is
generated live in their browser session. It is a commercial product: users buy credits or a
monthly plan through Paystack.

We would like to licence two things for use in production:

1. **`inswapper_128.onnx`** — the InSwapper-128 face swap model.
2. **The `buffalo_l` model pack** — detection, 2D landmarks and ArcFace recognition
   embeddings. Our engine's alignment and the swap adapter both depend on it.

We understand that the *code* in the insightface repository is MIT-licensed but that the
**pre-trained models are not**, and that a commercial licence is required for what we intend.
That is the reason for this email: we want to do this properly rather than take the weights
from a mirror and hope.

### How we intend to use them

* **Server-side inference only.** The models run on our infrastructure. They are **not**
  redistributed, bundled into a client, or exposed as downloadable files. Users interact with
  a web page; no user ever receives the model.
* **Consent-first.** Every session requires the user's own face and an explicit liveness step
  before any swap runs. We do not offer celebrity face packs, and we do not process third
  parties' images.
* **Scale.** We are at [current monthly active users / sessions], and we would be deploying on
  [GPU provider, e.g. one A10-class GPU with scale-to-zero]. We would rather report a real
  number now than be surprised by a term later.

### What we would like to know

1. **Availability and scope** — do you licence `inswapper_128` and `buffalo_l` for commercial
   hosted use, and does the licence cover both?
2. **Price model** — is it a one-off, annual, per-deployment, or revenue-share arrangement?
   Indicative pricing at the scale above, please.
3. **Hosted use vs redistribution** — does running the models server-side, without ever
   delivering them to a browser, fall under your standard terms, or is there a separate
   hosting agreement for it? We want this written down, not assumed.
4. **Acceptable-use conditions** — your licence terms and any acceptable-use policy that would
   apply to a consumer face-swap product. We would rather design to them now: if there are
   restrictions we should reflect in the product (consent flows, prohibited source images, a
   watermark, logging), we want to build them in from the start.
5. **Attribution and marking** — what you require in our terms of service, and whether you
   require any technical marking or watermark on generated output.
6. **If licensing is not available** — is the intended path instead your API (the Dax / Evi
   products)? We would consider that route, though it changes our architecture.

We can provide any of the following if it helps: use-case detail, screenshots, our consent and
liveness design, expected volume, or a call.

Thank you for your time.

[Name]
[Title], [company]
[email] · [domain]

---

## Notes for whoever sends this

* **The distinction that matters most is hosted use vs redistribution.** A licence that permits
  *distributing* weights to users is a different (and more expensive) thing than running them
  on your own servers. Answer their questions in those terms and do not let the two blur.
* **Do not offer a technical detail you cannot implement.** If they require a watermark or
  output marking, say you will build it and then build it — the studio already has a
  diagnostics panel that could carry a marker.
* **Keep the acknowledgement.** Whatever they reply with — including "no" — record it in
  `docs/` and keep the weights out of `models/` until the terms are signed. A file in the
  repository with an unclear provenance is the exact problem this document exists to avoid.
* **Do not use the mirrors in the meantime for anything user-facing.** Testing a pipeline
  locally against a mirror is one thing; serving a paid product from unlicensed weights is
  another, and `tools/fetch_models.py` refuses to do the latter without an explicit flag.
