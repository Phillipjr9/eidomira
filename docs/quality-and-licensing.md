# Output quality: what the best tools do, and what may legally be sold

Two separate questions decide whether Eidomira's output can compete:

1. **What actually makes a swap look real?** — answered by the stack, not the model.
2. **What is allowed to be in a paid product?** — answered per asset, not per project.

Both are recorded here because the second one constrains the first, and because "the
weights are on GitHub" is not a licence.

## The gap is not the swap model

It is tempting to look for a stronger swapper. The benchmark says there is not one to
find. On the comparison in *AmazingFS* (MDPI Electronics 13/15/2986), `inswapper` already
leads the one-shot class:

| Method | Identity retrieval |
|---|---|
| **inswapper** | **93.52** |
| SimSwap | 92.25 |
| DeepFaceLab | 89.56 |
| FaceSwap | 75.61 |

inswapper also takes Attribute, Anti-Occlusion and Fidelity. InsightFace's own material
calls InSwapper-128 "the de facto standard for open-source face swapping … best quality
results".

DeepFaceLab has the highest ceiling of all, but it reaches it by training a model *per
face pair* for hours. That is unusable for an on-demand request, and it is why the
ceiling is not the comparison that matters here.

inswapper's characteristic failure is also the reason a raw swap disappoints: it **drifts
toward the target face instead of producing obvious artifacts**, so the failure is subtle
rather than visible. The measured quality difference between a raw swap and a good one is
therefore not "artifacts vs no artifacts" but "soft and slightly wrong vs sharp and
correct".

### Where the difference actually comes from

Every production pipeline converges on the same shape. This is the architecture to match:

```
detect (retinaface)  →  swap at 128 (inswapper_128 / fp16)
                     →  pixel-boost multi-pass to 256/512 (subpixel sampling)
                     →  face restoration (GPEN-BFR-512 or GFPGAN v1.4) at visibility 0.7–0.8
                     →  semantic mask preserving hair, glasses, background
                     →  LAB tone transfer / ambient lighting match
                     →  soft-mask paste and encode
```

The reason restoration is not optional: *"the swap engine, inswapper_128, outputs faces at
128x128 pixels. That's why raw swaps look soft until you stack a face restore model on
top."* Recommended values from multiple sources agree closely — `face_restore_visibility`
**0.7–0.8** ("full strength looks airbrushed; blend it back"), `codeformer_weight`
0.5–0.75, detector `retinaface_resnet50`.

One comparable open pipeline (Volvox6767/FaceSwap) is worth naming because it is almost
exactly this list, including LAB skin-tone transfer and soft-mask paste — which is
evidence that the difference between a mediocre and a good open result is the stack, not
a secret model.

## The licence table — what may be shipped

**Code and model licences must be checked independently.** InsightFace's *code* is MIT
while its *pre-trained models* are non-commercial research only; that combination is the
single most common trap in this space.

| Asset | Licence | In a paid product? |
|---|---|---|
| GFPGAN v1.4 | Apache-2.0 | **yes** |
| GPEN-BFR-512 | Apache-2.0 (code) | yes — verify the weights separately |
| CodeFormer | NTU S-Lab License 1.0 | **no** — non-commercial; the Replicate endpoint forbids commercial use outright |
| inswapper_128 | non-commercial research (code MIT) | not as-is — a commercial licence is **sold by InsightFace** |
| InsightFace Dax / Evi | private | API only, no downloadable weights |
| FaceFusion third-party assets | per-asset (OpenRAIL-AS app; AlphaFace/INSwapper/ArcFace non-commercial) | case by case, never assumed |

Consequences for Eidomira, in order:

1. **CodeFormer is not an option**, despite being the strongest restorer on paper. This is
   a licence decision, not a technical one.
2. **GFPGAN or GPEN-BFR-512** is the restoration choice.
3. **inswapper_128 needs a bought licence** before any commercial launch. That is a
   procurement step with a known vendor, not a blocker with no answer.
4. Restoration stays a **pluggable model file** rather than a hardcoded dependency, so the
   model can be swapped without touching the pipeline.

## What Eidomira does now

| Stack stage | State |
|---|---|
| Semantic mask that preserves hair/glasses/background | **done** — `app/compositor.py`, occluder classes re-applied after morphology |
| LAB tone transfer | **done** — `transfer_tone()` in `app/enhance.py` |
| Restoration at partial visibility | **done, inactive** — `FaceRestorer`, starts only when the model file exists |
| Pixel-boost to 256/512 (subpixel) | not done |
| Trained swap weights in `models/` | not present |
| inswapper commercial licence | not purchased |

Measured effect of tone transfer on the synthetic render path (25 levels of lighting
mismatch between the pasted face and the room):

| | before | after |
|---|---|---|
| colour distance from the target | 32.9 | **7.2** |
| face region vs surround | 146 vs 169 | **165 vs 169** |
| seam ratio | 1.72 | below the detection threshold |

The reverse direction is symmetric (29.2 → 5.9), and a face that already matches is left
alone — mean change **0.04** levels.

## What is measured, and what is not

The numbers above come from **synthetic composites**: a face region constructed with a
known offset and texture. They prove the code does what it claims, and they are the only
quality evidence that exists, because of the following:

- **No swap weights are in `models/`**, so **no neural swap has ever executed in this
  repository**. `create_engine()` reports `backend="diagnostic"` on purpose.
- `onnxruntime` is not installed, so **`FaceRestorer` has never run against a real
  restoration model**.
- No competitor parity is claimed. The seam ratio a person accepts, and how much
  restoration is too much, can only be established on real swapped frames.

Closing the last gap is therefore not a code problem: it needs a commercial inswapper
licence and a GFPGAN/GPEN weight file, after which the existing code path activates
without changes.
