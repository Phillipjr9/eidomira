# Eidomira model-training program

The production model must not be trained from scraped or unlicensed faces.

## Required stages

1. **Dataset acquisition** — consenting performers, licensed datasets, and synthetic identities.
2. **Consent ledger** — every real subject has a signed grant, permitted uses, jurisdiction, expiry, and revocation status.
3. **Identity isolation** — subject-disjoint train, validation, and test splits.
4. **Quality filtering** — blur, exposure, pose, occlusion, duplicate, and corruption checks.
5. **Privacy review** — minors excluded unless a separately reviewed program explicitly permits them.
6. **Training** — identity encoder, attribute encoder, generator, discriminator, segmentation, and temporal objectives.
7. **Evaluation** — identity similarity, expression transfer, pose, flicker, occlusion, demographic slices, latency, and human review.
8. **Release gate** — model card, license inventory, red-team report, benchmark report, and signed approval.

## Budget reality

A proprietary one-shot model competitive with established products requires a separate one-time compute and data budget beyond the under-$100 monthly serving prototype. Start with small experiments, but do not label them production quality until they pass the benchmark and red-team gates.

See `dataset_manifest.schema.json` for the consent-first record format.
