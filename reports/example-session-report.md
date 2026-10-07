# Session report — session

Watched 45 frames over 3 seconds and measured 44 of them.
Skipped 1 measurements because the frame was already over budget.

## What went wrong

- **colour_mismatch** (warning, first at frame 1, 44x) — the swapped face is 40 levels from the colours around it, which reads as a face pasted from another room

## What it changed

- `tone_transfer_strength` 0.3 → 0.4 (colour 40.2 → 36.2) because colour mismatch measured at 40.2
- `tone_transfer_strength` 0.4 → 0.5 (colour 36.2 → 32.3) because colour mismatch measured at 36.2
- `tone_transfer_strength` 0.5 → 0.6 (colour 32.3 → 28.4) because colour mismatch measured at 32.3
- `tone_transfer_strength` 0.6 → 0.7 (colour 28.4 → 24.4) because colour mismatch measured at 28.4

## What it tried and undid

- `tone_transfer_strength` 0.7 → 0.8 was reverted (colour 24.4 → 25.2): no improvement
- `tone_transfer_strength` 0.7 → 0.8 was reverted (colour 24.4 → 25.2): no improvement

## Settings locked for the session

These were moved and reverted, or are already at their bound, so the trainer stopped trying rather than oscillate:

- `tone_transfer_strength` — past 1.0 the correction stops matching the target's lighting and starts replacing the swapped face with it

## What this report is not

The trainer tunes settings. It cannot retrain the swap model: there are no swap weights in `models/` and no accelerator in this loop, so nothing here improves the network itself.