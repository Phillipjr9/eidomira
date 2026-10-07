"""Search for better default settings by replaying frames, and record what it measured.

This is the offline half of the trainer. It answers one question: *given these frames, which
setting makes the output measurably better?* It writes `data/tuned_defaults.json`, which
`app/config.py` reads at startup and clamps through `app/knobs.py` — so the pipeline picks
up the improvement without anyone editing a file.

What it will not do:

* **It does not train anything.** No weights are touched, and none exist in this repository.
* **It does not score a knob it cannot evaluate.** `restoration_visibility` needs a
  restoration model to re-run, so with no model installed the tool reports that it could
  not measure it rather than inventing a good value.
* **It does not trust its own inputs.** Every candidate is clamped with the same bounds the
  live trainer uses, and the written file records whether the pairs were real outputs or
  generated, so a synthetic sweep can never be mistaken for a measurement of real output.

    python -m tools.train_defaults --synthetic 6          # labelled cases, answers known
    python -m tools.train_defaults --pairs ./captures      # real original/swapped pairs
    python -m tools.train_defaults --pairs ./captures --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enhance import transfer_tone                      # noqa: E402
from app.knobs import KNOBS, TUNED_DEFAULTS_PATH           # noqa: E402
from tools.quality_report import changed_mask, colour_shift, seam_ratio  # noqa: E402

#: A face-sized region, in the frame coordinates the synthetic cases are built in.
FACE = (slice(50, 150), slice(80, 180))

#: Only recommend a change that beats the current value by this much. Both metrics used
#: here are one-sided: nothing in them penalises over-correcting a face or over-softening a
#: mask, so at the top of a curve the remaining differences are noise. Recommending a
#: default change for a 10% gain would be churn reported as improvement.
IMPROVE_MARGIN = .15

#: What each metric cannot see, stated in the output rather than left for the reader to
#: work out. A search that only knows one direction is not a recommendation to go that way.
BLIND_SPOTS = {
    "tone_transfer_strength":
        "the colour metric sees a mismatch but not a face corrected until it is no longer "
        "the person that was swapped in",
    "parser_feather":
        "the seam metric sees a hard edge but not the hair and glasses that a softer mask "
        "stops protecting",
}


@dataclass
class Pair:
    """One original/swapped pair, and where it came from."""

    label: str
    original: np.ndarray
    swapped: np.ndarray


# ───────────────────────────── inputs ─────────────────────────────

def _scene(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = np.zeros((200, 260, 3), np.int16)
    texture[::3, ::3] = 12
    texture[1::5, 2::5] = -6
    texture += rng.integers(-2, 3, texture.shape)          # a little sensor noise
    return np.clip(168 + texture, 0, 255).astype(np.uint8)


def synthetic_pairs(count: int = 1) -> list[Pair]:
    """Cases whose correct answer is known, so the search itself can be checked.

    Each pair starts from an original frame and composites a face into it from a photograph
    lit differently — the defect tone transfer exists to fix — at a known offset, plus two
    controls: one already matched, and one that is merely soft. `count` varies the frame so
    a sweep is not scored against a single lucky texture.
    """
    pairs: list[Pair] = []
    for seed in range(max(1, count)):
        original = _scene(seed)
        for label, shift, feather in (
            ("lit 25 dark", -25, 21),
            ("lit 12 dark", -12, 21),
            ("lit 18 bright", 18, 21),
            ("hard edge, lit 25 dark", -25, 0),
        ):
            incoming = np.clip(original.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            alpha = np.zeros(original.shape[:2], np.float32)
            alpha[FACE] = 1.0
            if feather:
                alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
            blended = original * (1 - alpha[..., None]) + incoming * alpha[..., None]
            pairs.append(Pair(f"{label} #{seed}", original,
                              np.rint(blended).astype(np.uint8)))
        # controls: nothing to fix, so a search that "improves" these is measuring noise
        pairs.append(Pair(f"already matched #{seed}", original, original.copy()))
        pairs.append(Pair(f"soft, no colour defect #{seed}", original,
                          cv2.GaussianBlur(original, (9, 9), 0)))
    return pairs


def pairs_from_directory(directory: Path) -> list[Pair]:
    """Read `*_original.*` / `*_swapped.*` pairs from a directory of saved outputs.

    Real pairs are the ones that matter: a setting tuned on generated frames is a guess
    about real frames, which is why the source is recorded in the output file.
    """
    found = []
    for original_path in sorted(directory.glob("*_original.*")):
        stem = original_path.name[: -len("_original" + original_path.suffix)]
        matches = list(directory.glob(f"{stem}_swapped.*"))
        if not matches:
            continue
        original = cv2.imread(str(original_path))
        swapped = cv2.imread(str(matches[0]))
        if original is None or swapped is None or original.shape != swapped.shape:
            continue
        found.append(Pair(stem, cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                          cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)))
    return found


# ───────────────────────────── scoring ─────────────────────────────

def _mask_for(pair: Pair) -> np.ndarray:
    """Where the pair differs, as a soft weight the tone transfer can fade across."""
    changed = changed_mask(pair.original, pair.swapped).astype(np.float32)
    if not changed.any():
        return changed
    size = max(3, (min(changed.shape) // 20) | 1)
    return cv2.GaussianBlur(changed, (size, size), 0)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def score_tone(pairs: list[Pair], strength: float) -> float | None:
    """Median colour distance after running tone transfer at `strength`.

    Lower is better. The region is pinned to where the pair differs before correction, so
    every candidate is scored over the same pixels; deriving it per candidate made the
    curve non-monotonic and the winner noise (0.6 scored 4.9 and 0.7 scored 9.3, which no
    smooth blend parameter can do). Pairs with nothing to compare are skipped rather than
    counted as perfect: a metric over nothing is not a zero.
    """
    distances = []
    for pair in pairs:
        region = changed_mask(pair.original, pair.swapped)
        if not region.any():
            continue
        corrected = transfer_tone(pair.original, pair.swapped, _mask_for(pair), strength)
        measured = colour_shift(pair.original, corrected, mask=region)
        if measured:
            distances.append(measured["distance"])
    return _median(distances)


def score_feather(pairs: list[Pair], feather: float) -> float | None:
    """Median seam after re-compositing the pair with `feather`, lower is better.

    An approximation, and worth being explicit about: it re-pastes the changed region with
    a Gaussian edge the width the compositor would use, rather than running the face parser
    again. It measures the *edge*, which is what feathering controls, and nothing else.
    """
    seams = []
    for pair in pairs:
        changed = changed_mask(pair.original, pair.swapped)
        if not changed.any():
            continue
        height, width = changed.shape
        blur = max(3, int(min(height, width) * feather) | 1)
        alpha = cv2.GaussianBlur(changed.astype(np.float32), (blur, blur), 0)[..., None]
        rebuilt = pair.original * (1 - alpha) + pair.swapped * alpha
        rebuilt = np.rint(np.clip(rebuilt, 0, 255)).astype(np.uint8)
        measured = seam_ratio(pair.original, rebuilt)
        if measured is not None:
            seams.append(measured)
    return _median(seams)


def sweep(pairs: list[Pair], knob_name: str, scorer, current: float) -> dict:
    """Try every candidate the bounds allow, in a stable order, and report the winner."""
    knob = KNOBS[knob_name]
    values = [round(knob.low + index * knob.step, 4)
              for index in range(int(round((knob.high - knob.low) / knob.step)) + 1)]
    # The value in use has to be scored too, or there is nothing to compare against: the
    # grid does not generally contain it (0.035 sits between 0.03 and 0.04), and comparing
    # against an assumed baseline is how a search reports an improvement it never measured.
    values = sorted({*values, round(knob.clamp(current), 4)})
    scores = {}
    for value in values:
        score = scorer(pairs, value)
        if score is not None:
            scores[value] = round(score, 4)

    if not scores:
        return {"knob": knob_name, "measurable": False,
                "reason": "no pair in the set had anything to measure for this setting"}

    best = min(scores, key=lambda value: scores[value])
    current_score = scores.get(round(current, 4))
    improvement = (round(1 - scores[best] / current_score, 4)
                   if current_score and current_score > 0 else None)
    # The margin is what separates a measurement from a preference. Without it the tool
    # writes a new default whenever its noise happens to land lower, which is the same
    # failure the live trainer reverts.
    beats_margin = bool(improvement and improvement >= IMPROVE_MARGIN
                        and best != round(current, 4))
    # A one-sided metric walks to whichever bound it cannot see past: nothing in the seam
    # metric penalises a mask so soft it stops protecting hair and glasses, so its preferred
    # setting is always the softest. At a bound the preference carries no information, so it
    # is reported and not applied.
    at_bound = best in (round(knob.low, 4), round(knob.high, 4))
    return {
        "knob": knob_name,
        "measurable": True,
        "current": current,
        "best": best,
        "recommended": beats_margin and not at_bound,
        "at_bound": at_bound,
        "current_score": current_score,
        "best_score": scores[best],
        "improvement": improvement,
        "blind_spot": BLIND_SPOTS.get(knob_name),
        "curve": scores,
    }


def evaluate(pairs: list[Pair], current: dict) -> tuple[dict, list[dict]]:
    """Sweep what can be swept, and say plainly what could not be."""
    results = [
        sweep(pairs, "tone_transfer_strength", score_tone,
              current["tone_transfer_strength"]),
        sweep(pairs, "parser_feather", score_feather, current["parser_feather"]),
        {
            "knob": "restoration_visibility",
            "measurable": False,
            "reason": "needs a restoration model to re-run, and none is installed; the live "
                      "trainer can move it against measured latency instead",
        },
    ]
    # Every knob appears, measured or explained: a knob missing from this mapping would
    # keep its default with nothing said about it.
    return {result["knob"]: result for result in results}, results


# ───────────────────────────── output ─────────────────────────────

def describe(results: list[dict], source: str, pairs: int) -> str:
    lines = [
        "Default settings, searched rather than guessed",
        "",
        f"Pairs: {pairs} ({source}).",
        "",
    ]
    for result in results:
        name = result["knob"]
        if not result["measurable"]:
            lines.append(f"{name}: not measured — {result['reason']}")
            continue
        if result["recommended"]:
            lines.append(f"{name}: {result['current']} → {result['best']}"
                         f", {result['improvement']:.0%} better")
        elif result.get("at_bound") and result["current"] != result["best"]:
            lines.append(
                f"{name}: {result['best']} scored {result['improvement']:.0%} better than "
                f"{result['current']} and is the {'upper' if result['best'] > result['current'] else 'lower'} "
                "bound, so it is reported and not applied — at a bound this metric cannot "
                "see any reason not to keep going, which means its preference there is not "
                "evidence"
            )
        elif result["current"] == result["best"]:
            lines.append(f"{name}: {result['current']} scored best already")
        else:
            gain = result.get("improvement")
            lines.append(
                f"{name}: no change recommended — {result['best']} scored "
                f"{gain:.0%} better than {result['current']}, under the "
                f"{IMPROVE_MARGIN:.0%} needed to justify changing a default"
                if gain is not None else f"{name}: not comparable, left alone")
        lines.append(f"    {name} cannot see: {result['blind_spot']}")
        curve = ", ".join(f"{value}={score}" for value, score in result["curve"].items())
        lines.append(f"    {curve}")
    lines += [
        "",
        "No model was trained: this searches settings, and the swap weights are not here.",
    ]
    if source.startswith("generated"):
        lines += [
            "These pairs are generated, not captured, so they show that the search works "
            "and not what a real swap needs. Run it again on saved output before trusting "
            "the numbers.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path,
                        help="directory of *_original.* and *_swapped.* image pairs")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="use generated cases with known defects instead")
    parser.add_argument("--out", type=Path, default=TUNED_DEFAULTS_PATH,
                        help=f"where to record the result (default {TUNED_DEFAULTS_PATH})")
    parser.add_argument("--no-write", action="store_true",
                        help="report only; leave the application's defaults alone")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    if args.pairs:
        pairs = pairs_from_directory(args.pairs)
        source = f"captured output from {args.pairs}"
        if not pairs:
            print(f"no *_original.* / *_swapped.* pairs found in {args.pairs}", file=sys.stderr)
            return 1
    else:
        pairs = synthetic_pairs(args.synthetic or 1)
        source = "generated cases with known defects"

    from app.config import settings
    current = {name: float(getattr(settings, name)) for name in KNOBS}
    by_knob, results = evaluate(pairs, current)

    best = {name: result["best"] for name, result in by_knob.items()
            if result["measurable"] and result["recommended"]}
    changed = {name: value for name, value in best.items()
               if value != round(current[name], 4)}

    if args.json:
        print(json.dumps({"source": source, "pairs": len(pairs), "values": best,
                          "changed": changed, "results": results}, indent=2, default=str))
    else:
        print(describe(results, source, len(pairs)))

    if args.no_write:
        return 0

    payload = {
        "values": best,
        "source": source,
        "pairs": len(pairs),
        "measured": {name: {"best_score": result["best_score"],
                            "at_default": result["current_score"]}
                     for name, result in by_knob.items() if result["measurable"]},
        "not_measured": {result["knob"]: result["reason"]
                         for result in results if not result["measurable"]},
        "note": "Settings only. No model is trained by this tool.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nrecorded in {args.out}"
          + (f"; {len(changed)} setting(s) differ from the current defaults" if changed
             else "; the current defaults already scored best"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
