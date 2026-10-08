"""Fetch the three neural artefacts the swap stack wants, and refuse the ones it may not have.

Run this on a host that can reach the internet — this sandbox cannot reach Hugging Face, so
the URLs here are **unverified from inside the repository** and every fetch reports what it
actually received (bytes, sha256) so a wrong source is visible immediately rather than at
inference time.

    python -m tools.fetch_models --check          what is present, and what each one is
    python -m tools.fetch_models --print-urls     where each comes from
    python -m tools.fetch_models --fetch restorer free artefact, no licence question
    python -m tools.fetch_models --fetch swap --accept-licence   non-commercial: asks first

The licence column is the point of the exercise. `inswapper_128.onnx` is the de facto standard
swap model and its **pre-trained weights are non-commercial** (the code is MIT; the weights
sold by insightface are not the same thing). Shipping it in a paid product needs a purchase,
and this tool will not silently download it: `--accept-licence` states that the operator has
the right to use it, and nothing else does.

Hashes are recorded in `models/models.lock.json` on the first successful fetch and verified on
every one after that. The first fetch is trust-on-first-use — there is no way to know the
canonical digest without having fetched it once — but a later fetch of the same URL that
returns different bytes is a hard failure, which is the case worth catching.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DIRECTORY = Path("models")
DEFAULT_LOCK = Path("models/models.lock.json")

#: Hugging Face is the usual host for all three; the FaceFusion asset repositories are the
#: most likely to stay up for the parser and the restorer.
HUGGING_FACE = "https://huggingface.co"


@dataclasses.dataclass(frozen=True)
class Artefact:
    role: str
    filename: str
    setting: str
    licence: str
    commercial: bool
    urls: tuple[str, ...]
    note: str = ""

    @property
    def hugging_face(self) -> bool:
        return any(url.startswith(HUGGING_FACE) for url in self.urls)


ARTEFACTS: tuple[Artefact, ...] = (
    Artefact(
        role="swap",
        filename="inswapper_128.onnx",
        setting="STUDIO_MODEL_PATH",
        licence="insightface pre-trained models — non-commercial (code is MIT)",
        commercial=False,
        urls=(
            f"{HUGGING_FACE}/xingren23/comfyflow-models/resolve/main/insightface/inswapper_128.onnx",
            f"{HUGGING_FACE}/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        ),
        note=("the one artefact that cannot be shipped without buying a licence; ~554 MB. "
              "Buy from insightface, then point STUDIO_MODEL_PATH at whatever they deliver."),
    ),
    Artefact(
        role="parser",
        filename="face_parser.onnx",
        setting="STUDIO_PARSER_MODEL_PATH",
        licence="BiSeNet face parsing, as redistributed by FaceFusion — OpenRAIL-AS",
        commercial=True,
        urls=(
            f"{HUGGING_FACE}/facefusion/models-3.3.0/resolve/main/face_parser.onnx",
            f"{HUGGING_FACE}/facefusion/models-3.0.0/resolve/main/face_parser.onnx",
        ),
        note=("OpenRAIL-AS permits commercial use with use-restrictions. The 19-class layout "
              "is what app/compositor.py's class sets are written against."),
    ),
    Artefact(
        role="restorer",
        filename="gfpgan_1.4.onnx",
        setting="STUDIO_RESTORATION_MODEL_PATH",
        licence="GFPGAN v1.4 — Apache-2.0 (the ONNX conversion is third-party)",
        commercial=True,
        urls=(
            f"{HUGGING_FACE}/facefusion/models-3.3.0/resolve/main/gfpgan_1.4.onnx",
            f"{HUGGING_FACE}/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
        ),
        note="the stage that removes a 128px swap's softness; ~340 MB.",
    ),
)


def by_role(role: str) -> Artefact:
    for artefact in ARTEFACTS:
        if artefact.role == role:
            return artefact
    raise KeyError(f"unknown role {role!r}; try one of {[a.role for a in ARTEFACTS]}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lock(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_lock(path: Path, lock: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download(url: str, destination: Path, timeout: float = 120.0,
             opener=urllib.request.urlopen) -> tuple[int, str]:
    """Stream `url` to `destination`, returning (bytes, sha256).

    Written to a `.part` file first: a download interrupted by a flaky connection must not
    leave a truncated model in place, because a truncated ONNX file fails at load with a
    parse error that says nothing about the cause.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        with opener(url, timeout=timeout) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    except BaseException:
        # A connection that dies mid-body must not leave a truncated model behind: it fails
        # later at load time with a parse error that says nothing about the cause, and it
        # looks exactly like a model that is present and broken.
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return written, digest.hexdigest()


def plan(directory: Path, lock: dict,
         artefacts: tuple[Artefact, ...] | None = None) -> list[tuple[Artefact, str, str]]:
    """What is present, what its hash is, and whether it matches what was recorded."""
    rows = []
    for artefact in (artefacts if artefacts is not None else ARTEFACTS):
        path = directory / artefact.filename
        if not path.is_file():
            rows.append((artefact, "missing", ""))
            continue
        digest = sha256_of(path)
        recorded = (lock.get(artefact.filename) or {}).get("sha256")
        if recorded is None:
            state = "present (unrecorded)"
        elif recorded == digest:
            state = "present"
        else:
            state = "PRESENT BUT DIFFERENT FROM THE LOCK"
        rows.append((artefact, state, digest[:16]))
        del recorded
    return rows


def fetch(artefact: Artefact, directory: Path, lock_path: Path, accept_licence: bool,
          timeout: float = 120.0, opener=urllib.request.urlopen) -> int:
    if not artefact.commercial and not accept_licence:
        print(f"refusing to fetch {artefact.filename}: {artefact.licence}", file=sys.stderr)
        print("  shipping these weights in a paid product needs a licence from insightface.",
              file=sys.stderr)
        print("  if you hold one, re-run with --accept-licence.", file=sys.stderr)
        return 2

    lock = read_lock(lock_path)
    recorded = (lock.get(artefact.filename) or {}).get("sha256")
    destination = directory / artefact.filename
    failures = []

    for url in artefact.urls:
        print(f"fetching {artefact.filename} from {url}")
        try:
            written, digest = download(url, destination, timeout=timeout, opener=opener)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as error:
            print(f"  failed: {type(error).__name__}: {error}")
            failures.append(f"{url}: {error}")
            continue
        print(f"  got {written / 1e6:.1f} MB, sha256 {digest}")
        if recorded and digest != recorded:
            print("  REFUSING: this is not the file recorded in the lock.", file=sys.stderr)
            print(f"    recorded {recorded}", file=sys.stderr)
            print("    received " + digest, file=sys.stderr)
            destination.unlink(missing_ok=True)
            return 3
        lock[artefact.filename] = {
            "sha256": digest, "url": url, "bytes": written,
            "licence": artefact.licence, "commercial": artefact.commercial,
            "fetched_at": int(time.time()),
        }
        write_lock(lock_path, lock)
        print(f"  recorded in {lock_path}")
        return 0

    print(f"every source failed for {artefact.filename}", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    return 1


def report(directory: Path, lock_path: Path,
           artefacts: tuple[Artefact, ...] | None = None) -> int:
    rows = plan(directory, read_lock(lock_path), artefacts)
    width = max(len(a.role) for a, _, _ in rows)
    for artefact, state, digest in rows:
        marker = "commercial ok" if artefact.commercial else "NON-COMMERCIAL"
        print(f"{artefact.role:<{width}}  {state:<34} {marker}")
        print(f"{'':<{width}}  {artefact.licence}")
        if digest:
            print(f"{'':<{width}}  sha256 {digest}")
    if any(state.startswith("PRESENT BUT") for _, state, _ in rows):
        print("\nthe lock and the file disagree: delete the file and fetch again.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="report what is present")
    actions.add_argument("--print-urls", action="store_true", help="where each comes from")
    actions.add_argument("--fetch", metavar="ROLE", help="fetch one artefact")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--accept-licence", action="store_true",
                        help="state that you hold a licence for non-commercial weights")
    options = parser.parse_args(argv)

    if options.print_urls:
        for artefact in ARTEFACTS:
            print(f"{artefact.role}: {artefact.setting}={options.directory / artefact.filename}")
            for url in artefact.urls:
                print(f"  {url}")
        return 0
    if options.fetch:
        try:
            artefact = by_role(options.fetch)
        except KeyError as error:
            print(error, file=sys.stderr)
            return 2
        return fetch(artefact, options.directory, options.lock, options.accept_licence)
    return report(options.directory, options.lock)


if __name__ == "__main__":
    raise SystemExit(main())
