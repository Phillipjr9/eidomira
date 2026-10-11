"""Serve the HTML pages with their asset URLs stamped with the asset's content.

A browser — or the service worker this app registers — is entitled to reuse a copy of
`/static/landing.js` without asking the server. That is how a fix that is live on the server
stays invisible to a user: the page arrives (the HTML is revalidated), and the script that
makes it work does not. It happened here, and it looked exactly like a feature that had never
been built: the markup carried the demo row, the script that unhides it was three commits old,
and no request for it ever reached the server.

`Cache-Control: no-cache` on `/static/` narrows the window but does not close it, because the
copy already in a cache may have been stored under the old headers, with an invented freshness
lifetime still running.

So the reference itself changes when the file changes. `?v=<digest>` makes each version of an
asset a different URL, which no cache can match to anything but that exact content. The pages
stay revalidated — this is belt and braces on purpose, because the failure mode is silent and
expensive to diagnose.

The digest is computed from the file, not written into the markup, so it cannot go stale: edit
`landing.js` and the next page load references the new URL with no version to remember.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

#: `href="/static/landing.js"`, with or without a query already present, plus `content="..."` —
#: which is how the social-preview image is referenced. Matched only when the value begins with
#: `/static/` at the quote, so an attribute that merely mentions the path in passing is left
#: alone, and a reference to another origin is not ours to stamp either.
LOCAL_ASSET = re.compile(r'((?:href|src|content)=")/static/([^"?]+)(?:\?[^"]*)?(")')

def version(relative: str) -> str | None:
    """A short digest of the file's bytes, or None when there is no such file.

    Read every time, deliberately. The obvious optimisation — remember the digest while the
    file's size and mtime are unchanged — was written first and removed, because it reintroduced
    the exact problem this module exists to prevent: a same-length rewrite within one mtime tick
    was served the previous digest, so the page handed out a URL for content that was no longer
    there. The whole set of assets a page names hashes in about a millisecond; correctness is
    worth more than that.
    """
    path = STATIC / relative
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    except OSError:
        return None


def stamp(markup: str) -> str:
    """Rewrite every local `/static/` reference in `markup` to its content-stamped URL."""
    def replace(match: re.Match) -> str:
        attribute, relative, tail = match.group(1), match.group(2), match.group(3)
        digest = version(relative)
        if digest is None:
            # A reference to a file that is not there is left exactly as it is: rewriting it
            # would invent a URL that cannot be fetched, and the 404 it produces today is the
            # useful signal.
            return match.group(0)
        return f"{attribute}/static/{relative}?v={digest}{tail}"

    return LOCAL_ASSET.sub(replace, markup)


def html(path: Path) -> HTMLResponse:
    """A page, with its asset references stamped. The file itself is never rewritten."""
    return HTMLResponse(stamp(path.read_text(encoding="utf-8")))
