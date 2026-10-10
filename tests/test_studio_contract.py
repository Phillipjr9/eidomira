"""Static integrity checks for the private studio page.

`app.js` is the working studio (camera, WebRTC, recording, calls) and it reaches into the
DOM by id. A restyle that drops or renames one of those ids breaks a feature silently —
nothing throws until a visitor presses the button. These tests pin that contract.

They also pin the structural hooks that `cleanMode` (the OBS full-screen output) depends
on, because that behaviour is pure CSS driven by a single class on <body>.
"""
import re
from html.parser import HTMLParser
from pathlib import Path

from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
APP_HTML = (STATIC / "app.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
STUDIO_CSS = (STATIC / "studio.css").read_text(encoding="utf-8")
TOKENS_CSS = (STATIC / "tokens.css").read_text(encoding="utf-8")

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def _ids() -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', APP_HTML))


def test_studio_markup_is_balanced():
    class Balance(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack: list[tuple[str, int]] = []
            self.errors: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in VOID_ELEMENTS:
                self.stack.append((tag, self.getpos()[0]))

        def handle_endtag(self, tag):
            if tag in VOID_ELEMENTS:
                return
            if not self.stack:
                self.errors.append(f"stray </{tag}> at line {self.getpos()[0]}")
                return
            if self.stack[-1][0] != tag:
                self.errors.append(
                    f"</{tag}> at line {self.getpos()[0]} closes "
                    f"<{self.stack[-1][0]}> opened at line {self.stack[-1][1]}"
                )
                return
            self.stack.pop()

    parser = Balance()
    parser.feed(APP_HTML)
    assert not parser.errors, parser.errors
    assert not parser.stack, [tag for tag, _ in parser.stack]


def test_studio_has_no_duplicate_ids():
    ids = re.findall(r'\bid="([^"]+)"', APP_HTML)
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    assert not duplicates, f"duplicate ids: {duplicates}"


def test_every_element_app_js_queries_still_exists():
    """The whole studio contract: any id app.js looks up must be in the markup."""
    needed = set(re.findall(r"\$\('([\w-]+)'\)", APP_JS))
    needed |= set(re.findall(r"getElementById\('([\w-]+)'\)", APP_JS))
    assert len(needed) > 50, "app.js should be querying the full workbench"
    missing = sorted(needed - _ids())
    assert not missing, f"app.js queries ids missing from app.html: {missing}"


def test_clean_mode_hooks_survive():
    """OBS clean output promotes .stage.output and hides the camera stage."""
    assert "body.cleanMode" in STUDIO_CSS
    for hook in (".stage.output", ".viewport", ".stage.output .outputActions"):
        assert hook in STUDIO_CSS, f"clean-mode selector missing: {hook}"

    # the camera stage must remain the first child, or clean mode would hide the output
    canvases = APP_HTML.split('class="canvases"', 1)[1]
    first_stage = canvases.split("<div class=\"stage", 1)[1][:40]
    assert "output" not in first_stage, f"output stage must not be first: {first_stage!r}"


def test_style_hooks_used_by_app_js_are_styled():
    """app.js toggles these class names; each needs a real rule, not just a mention.

    Matching an opening brace keeps the check honest: deleting the rule body while other
    rules still reference the class name would otherwise pass.
    """
    for hook in ("hasImage", "recording", "active"):
        assert re.search(rf"\.{hook}\b[^{{}}]*\{{", STUDIO_CSS), (
            f"app.js toggles .{hook} but studio.css has no rule for it"
        )


def test_studio_uses_the_shared_token_layer():
    links = re.findall(r'<link rel="stylesheet" href="([^"]+)"', APP_HTML)
    assert "/static/tokens.css" in links, links
    assert "/static/studio.css" in links, links
    assert links.index("/static/tokens.css") < links.index("/static/studio.css"), links
    # the old monolith is gone
    assert "/static/style.css" not in links, links
    assert not (STATIC / "style.css").exists(), "style.css should have been removed"


def test_no_page_references_the_removed_stylesheet():
    for page in ("index.html", "app.html", "sw.js"):
        text = (STATIC / page).read_text(encoding="utf-8")
        assert "/static/style.css" not in text, f"{page} still references style.css"


def test_shared_layer_does_not_redefine_page_styles():
    """tokens.css owns the tokens; page sheets must consume them via var(), not hardcode."""
    assert ":root" in TOKENS_CSS
    for token in ("--paper", "--ink-800", "--violet", "--cyan", "--ease", "--r-lg"):
        assert token in TOKENS_CSS, f"{token} missing from the token layer"
    # a page sheet redeclaring :root would let the two surfaces drift
    assert ":root" not in STUDIO_CSS, "studio.css must not redeclare :root tokens"


def test_studio_avoids_the_landing_only_class_names():
    """`body.studio` and `.studio` would collide: one is a class on <body>, the other was
    the two-column video grid. The page class is `workspace` for exactly that reason."""
    assert 'class="workspace"' in APP_HTML
    assert "body.workspace" in STUDIO_CSS


def test_studio_css_parses():
    import pytest

    tinycss2 = pytest.importorskip("tinycss2", reason="tinycss2 is an optional dev dependency")
    rules = tinycss2.parse_stylesheet(STUDIO_CSS, skip_comments=True, skip_whitespace=True)
    assert not [r for r in rules if r.type == "error"]
