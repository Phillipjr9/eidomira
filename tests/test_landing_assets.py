"""Static integrity checks for the public landing page.

These guard the failure modes that are invisible in a diff but fatal in a browser:
a script referencing an element that no longer exists, an anchor that scrolls nowhere,
a referenced asset that was never committed, or CSS that no longer parses.
"""
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")
LANDING_JS = (STATIC / "landing.js").read_text(encoding="utf-8")

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class _Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, tuple[int, int]]] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_ELEMENTS:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in VOID_ELEMENTS:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}> at line {self.getpos()[0]}")
            return
        if self.stack[-1][0] != tag:
            self.errors.append(
                f"</{tag}> at line {self.getpos()[0]} closes <{self.stack[-1][0]}> "
                f"opened at line {self.stack[-1][1][0]}"
            )
            return
        self.stack.pop()


def _ids() -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', INDEX))


def test_markup_is_balanced_and_ids_are_unique():
    parser = _Balance()
    parser.feed(INDEX)
    assert not parser.errors, parser.errors
    assert not parser.stack, [tag for tag, _ in parser.stack]

    ids = re.findall(r'\bid="([^"]+)"', INDEX)
    duplicates = {value for value in ids if ids.count(value) > 1}
    assert not duplicates, f"duplicate ids: {duplicates}"


def test_every_element_the_landing_script_queries_exists():
    ids = _ids()
    queried = set(re.findall(r'\$\("#([\w-]+)"\)', LANDING_JS))
    queried |= set(re.findall(r'getElementById\("([\w-]+)"\)', LANDING_JS))
    assert not (queried - ids), f"script queries missing ids: {sorted(queried - ids)}"


def test_every_in_page_anchor_has_a_target():
    ids = _ids()
    anchors = set(re.findall(r'href="#([\w-]+)"', INDEX))
    assert not (anchors - ids), f"anchors with no target: {sorted(anchors - ids)}"


def test_aria_references_resolve():
    ids = _ids()
    for attribute in ("aria-controls", "aria-labelledby", "aria-describedby"):
        for value in re.findall(rf'{attribute}="([^"]+)"', INDEX):
            for token in value.split():
                assert token in ids, f"{attribute} points at unknown id: {token}"


def test_every_referenced_static_asset_is_committed():
    assets = set(re.findall(r'(?:src|href)="(/static/[^"]+)"', INDEX))
    assert assets, "landing page references no static assets"
    missing = sorted(a for a in assets if not (ROOT / a.lstrip("/")).exists())
    assert not missing, f"referenced but missing on disk: {missing}"


def test_landing_css_parses_without_errors():
    tinycss2 = pytest.importorskip("tinycss2", reason="tinycss2 is an optional dev dependency")
    css = (STATIC / "landing.css").read_text(encoding="utf-8")
    rules = tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)
    assert not [r for r in rules if r.type == "error"]

    declaration_errors: list[tuple[str, str]] = []

    def walk(nodes):
        for node in nodes:
            if node.type == "qualified-rule":
                selector = tinycss2.serialize(node.prelude).strip()
                for declaration in tinycss2.parse_declaration_list(
                    node.content, skip_whitespace=True, skip_comments=True
                ):
                    if declaration.type == "error":
                        declaration_errors.append((selector, declaration.message))
                    elif declaration.type == "declaration" and declaration.value:
                        for token in declaration.value:
                            if token.type == "error":
                                declaration_errors.append((declaration.name, token.message))
            elif node.type == "at-rule" and node.content:
                walk(tinycss2.parse_rule_list(node.content, skip_whitespace=True, skip_comments=True))

    walk(rules)
    assert not declaration_errors, declaration_errors


def test_landmark_model_route_never_mounts_the_whole_models_directory():
    """The demo needs one model file; the directory also holds licensed weights."""
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert '"/models/face_landmarker.task"' in source
    assert 'mount("/models"' not in source
    assert "StaticFiles(directory=ROOT / \"models\")" not in source


def test_demo_engine_is_version_pinned():
    """An unpinned CDN URL would let the tracking demo break without a commit."""
    urls = re.findall(r'https://cdn\.jsdelivr\.net/[^"\']+', LANDING_JS)
    assert urls, "no pinned engine URL found"
    for url in urls:
        assert re.search(r'@\d+\.\d+\.\d+', url), f"unpinned engine URL: {url}"
