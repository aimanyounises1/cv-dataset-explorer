#!/usr/bin/env python3
"""Fail when a relative link or image path in the Markdown points nowhere.

The defect class this catches is small and specific: a reviewer clicking a path
that moved. It deliberately checks nothing else. External URLs are not fetched
-- a network flake is not a repository defect -- and heading anchors are not
resolved, because the slug rules belong to the renderer and guessing them here
would produce failures that mean nothing.

    python scripts/check_links.py   # from the repository root
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
EXTERNAL = ("http://", "https://", "mailto:", "#")
# Generated data, dependency trees, and hidden tool directories are not
# repository documentation; checking them would make the result depend on one
# developer's local tooling rather than on the submitted tree. .github is the
# one hidden directory whose Markdown faces reviewers, so it stays in scope.
SKIP_DIRS = {"node_modules", "data", "dist"}


def _skipped(parts: tuple[str, ...]) -> bool:
    return any(
        part in SKIP_DIRS or (part.startswith(".") and part != ".github")
        for part in parts
    )


def main() -> int:
    broken: list[str] = []
    checked = 0
    for md in sorted(ROOT.rglob("*.md")):
        if _skipped(md.parts):
            continue
        for target in LINK.findall(md.read_text(encoding="utf-8")):
            if target.startswith(EXTERNAL):
                continue
            checked += 1
            # Relative to the file that contains the link, as the renderer
            # resolves it -- not to the repository root.
            if not (md.parent / target.split("#")[0]).exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    print(f"checked {checked} relative Markdown links")
    for item in broken:
        print(f"BROKEN: {item}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
