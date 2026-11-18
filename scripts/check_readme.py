"""Check README.md: image and file links exist, in-page anchors exist, and (with --quickstart) the
Quickstart block runs as written and prints what the README says it prints.

    python scripts/check_readme.py                 # links and anchors
    python scripts/check_readme.py --quickstart    # also run the Quickstart in a temporary folder
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess  # runs the README's own commands, in CI  # nosec B404
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
INSTALL = 'pip install "git+https://github.com/Tejascodz/Axon"'


def anchor(heading: str) -> str:
    """GitHub's anchor for a heading."""
    h = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return h.replace(" ", "-")


def check_links(text: str) -> list[str]:
    errors = []
    code_free = re.sub(r"```.*?```", "", text, flags=re.S)
    anchors = {anchor(h) for h in re.findall(r"^#{1,6} (.+)$", code_free, flags=re.M)}
    targets = re.findall(r"\]\(([^)\s]+)\)", code_free) + re.findall(r'(?:src|href)="([^"]+)"', code_free)
    for t in targets:
        if t.startswith(("http://", "https://", "mailto:")):
            continue
        if t.startswith("#"):
            if t[1:] not in anchors:
                errors.append(f"missing anchor {t}")
            continue
        path = os.path.join(ROOT, t.split("#")[0])
        if not os.path.exists(path):
            errors.append(f"missing file {t}")
    return errors


def quickstart(text: str) -> list[str]:
    block = re.search(r"<!-- quickstart:start -->\s*```bash\n(.*?)```", text, flags=re.S).group(1)
    assert INSTALL in block, "the install line changed: update INSTALL here"
    script = block.replace(INSTALL, f'pip install "{ROOT}"')
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp, capture_output=True,  # nosec B603 B607
                             text=True, timeout=3600)
    print(run.stdout)
    if run.returncode:
        print(run.stderr[-4000:], file=sys.stderr)
        return [f"quickstart exited with {run.returncode}"]
    expected = re.search(r"Expected output[^\n]*\n\n```text\n(.*?)```", text, flags=re.S).group(1)
    want = [ln for ln in expected.splitlines() if ln.startswith("[")]
    got = [ln for ln in run.stdout.splitlines() if ln.startswith("[")]
    return [] if want == got else [f"quickstart results differ:\nREADME: {want}\nactual: {got}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quickstart", action="store_true")
    a = ap.parse_args()
    with open(README, encoding="utf-8") as f:
        text = f.read()
    errors = check_links(text) + (quickstart(text) if a.quickstart else [])
    for e in errors:
        print("ERROR", e)
    print("README check:", "failed" if errors else "ok")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
