"""Download the sample corpus at runtime: the English Kubernetes docs (CC BY 4.0) from
github.com/kubernetes/website. Nothing is committed to this repo.

    python -m ingest.fetch --dest data/corpus --sections concepts tasks/configure-pod-container
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # noqa: S404
import tempfile
from pathlib import Path

REPO = "https://github.com/kubernetes/website.git"
DEFAULT_REF = "main"
DEFAULT_SECTIONS = ["concepts"]


def fetch(dest: Path, sections: list[str], ref: str = DEFAULT_REF) -> Path:
    dest = dest.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "website"
        git = shutil.which("git")
        if git is None:
            raise SystemExit("git is required to fetch the corpus")
        subprocess.run(  # noqa: S603
            [
                git,
                "clone",
                "--depth=1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                ref,
                REPO,
                str(work),
            ],
            check=True,
        )
        paths = [f"content/en/docs/{s.strip('/')}" for s in sections]
        subprocess.run([git, "-C", str(work), "sparse-checkout", "set", *paths], check=True)  # noqa: S603
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(work / "content/en/docs", dest)
    count = sum(1 for _ in dest.rglob("*.md"))
    print(f"fetched {count} markdown files into {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("data/corpus"))
    ap.add_argument("--sections", nargs="+", default=DEFAULT_SECTIONS)
    ap.add_argument("--ref", default=DEFAULT_REF)
    args = ap.parse_args()
    fetch(args.dest, args.sections, args.ref)


if __name__ == "__main__":
    main()
