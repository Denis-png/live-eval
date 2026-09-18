"""Build the local GEC benchmark: one split of the FCE corpus in M2 format, from
the official BEA-2019 release (FCE v2.1), as the file the gec config points at.

The final sweep ran on the test split, fce.test.gold.bea19.m2 (2,695 sentences),
unmodified; framework.data_loading reads annotator 0's edits from it. The FCE
corpus is released for research use; see the licence in the archive.

framework/data/** is gitignored, so the file does not ship with the repo; this
script is how it is rebuilt. It reports whether the result matches the sweep's
file byte for byte.

Usage:
    python -m scripts.benchmarks.prepare_gec_benchmark \\
        [--split test] [--output framework/data/benchmarks/gec/fce.m2]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz"
DEFAULT_OUTPUT = "framework/data/benchmarks/gec/fce.m2"
SWEEP_SPLIT, SWEEP_MD5 = "test", "0e9a42192cc86d8cc6e4275338e4c4c2"


def member_for(split: str) -> str:
    return f"fce/m2/fce.{split}.gold.bea19.m2"


def extract_split(archive: str | Path, split: str, output: str | Path) -> int:
    """Copy one split's M2 file out of the release archive to output. Returns
    its sentence count."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            source = tar.extractfile(member_for(split))
        except KeyError:
            raise ValueError(f"{member_for(split)} is not in {archive}") from None
        with source, output.open("wb") as f:
            shutil.copyfileobj(source, f)
    with output.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith("S "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test",
                        help="FCE split (default test, the one the sweep ran on)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"M2 file to write (default {DEFAULT_OUTPUT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "fce.tar.gz"
        urllib.request.urlretrieve(URL, archive)
        sentences = extract_split(archive, args.split, args.output)
    digest = hashlib.md5(Path(args.output).read_bytes()).hexdigest()
    same = args.split == SWEEP_SPLIT and digest == SWEEP_MD5
    print("GEC benchmark preparation summary")
    print("=" * 33)
    print(f"Source            : FCE v2.1 (BEA-2019), {args.split} split")
    print(f"Sentences         : {sentences}")
    print(f"Output            : {args.output}")
    print(f"Matches the sweep : {'yes' if same else 'NO -- md5 ' + digest}")


if __name__ == "__main__":
    main()
