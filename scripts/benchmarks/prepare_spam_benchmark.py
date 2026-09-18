"""Build the local spam benchmark: 300 rows (150 SPAM, 150 HAM) of the UCI SMS
Spam Collection, as the id,label,text CSV the spam config points at.

The rows are FIXED, not re-drawn. The file the final sweep ran on was sampled in
July 2026 by code that was not kept, and no common seeded recipe reproduces its
draw, so this rebuilds exactly that file from the position each of its rows
holds in the collection (SOURCE_ROWS, in the file's own order).

The collection comes from HuggingFace (ucirvine/sms_spam) and is re-read the way
the original sample was: as pandas parses the raw tab-separated file, quote
handling included. That matters for one row. A message that opens with a stray
double quote swallows the two lines after it, so row 238 is three messages glued
together, with the raw file's "ham<TAB>" prefixes inside its text. It is kept,
because it is part of the benchmark the sweep's spam results were measured on.

framework/data/** is gitignored, so the CSV does not ship with the repo; this
script is how it is rebuilt. It reports whether the result matches the sweep's
file byte for byte.

Usage:
    python -m scripts.benchmarks.prepare_spam_benchmark \\
        [--output framework/data/benchmarks/spam/sms_spam_ham_300.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

DATASET = "ucirvine/sms_spam"
DEFAULT_OUTPUT = "framework/data/benchmarks/spam/sms_spam_ham_300.csv"
SWEEP_MD5 = "61fa8f21664880da403ac9b4d5fb9a00"

# Row positions in the pandas parse of the collection, in benchmark order.
SOURCE_ROWS = (
    3571, 455, 609, 840, 5520, 911, 3411, 3415, 1699, 1027, 241, 2548,
    2834, 3418, 876, 1521, 2089, 1793, 1518, 4460, 80, 660, 4068, 5087,
    3502, 3219, 1513, 1545, 1017, 2540, 2169, 1706, 1741, 2024, 2331, 5068,
    910, 3961, 93, 2782, 120, 3580, 841, 5115, 5018, 4782, 401, 2090,
    3434, 1807, 1554, 5242, 2212, 1897, 4434, 1502, 1573, 1329, 3981, 5076,
    3885, 755, 4112, 2919, 2561, 5200, 1659, 870, 4086, 111, 4355, 4931,
    1765, 4737, 1339, 2332, 1896, 1527, 763, 5205, 2047, 3073, 1203, 4303,
    3790, 1077, 2515, 3217, 5377, 415, 2098, 529, 273, 564, 5008, 2497,
    123, 5028, 253, 634, 537, 5274, 3772, 2214, 5449, 3389, 319, 5020,
    885, 4371, 1069, 709, 3590, 2830, 4257, 5106, 2983, 2915, 19, 3620,
    66, 255, 250, 2664, 188, 4845, 191, 2064, 962, 35, 3074, 3187,
    1904, 268, 3222, 3298, 375, 665, 5190, 3176, 4166, 492, 4985, 2534,
    5459, 4052, 1714, 3556, 4280, 4373, 4723, 3989, 2525, 1628, 5137, 477,
    4808, 682, 1487, 5166, 1623, 3535, 4272, 2502, 3010, 2992, 5313, 4382,
    2009, 2086, 1832, 4102, 418, 3596, 2637, 672, 3736, 2394, 3906, 978,
    1106, 4967, 5530, 3220, 824, 4353, 2378, 4297, 2300, 1653, 3419, 5269,
    487, 991, 4672, 5230, 1229, 3451, 414, 515, 856, 1118, 1688, 3968,
    4347, 3994, 1430, 3356, 2014, 4258, 1368, 690, 5537, 2347, 2180, 304,
    989, 1530, 3778, 579, 554, 827, 1318, 3986, 5147, 2511, 3385, 422,
    5162, 2636, 2609, 4626, 4060, 1781, 6, 3638, 4968, 5081, 4407, 1930,
    901, 2427, 5367, 1076, 287, 1560, 3064, 1057, 1120, 2638, 2081, 5012,
    5060, 1350, 522, 1122, 4200, 583, 2804, 450, 192, 5109, 307, 765,
    4301, 3646, 5058, 2747, 3014, 1269, 3463, 4914, 2993, 1080, 1597, 2785,
    938, 3443, 823, 1780, 8, 1464, 2309, 5526, 440, 3709, 2127, 3253,
    4394, 3425, 598, 1632, 5071, 2833, 2532, 3639, 4822, 4804, 9, 1050,
)


def load_source():
    """The collection as a (label, text) frame, parsed as the original sample's
    source was. Deferred imports: only this step needs the network."""
    import io

    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    names = ds.features["label"].names
    raw = "\n".join(f"{names[label]}\t{text.rstrip(chr(10))}"
                    for text, label in zip(ds["sms"], ds["label"]))
    return pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["label", "text"])


def write_benchmark(source, output: str | Path, rows=SOURCE_ROWS) -> Counter:
    """Write `rows` of `source` (a frame with label and text columns) to output
    as id,label,text, ids from 1, labels upper-cased. Returns the label counts."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = Counter()
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "text"])
        for i, row in enumerate(rows, 1):
            label = str(source["label"].iloc[row]).upper()
            labels[label] += 1
            writer.writerow([i, label, source["text"].iloc[row]])
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"CSV to write (default {DEFAULT_OUTPUT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = write_benchmark(load_source(), args.output)
    digest = hashlib.md5(Path(args.output).read_bytes()).hexdigest()
    print("Spam benchmark preparation summary")
    print("=" * 34)
    print(f"Source            : {DATASET} ({len(SOURCE_ROWS)} fixed rows)")
    print(f"Labels            : {dict(sorted(counts.items()))}")
    print(f"Output            : {args.output}")
    print(f"Matches the sweep : {'yes' if digest == SWEEP_MD5 else 'NO -- md5 ' + digest}")


if __name__ == "__main__":
    main()
