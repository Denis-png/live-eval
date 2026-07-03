"""Prepare a balanced real benchmark sample from the UCI SMS Spam Collection.

The script downloads the source archive, samples 150 HAM and 150 SPAM messages
with a fixed seed, shuffles the combined sample, and writes a CSV with columns:
id,label,text.
"""

from __future__ import annotations

import csv
import io
import random
import ssl
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
OUTPUT_PATH = Path("framework/data/spam/sms_spam_sample_300.csv")
SEED = 42
HAM_COUNT = 150
SPAM_COUNT = 150


def _download_archive(url: str) -> bytes:
    """Download the source ZIP archive."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            return response.read()


def _load_source_rows(archive_bytes: bytes) -> list[dict[str, str]]:
    """Read and normalize rows from the UCI SMS Spam Collection archive."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open("SMSSpamCollection") as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8", newline="")
            reader = csv.reader(text_file, delimiter="\t")
            for raw_label, text in reader:
                label = raw_label.strip().upper()
                if label not in {"HAM", "SPAM"}:
                    continue
                rows.append({"label": label, "text": text})
    return rows


def _balanced_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sample 150 HAM and 150 SPAM rows, then shuffle the combined result."""
    rng = random.Random(SEED)
    ham_rows = [row for row in rows if row["label"] == "HAM"]
    spam_rows = [row for row in rows if row["label"] == "SPAM"]

    if len(ham_rows) < HAM_COUNT or len(spam_rows) < SPAM_COUNT:
        raise ValueError(
            f"Not enough rows to sample {HAM_COUNT} HAM and {SPAM_COUNT} SPAM "
            f"(found {len(ham_rows)} HAM, {len(spam_rows)} SPAM)."
        )

    sample = rng.sample(ham_rows, HAM_COUNT) + rng.sample(spam_rows, SPAM_COUNT)
    rng.shuffle(sample)
    return [
        {"id": str(index), "label": row["label"], "text": row["text"]}
        for index, row in enumerate(sample, start=1)
    ]


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write the benchmark sample as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create the shared 300-row SMS spam benchmark sample."""
    archive_bytes = _download_archive(SOURCE_URL)
    source_rows = _load_source_rows(archive_bytes)
    sample_rows = _balanced_sample(source_rows)
    _write_csv(sample_rows, OUTPUT_PATH)

    counts = Counter(row["label"] for row in sample_rows)
    print(f"Total rows : {len(sample_rows)}")
    print(f"HAM count  : {counts.get('HAM', 0)}")
    print(f"SPAM count : {counts.get('SPAM', 0)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
