"""Dataset-source resolution and local benchmark file loaders (m2 / csv / tsv).

Local rows are plain dicts fed through task.parse_row(), exactly like
HuggingFace rows — everything downstream (generation, profiling, evaluation)
is source-agnostic.
"""
import csv
import os

SUPPORTED_LOCAL_FORMATS = ("m2", "csv", "tsv")

_FORMAT_BY_EXTENSION = {".m2": "m2", ".csv": "csv", ".tsv": "tsv"}


def resolve_dataset_config(ds_config: dict) -> dict:
    """Normalize the dataset block to one flat dict, so the rest of the code
    never cares which shape the user wrote.

    Preferred shape keeps per-source settings in nested blocks — switching
    source is a one-field change, nothing needs commenting out:

        dataset:
          source: local            # huggingface | local
          sample_size: 50
          huggingface: {name: ..., split: ..., streaming: false}
          local: {path: ..., format: csv}

    Legacy flat keys (name/split/streaming/hf_token at the top level,
    local_path/format for local files) are still accepted; a nested block
    wins over its flat counterpart.
    """
    ds = ds_config or {}
    hf = ds.get("huggingface") or {}
    local = ds.get("local") or {}
    return {
        "source": ds.get("source", "huggingface"),
        "sample_size": ds.get("sample_size"),
        "name": hf.get("name", ds.get("name")),
        "split": hf.get("split", ds.get("split")),
        "streaming": hf.get("streaming", ds.get("streaming", False)),
        "hf_token": hf.get("hf_token", ds.get("hf_token")),
        "path": local.get("path", ds.get("local_path")),
        "format": local.get("format", ds.get("format")),
    }


def infer_format(path: str) -> str | None:
    """Local file format from the extension, or None when unrecognizable."""
    return _FORMAT_BY_EXTENSION.get(os.path.splitext(str(path))[1].lower())


def iter_local_rows(path: str, fmt: str | None = None):
    """Yield dict rows from a local benchmark file.

    m2  → {"incorrect", "correct"} pairs (annotator 0, no-edit sentences skipped)
    csv/tsv → one dict per row keyed by the header

    Raises ValueError (not FileNotFoundError) so entry points surface a clean
    [ERROR] instead of a traceback."""
    fmt = fmt or infer_format(path)
    if fmt not in SUPPORTED_LOCAL_FORMATS:
        raise ValueError(
            f"Cannot determine local dataset format for '{path}'. Set "
            f"dataset.local.format to one of: {', '.join(SUPPORTED_LOCAL_FORMATS)}."
        )
    if not os.path.isfile(path):
        raise ValueError(
            f"Local dataset file not found: '{path}' (paths are relative to the "
            f"directory you run from — the repo root)."
        )
    if fmt == "m2":
        yield from _iter_m2_rows(path)
    else:
        yield from _iter_delimited_rows(path, delimiter="," if fmt == "csv" else "\t")


def _iter_delimited_rows(path: str, delimiter: str):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)


def _iter_m2_rows(path: str, annotator_id: int = 0):
    """Parse an .m2 file into {"incorrect", "correct"} rows.

    Ported from experiments/Denis/datasets_experiments/sampling.py::load_m2.
    Applies the selected annotator's edits right-to-left so earlier spans stay
    valid; skips sentences that annotator left unedited (no error signal)."""
    with open(path, encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    for block in blocks:
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines or not lines[0].startswith("S "):
            continue
        src_tokens = lines[0][2:].split()
        edits = []
        for line in lines[1:]:
            if not line.startswith("A "):
                continue
            parts = line[2:].split("|||")
            if len(parts) < 6:
                continue
            span = parts[0].split()
            start, end = int(span[0]), int(span[1])
            correction = parts[2]
            annotator = int(parts[5])
            if annotator == annotator_id and correction != "-NONE-":
                edits.append((start, end, correction))
        if not edits:
            continue
        target = src_tokens[:]
        for start, end, correction in sorted(edits, key=lambda e: -e[0]):
            target[start:end] = correction.split() if correction else []
        yield {"incorrect": " ".join(src_tokens), "correct": " ".join(target)}
