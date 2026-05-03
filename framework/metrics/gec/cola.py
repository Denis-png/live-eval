"""CoLA acceptability metric (Denis's `cola_*` columns).

Reports the fraction of *acceptable* sentences before and after correction
using the textattack/bert-base-uncased-CoLA classifier. A model that
actually fixes errors should push `corrected` above `input` (positive `delta`).
"""
import os

import torch
from transformers import pipeline as hf_pipeline

_MODEL_ID = "textattack/bert-base-uncased-CoLA"
_scorer = None


def _resolve_device() -> int:
    """Return transformers-pipeline device id (-1=CPU, 0=GPU).

    Honors FRAMEWORK_DEVICE env var (auto|cpu|cuda). In `auto`, picks GPU only
    if it's both available AND its compute capability is in this torch build's
    supported arch list — guards against old cards (e.g. sm_50) that pass
    is_available() but crash with cudaErrorNoKernelImageForDevice.
    """
    pref = os.environ.get("FRAMEWORK_DEVICE", "auto").lower()
    if pref == "cpu":
        return -1
    if pref == "cuda":
        return 0
    if not torch.cuda.is_available():
        return -1
    try:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        if not any(sm == a for a in torch.cuda.get_arch_list()):
            print(
                f"[INFO] GPU capability {sm} not in torch arch list "
                f"{torch.cuda.get_arch_list()} — falling back to CPU."
            )
            return -1
    except Exception:
        pass
    return 0


def _get_scorer():
    global _scorer
    if _scorer is None:
        device = _resolve_device()
        print(f"Loading CoLA scorer: {_MODEL_ID} (device={'cpu' if device == -1 else f'cuda:{device}'}) ...")
        _scorer = hf_pipeline("text-classification", model=_MODEL_ID, device=device)
    return _scorer


def _cola_scores(texts: list[str], batch_size: int = 32) -> list[float]:
    scorer = _get_scorer()
    scores = []
    for i in range(0, len(texts), batch_size):
        preds = scorer(texts[i:i + batch_size], truncation=True, max_length=128)
        scores.extend(1.0 if p["label"] == "LABEL_1" else 0.0 for p in preds)
    return scores


def compute_cola(results: list[dict]) -> dict:
    """Mean CoLA acceptability of inputs and predictions, plus delta."""
    if not results:
        return {"input": 0.0, "corrected": 0.0, "delta": 0.0}
    inputs      = [r["corrupted"]  for r in results]
    predictions = [r["prediction"] for r in results]
    a = sum(_cola_scores(inputs))      / len(inputs)
    c = sum(_cola_scores(predictions)) / len(predictions)
    return {
        "input":     round(a,     4),
        "corrected": round(c,     4),
        "delta":     round(c - a, 4),
    }
