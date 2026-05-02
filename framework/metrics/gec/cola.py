"""CoLA acceptability metric (Denis's `cola_*` columns).

Reports the fraction of *acceptable* sentences before and after correction
using the textattack/bert-base-uncased-CoLA classifier. A model that
actually fixes errors should push `corrected` above `input` (positive `delta`).
"""
import torch
from transformers import pipeline as hf_pipeline

_MODEL_ID = "textattack/bert-base-uncased-CoLA"
_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading CoLA scorer: {_MODEL_ID} ...")
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
