def compute_macro_precision(results: list[dict], labels: tuple[str, ...]) -> float:
    scores = []
    for cls in labels:
        tp = sum(1 for r in results if r["prediction"] == cls and r["label"] == cls)
        fp = sum(1 for r in results if r["prediction"] == cls and r["label"] != cls)
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0
