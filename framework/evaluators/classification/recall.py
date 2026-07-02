def compute_recall(results: list[dict], positive_label: str) -> float:
    tp = sum(1 for r in results if r["prediction"] == positive_label and r["label"] == positive_label)
    fn = sum(1 for r in results if r["prediction"] != positive_label and r["label"] == positive_label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
