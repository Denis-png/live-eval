def compute_fpr(results: list[dict], positive_label: str) -> float:
    fp = sum(1 for r in results if r["prediction"] == positive_label and r["label"] != positive_label)
    tn = sum(1 for r in results if r["prediction"] != positive_label and r["label"] != positive_label)
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0
