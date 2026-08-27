def compute_macro_recall(results: list[dict], labels: tuple[str, ...]) -> float:
    scores = []
    for cls in labels:
        tp = sum(1 for r in results if r["prediction"] == cls and r["label"] == cls)
        fn = sum(1 for r in results if r["prediction"] != cls and r["label"] == cls)
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0
