from framework.evaluators.prf import recall


def compute_macro_recall(results: list[dict], labels: tuple[str, ...]) -> float:
    scores = []
    for cls in labels:
        tp = sum(1 for r in results if r["prediction"] == cls and r["label"] == cls)
        fn = sum(1 for r in results if r["prediction"] != cls and r["label"] == cls)
        scores.append(recall(tp, fn))
    return sum(scores) / len(scores) if scores else 0.0
