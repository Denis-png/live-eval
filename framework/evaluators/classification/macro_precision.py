from framework.evaluators.prf import precision


def compute_macro_precision(results: list[dict], labels: tuple[str, ...]) -> float:
    scores = []
    for cls in labels:
        tp = sum(1 for r in results if r["prediction"] == cls and r["label"] == cls)
        fp = sum(1 for r in results if r["prediction"] == cls and r["label"] != cls)
        scores.append(precision(tp, fp))
    return sum(scores) / len(scores) if scores else 0.0
