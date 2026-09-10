from framework.evaluators.prf import precision


def compute_precision(results: list[dict], positive_label: str) -> float:
    tp = sum(1 for r in results if r["prediction"] == positive_label and r["label"] == positive_label)
    fp = sum(1 for r in results if r["prediction"] == positive_label and r["label"] != positive_label)
    return precision(tp, fp)
