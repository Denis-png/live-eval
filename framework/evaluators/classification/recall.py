from framework.evaluators.prf import recall


def compute_recall(results: list[dict], positive_label: str) -> float:
    tp = sum(1 for r in results if r["prediction"] == positive_label and r["label"] == positive_label)
    fn = sum(1 for r in results if r["prediction"] != positive_label and r["label"] == positive_label)
    return recall(tp, fn)
