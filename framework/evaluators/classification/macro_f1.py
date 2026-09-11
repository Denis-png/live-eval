from framework.evaluators.prf import precision_recall_f


def compute_macro_f1(results: list[dict], labels: tuple[str, ...]) -> float:
    # The standard macro-F1, sklearn's average="macro": each class's own F1,
    # averaged. It used to be the F1 of macro precision and macro recall, which
    # is never lower than this and is higher whenever classes differ in their
    # precision/recall balance: precision on one class offsets recall on
    # another. A class never gold and never predicted scores 0, as it does in
    # macro precision and recall.
    scores = []
    for cls in labels:
        tp = sum(1 for r in results if r["prediction"] == cls and r["label"] == cls)
        fp = sum(1 for r in results if r["prediction"] == cls and r["label"] != cls)
        fn = sum(1 for r in results if r["prediction"] != cls and r["label"] == cls)
        scores.append(precision_recall_f(tp, fp, fn)[2])
    return sum(scores) / len(scores) if scores else 0.0
