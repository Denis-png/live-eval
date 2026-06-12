from ._errant_shared import annotator


def compute_errant(results: list[dict]) -> dict:
    """
    Compute ERRANT precision, recall, and F0.5 over a batch of results.

    Each item in results must have:
        original   — ground-truth correct sentence
        corrupted  — sentence with introduced error (model input)
        prediction — model's correction attempt

    Edits are matched on (start, end, type), so this is the strict,
    position-aware variant. See `errant_dist` for the position-insensitive
    distribution-level F0.5 used in Denis's experiments.

    F0.5 weights precision twice as much as recall: in GEC, over-correcting
    fluent text is worse than missing an error.
    """
    tp = fp = fn = 0

    for item in results:
        orig = annotator.parse(item["corrupted"])
        ref  = annotator.parse(item["original"])
        pred = annotator.parse(item["prediction"])

        ref_edits  = annotator.annotate(orig, ref)
        pred_edits = annotator.annotate(orig, pred)

        ref_set  = {(e.o_start, e.o_end, e.type) for e in ref_edits}
        pred_set = {(e.o_start, e.o_end, e.type) for e in pred_edits}

        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set  - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) \
          if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f0.5":      round(f05,       4),
    }
