# ============================================================
# ERRANT Metric
# ============================================================
# ERRANT (ERRor ANnotation Toolkit) is the standard metric
# for GEC evaluation. It analyzes corrections at the
# linguistic level and categorizes error types.
#
# It computes:
#   - Precision: of corrections made, how many were correct
#   - Recall: of all errors, how many were found
#   - F0.5: weighted balance (precision weighted 2x more)
#
# F0.5 is used instead of F1 because in GEC, precision
# matters more than recall — it's better to make fewer
# but accurate corrections than to over-correct.
# ============================================================

import errant


# Load the ERRANT annotator once (expensive to reload each time)
annotator = errant.load("en")


def compute_errant(results: list[dict]) -> dict:
    """
    Compute ERRANT precision, recall, and F0.5.

    Args:
        results: list of {
            "original":   correct sentence (ground truth),
            "corrupted":  incorrect sentence (model input),
            "prediction": model's correction
        }

    Returns:
        {
            "precision": float,
            "recall":    float,
            "f0.5":      float
        }
    """
    tp = fp = fn = 0

    for item in results:
        # Parse sentences into spaCy Doc objects
        orig = annotator.parse(item["corrupted"])   # incorrect sentence
        ref  = annotator.parse(item["original"])    # ground truth
        pred = annotator.parse(item["prediction"])  # model output

        # Get edits (what changed between sentences)
        ref_edits  = annotator.annotate(orig, ref)   # what should change
        pred_edits = annotator.annotate(orig, pred)  # what model changed

        # Compare as sets (position + error type)
        ref_set  = {(e.o_start, e.o_end, e.type) for e in ref_edits}
        pred_set = {(e.o_start, e.o_end, e.type) for e in pred_edits}

        tp += len(pred_set & ref_set)   # correctly fixed errors
        fp += len(pred_set - ref_set)   # wrongly changed things
        fn += len(ref_set - pred_set)   # missed errors

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) \
          if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f0.5":      round(f05,       4)
    }