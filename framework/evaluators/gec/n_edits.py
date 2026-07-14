"""Average number of ERRANT edits between input and prediction."""
from ._errant_shared import annotate_results


def compute_n_edits(results: list[dict]) -> float:
    if not results:
        return 0.0
    counts = [len(a["pred_edits"]) for a in annotate_results(results)]
    return round(sum(counts) / len(counts), 4)
