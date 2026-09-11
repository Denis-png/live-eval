from ._taxonomy_shared import compute_taxonomy_scores


def compute_diagnostics(results: list[dict]) -> dict:
    """Every counter the scoring pass produced, minus the three headline scores.

    A dict rather than a float: the pipeline's aggregate() flattens it into
    dotted subkeys, the same way GEC's errant metric reports its parts.
    """
    return {
        key: value
        for key, value in compute_taxonomy_scores(results).items()
        if key not in {"precision", "recall", "f1"}
    }
