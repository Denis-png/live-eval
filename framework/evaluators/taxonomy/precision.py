from ._taxonomy_shared import compute_taxonomy_scores


def compute_precision(results: list[dict]) -> float:
    return compute_taxonomy_scores(results)["precision"]
