from ._taxonomy_shared import compute_taxonomy_scores


def compute_recall(results: list[dict]) -> float:
    return compute_taxonomy_scores(results)["recall"]
