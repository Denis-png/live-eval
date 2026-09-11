from ._taxonomy_shared import compute_taxonomy_scores


def compute_f1(results: list[dict]) -> float:
    return compute_taxonomy_scores(results)["f1"]
