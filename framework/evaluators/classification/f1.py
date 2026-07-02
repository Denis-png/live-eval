from .precision import compute_precision
from .recall import compute_recall


def compute_f1(results: list[dict], positive_label: str) -> float:
    p = compute_precision(results, positive_label)
    r = compute_recall(results, positive_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
