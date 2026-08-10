from .macro_precision import compute_macro_precision
from .macro_recall import compute_macro_recall


def compute_macro_f1(results: list[dict], labels: tuple[str, ...]) -> float:
    p = compute_macro_precision(results, labels)
    r = compute_macro_recall(results, labels)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
