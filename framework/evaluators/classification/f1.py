from framework.evaluators.prf import f_beta

from .precision import compute_precision
from .recall import compute_recall


def compute_f1(results: list[dict], positive_label: str) -> float:
    return f_beta(compute_precision(results, positive_label),
                  compute_recall(results, positive_label))
