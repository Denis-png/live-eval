from framework.evaluators.prf import f_beta

from .macro_precision import compute_macro_precision
from .macro_recall import compute_macro_recall


def compute_macro_f1(results: list[dict], labels: tuple[str, ...]) -> float:
    # Macro-F1 here is F1 of the macro-averaged precision and recall, not the
    # mean of per-class F1s -- which is why f_beta takes p and r, not counts.
    return f_beta(compute_macro_precision(results, labels),
                  compute_macro_recall(results, labels))
