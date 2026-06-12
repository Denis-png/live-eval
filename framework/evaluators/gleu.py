# ============================================================
# GLEU Metric
# ============================================================
# GLEU (Generalized Language Evaluation Understanding) is a
# variant of BLEU adapted for GEC tasks.
#
# It measures how similar the model's correction is to the
# reference (ground truth) sentence.
#
# Score range: 0 to 1
#   - 1.0 = perfect correction
#   - 0.0 = completely wrong
# ============================================================

from nltk.translate.gleu_score import sentence_gleu


def compute_gleu(results: list[dict]) -> float:
    """
    Compute average GLEU score across all samples.

    Args:
        results: list of {
            "original":   correct sentence (ground truth),
            "corrupted":  incorrect sentence (model input),
            "prediction": model's correction
        }

    Returns:
        average GLEU score (float between 0 and 1)
    """
    scores = []
    for item in results:
        score = sentence_gleu(
            [item["original"].split()],   # reference (ground truth)
            item["prediction"].split()     # model output
        )
        scores.append(score)

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 4)