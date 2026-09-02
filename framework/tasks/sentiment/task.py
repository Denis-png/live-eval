import functools
import json
import os
from ..base_task import BaseTask

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sentiment", "sentiment.json")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


class SentimentTask(BaseTask):

    def __init__(self):
        self._config = _load_config()

    def get_error_types(self) -> list[str]:
        return self._config["error_types"]

    def get_prompt_instruction(self) -> str:
        return self._config["prompt"]

    def get_judge_prompt(self) -> str | None:
        return self._config.get("judge_prompt")

    def get_inverse_prompt(self) -> str | None:
        return self._config.get("inverse_prompt")

    def get_inverse_judge_prompt(self) -> str | None:
        return self._config.get("inverse_judge_prompt")

    def get_error_descriptions(self) -> dict[str, str]:
        return self._config.get("error_descriptions", {})

    def get_carrier_prompt(self) -> str | None:
        return self._config.get("carrier_prompt")

    def get_seedless_forward_prompt(self) -> str | None:
        return self._config.get("seedless_forward_prompt")

    def get_profile_side(self, mode: str) -> str:
        return "incorrect"

    def get_evaluators(self) -> list[str]:
        return self._config["evaluators"]

    _CLASSES = ("NEGATIVE", "NEUTRAL", "POSITIVE")

    def get_evaluator_fns(self) -> dict:
        from framework.evaluators.classification.accuracy import compute_accuracy
        from framework.evaluators.classification.macro_precision import compute_macro_precision
        from framework.evaluators.classification.macro_recall import compute_macro_recall
        from framework.evaluators.classification.macro_f1 import compute_macro_f1
        return {
            "accuracy": compute_accuracy,
            "macro_precision": functools.partial(compute_macro_precision, labels=self._CLASSES),
            "macro_recall": functools.partial(compute_macro_recall, labels=self._CLASSES),
            "macro_f1": functools.partial(compute_macro_f1, labels=self._CLASSES),
        }

    def get_model(self, model_config: dict):
        model_type = model_config["type"]
        params = self._config["models"].get(model_type, {})
        merged = {**model_config, **params}

        if model_type in ("bertweet", "multilingual"):
            from framework.models.sentiment.transformer import TransformerSentimentModel
            return TransformerSentimentModel(merged)
        raise ValueError(
            f"Unsupported sentiment model type: '{model_type}'. "
            f"Add it to configs/sentiment/sentiment.json and tasks/sentiment/task.py."
        )

    def get_label(self, result: dict) -> str | None:
        error_type = result.get("error_type", "")
        if "negative" in error_type or error_type in ("sarcasm_injection", "negation_insertion"):
            return "NEGATIVE"
        if "positive" in error_type:
            return "POSITIVE"
        if error_type == "intensity_reduction":
            return "NEUTRAL"
        return None  # paraphrase and unknown types: original sentiment unknown, skip

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        out = []
        for item in synthetic:
            label = self.get_label(item)
            if label is None:
                continue  # skip items without a deterministic ground-truth label
            out.append({**item, "text": item["corrupted"], "label": label})
        return out

    _LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    @classmethod
    def _normalize_label(cls, raw) -> str | None:
        """Map a dataset label onto a class name, or None when the row carries
        no usable one.

        An unusable label must skip the row, never be stringified: "None" would
        reach the real baseline as a class no model can predict, so every model
        scores wrong on it and the baseline is silently depressed. Digit strings
        are normalized because a local CSV delivers every field as text, and "0"
        would become the same kind of phantom class. Anything else still passes
        through, so datasets that already use class names keep working. Note 0
        is a VALID label (NEGATIVE) — this tests for None, not falsiness."""
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return None
            if raw.lstrip("-").isdigit():
                raw = int(raw)
        return cls._LABEL_MAP.get(raw, str(raw))

    def parse_row(self, row: dict) -> dict | None:
        text = row.get("text") or row.get("sentence") or row.get("review")
        if not text:
            return None
        label = self._normalize_label(row.get("label"))
        if label is None:
            return None
        return {"incorrect": text, "sentiment_label": label}

    def get_real_eval_samples(self, config: dict, real_data: list[dict]) -> list[dict]:
        return [
            {"text": r["incorrect"], "label": r["sentiment_label"]}
            for r in real_data
            if r.get("incorrect") and r.get("sentiment_label")
        ]

    def profile_dataset(self, rows: list[dict]) -> dict | None:
        from collections import Counter
        from framework.profiling.dataset_profiler import tokenize
        from framework.profiling.text_stats import WORD_BINS, length_distribution, style_profile

        texts = [r.get("text") or r.get("corrupted") or r.get("incorrect") for r in rows]
        texts = [t for t in texts if t]
        if not texts:
            return None

        error_types = [r.get("error_type") for r in rows if r.get("error_type")]
        type_counts = Counter(error_types)
        total = max(sum(type_counts.values()), 1)

        label_counts = Counter(r.get("label") for r in rows if r.get("label"))

        return {
            "num_samples": len(rows),
            "error_type_dist": {k: round(v / total, 4) for k, v in type_counts.most_common()},
            "label_dist": dict(label_counts),
            "word_count_hist": length_distribution(
                [len(tokenize(t)) for t in texts], WORD_BINS
            )["bins"],
            "style": style_profile(texts),
        }

    def compare_profiles(self, real: dict, generated: dict) -> dict:
        from framework.profiling.fidelity import jensen_shannon_divergence
        return {
            "type_dist_jsd": jensen_shannon_divergence(
                real.get("error_type_dist", {}), generated.get("error_type_dist", {})
            ),
            "label_dist_jsd": jensen_shannon_divergence(
                {k: v for k, v in (real.get("label_dist") or {}).items()},
                {k: v for k, v in (generated.get("label_dist") or {}).items()},
            ),
            "length_jsd": jensen_shannon_divergence(
                real.get("word_count_hist", {}), generated.get("word_count_hist", {})
            ),
            "style_deltas": {
                key: round(
                    generated.get("style", {}).get(key, 0.0)
                    - real.get("style", {}).get(key, 0.0), 4
                )
                for key in sorted(
                    set(real.get("style", {})) | set(generated.get("style", {}))
                )
            },
        }

    def get_task_name(self) -> str:
        return "sentiment"
