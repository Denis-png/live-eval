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
        # The corruption contract is {"incorrect", "correct"}: forward rewrites
        # the incorrect side, inverse the clean one. A real tweet is both -- the
        # unmodified source either mode transforms.
        return {"incorrect": text, "correct": text, "sentiment_label": label}

    def get_real_eval_samples(self, config: dict, real_data: list[dict]) -> list[dict]:
        return [
            {"text": r["incorrect"], "label": r["sentiment_label"]}
            for r in real_data
            if r.get("incorrect") and r.get("sentiment_label")
        ]

    def profile_error_distribution(self, real_data: list[dict], count_max: int = 5,
                                   config: dict | None = None) -> dict | None:
        """The error mix the inverse and seedless cells impose, from the real data.

        Real tweets carry a sentiment label, not an error type, so the empirical
        quantity is the benchmark's LABEL balance. Each label's share is spread
        evenly over the error types that produce it (get_label), so a benchmark
        drawn from this mix has the real class balance. One transformation per
        sample: two would contradict each other and leave the label ambiguous.
        None below 5 labelled rows, as the pipeline expects.
        """
        from collections import Counter

        counts = Counter(r.get("sentiment_label") for r in real_data
                         if r.get("sentiment_label"))
        producers: dict[str, list[str]] = {}
        for etype in self.get_error_descriptions():
            label = self.get_label({"error_type": etype})
            if label is not None:
                producers.setdefault(label, []).append(etype)
        total = sum(counts[label] for label in producers)
        if sum(counts.values()) < 5 or not total:
            return None
        type_dist = {etype: counts[label] / total / len(etypes)
                     for label, etypes in producers.items() for etype in etypes}
        return {"type_dist": type_dist, "count_dist": {1: 1.0}}

    def get_calibration_keys(self) -> dict[str, str]:
        # Measured on the error types generated records carry. In the cells that
        # impose a mix those are the requested types, so calibration corrects
        # the uneven attrition of parsing and judging, per type.
        return {"type_dist": "error_type_dist", "count_dist": "error_count_dist"}

    def build_fidelity_profile(self, rows: list[dict]) -> dict:
        """Label balance and text length/style, measured the same way on both sides.

        The pipeline passes real eval samples ({"text", "label"}) and raw generated
        records ({"original", "corrupted", "error_type"}). A generated record is
        labelled by its error type -- the label evaluation scores it against -- and
        one with no deterministic label (paraphrase) is left out, as evaluation
        leaves it out. Real tweets carry no error type, so the error-type mix has
        nothing to be compared with; error_type_dist.png shows it on its own.
        It is still MEASURED here -- error_type_dist and error_count_dist, empty
        on the real side -- because calibration steers on it.
        """
        from collections import Counter

        from framework.profiling.dataset_profiler import tokenize
        from framework.profiling.text_stats import WORD_BINS, length_distribution, style_profile

        pairs = [(r.get("text") or r.get("corrupted"), r.get("label") or self.get_label(r))
                 for r in rows]
        pairs = [(text, label) for text, label in pairs if text and label]
        texts = [text for text, _ in pairs]
        counts = Counter(label for _, label in pairs)
        n = len(pairs)
        typed = [[t.strip() for t in r["error_type"].split(",") if t.strip()]
                 for r in rows if r.get("error_type")]
        type_counts = Counter(t for types in typed for t in types)
        mentions = sum(type_counts.values())
        return {
            "num_samples": n,
            "label_dist": {label: round(counts[label] / n, 4) if n else 0.0
                           for label in sorted(set(self._CLASSES) | set(counts))},
            "word_count_hist": length_distribution(
                [len(tokenize(t)) for t in texts], WORD_BINS)["bins"],
            "style": style_profile(texts),
            "error_type_dist": {t: type_counts[t] / mentions
                                for t in sorted(type_counts)} if mentions else {},
            "error_count_dist": {c: k / len(typed) for c, k in
                                 sorted(Counter(len(types) for types in typed).items())}
                                if typed else {},
            # calibrate.informative_count reads this for corruption tasks (GEC:
            # pairs ERRANT annotated). Here: records carrying an error type.
            "n_annotated": len(typed),
        }

    def compare_fidelity_profiles(self, real: dict, generated: dict) -> dict:
        """Real->generated divergences over labels and lengths, plus per-label and
        style deltas. `profile_type` routes the session plot to its own figure."""
        from framework.profiling.fidelity import jensen_shannon_divergence

        def deltas(key: str) -> dict:
            r, g = real.get(key) or {}, generated.get(key) or {}
            return {k: round(g.get(k, 0.0) - r.get(k, 0.0), 4) for k in sorted(set(r) | set(g))}

        def jsd(key: str) -> float:
            return round(jensen_shannon_divergence(real.get(key) or {},
                                                   generated.get(key) or {}), 6)

        return {
            "profile_type": "sentiment_fidelity",
            "label_dist_jsd": jsd("label_dist"),
            "length_jsd": jsd("word_count_hist"),
            "label_deltas": deltas("label_dist"),
            "style_deltas": deltas("style"),
        }

    def get_task_name(self) -> str:
        return "sentiment"
