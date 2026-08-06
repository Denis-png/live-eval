import json
import os
import functools
from ..base_task import BaseTask
from framework.profiling.spam_profiler import (
    DEFAULT_SPAM_DATASET,
    DEFAULT_SPAM_SPLIT,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "spam", "spam.json")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


class SpamTask(BaseTask):

    def __init__(self):
        self._config = _load_config()

    def get_error_types(self) -> list[str]:
        return self._config["error_types"]

    def get_carrier_prompt(self) -> str | None:
        return self._config.get("carrier_prompt")

    def get_forward_prompt(self) -> str | None:
        return self._config.get("forward_prompt")

    def get_seedless_class_prompts(self) -> dict[str, str]:
        return self._config.get("seedless_class_prompts", {})

    def get_seed_pool(self, config: dict, real_data: list[dict], mode: str) -> list[dict]:
        """Forward mode imitates within a class, so it needs labeled seeds of
        BOTH classes — parse_row keeps HAM only. Inverse keeps today's pool."""
        if mode != "forward":
            return real_data
        return [r for r in self._load_reference_rows(config) if r.get("text")]

    def get_prompt_instruction(self) -> str:
        return self._config["prompt"]

    def get_judge_prompt(self) -> str | None:
        return self._config.get("judge_prompt")

    def get_inverse_prompt(self) -> str | None:
        return self._config.get("inverse_prompt")

    def get_ham_generation_prompt(self) -> str:
        return self._config["ham_generation_prompt"]

    def get_inverse_judge_prompt(self) -> str | None:
        return self._config.get("inverse_judge_prompt")

    def get_generation_strategy(self) -> str:
        return "class_conditional"

    def get_error_descriptions(self) -> dict[str, str]:
        return self._config.get("inverse_error_descriptions", {})

    def _load_reference_rows(self, config: dict | None) -> list[dict]:
        """Load the labeled real reference ({"text","label"} rows, both classes)
        from the configured dataset, defaulting to the whole split. Shared by
        profiling and the real baseline."""
        from framework.profiling import spam_profiler
        from framework.data_loading import iter_local_rows, resolve_dataset_config

        cfg = config or {}
        ds = resolve_dataset_config(cfg.get("dataset") or {})
        # None -> no cap: the labeled reference defaults to the whole split.
        reference_size = ds.get("reference_size")

        if ds["source"] == "local":
            rows = []
            for raw in iter_local_rows(ds["path"], ds["format"]):
                parsed = spam_profiler.normalize_spam_row(raw)
                if parsed is not None:
                    rows.append(parsed)
                if reference_size is not None and len(rows) >= reference_size:
                    break
            return rows
        hf_token = ds.get("hf_token") or (cfg.get("api_keys") or {}).get("huggingface")
        return spam_profiler.load_spam_rows(
            sample_size=reference_size,
            streaming=ds.get("streaming", False),
            hf_token=hf_token,
            dataset_name=ds.get("name") or DEFAULT_SPAM_DATASET,
            split=ds.get("split") or DEFAULT_SPAM_SPLIT,
        )

    def get_real_eval_samples(self, config: dict, real_data: list[dict]) -> list[dict]:
        rows = self._load_reference_rows(config)
        return [{"text": r["text"], "label": r["label"]} for r in rows if r.get("text")]

    def profile_error_distribution(self, real_data, count_max=5, config=None):
        """Empirical inverse-mode distribution over spam signals. real_data is
        HAM-only (parse_row drops SPAM), so load the SPAM subset separately via
        the spam profiler and count which signals fire per SPAM message."""
        from framework.profiling.spam_distribution import profile_spam_distribution
        rows = self._load_reference_rows(config)
        spam_rows = [r for r in rows if r["label"] == "SPAM"]
        supported = set(self.get_error_descriptions().keys())
        return profile_spam_distribution(spam_rows, supported, count_max=count_max)

    def profile_dataset(self, rows: list[dict]) -> dict:
        """Class balance + per-signal fire rate + normalized signal / count
        distributions over the SPAM messages, using the same detectors the
        generator injects (so real and generated are measured identically)."""
        from collections import Counter
        from framework.profiling.spam_profiler import detect_signals
        from framework.profiling.dataset_profiler import tokenize
        from framework.profiling.text_stats import (
            WORD_BINS,
            length_distribution,
            style_profile,
        )

        supported = list(self.get_error_descriptions().keys())
        n = len(rows)
        ham = sum(1 for r in rows if r.get("label") == "HAM")
        spam = sum(1 for r in rows if r.get("label") == "SPAM")
        spam_texts = [r["text"] for r in rows if r.get("label") == "SPAM" and r.get("text")]

        fire = Counter()
        per_msg = []
        for text in spam_texts:
            sigs = [s for s in detect_signals(text) if s in supported]
            fire.update(sigs)
            per_msg.append(min(len(sigs), 5))
        m = len(spam_texts)
        signal_rate = {s: (fire.get(s, 0) / m if m else 0.0) for s in supported}
        total = sum(fire.values())
        signal_type_dist = {s: (fire.get(s, 0) / total if total else 0.0) for s in supported}
        count_counter = Counter(per_msg)
        signal_count_dist = (
            {k: count_counter[k] / len(per_msg) for k in sorted(count_counter)}
            if per_msg else {}
        )

        texts_by_label = {
            "HAM": [r["text"] for r in rows if r.get("label") == "HAM" and r.get("text")],
            "SPAM": spam_texts,
        }
        word_count_hist_per_label = {
            label: length_distribution(
                [len(tokenize(t)) for t in texts], WORD_BINS
            )["bins"]
            for label, texts in texts_by_label.items()
        }
        style_per_label = {
            label: style_profile(texts) for label, texts in texts_by_label.items()
        }

        return {
            "n": n,
            "class_balance": {
                "HAM": ham, "SPAM": spam,
                "spam_fraction": (spam / n if n else 0.0),
            },
            "signal_rate": signal_rate,
            "signal_type_dist": signal_type_dist,
            "signal_count_dist": signal_count_dist,
            "word_count_hist_per_label": word_count_hist_per_label,
            "style_per_label": style_per_label,
        }

    def compare_profiles(self, real: dict, generated: dict) -> dict:
        """Real→generated deltas + Jensen-Shannon divergences. See fidelity honesty
        note: signals are re-detected by regex, so this measures detector-visible
        distribution match, not ground-truth semantics."""
        from framework.profiling.fidelity import jensen_shannon_divergence
        signals = set(real.get("signal_rate", {})) | set(generated.get("signal_rate", {}))
        return {
            "class_balance_delta": (
                generated["class_balance"]["spam_fraction"]
                - real["class_balance"]["spam_fraction"]
            ),
            "signal_deltas": {
                s: generated.get("signal_rate", {}).get(s, 0.0)
                   - real.get("signal_rate", {}).get(s, 0.0)
                for s in signals
            },
            "type_dist_jsd": jensen_shannon_divergence(
                real.get("signal_type_dist", {}), generated.get("signal_type_dist", {})
            ),
            "count_dist_jsd": jensen_shannon_divergence(
                real.get("signal_count_dist", {}), generated.get("signal_count_dist", {})
            ),
            "length_jsd_per_label": {
                label: jensen_shannon_divergence(
                    real.get("word_count_hist_per_label", {}).get(label, {}),
                    generated.get("word_count_hist_per_label", {}).get(label, {}),
                )
                for label in ("HAM", "SPAM")
            },
            "style_deltas_per_label": {
                label: {
                    key: round(
                        generated.get("style_per_label", {}).get(label, {}).get(key, 0.0)
                        - real.get("style_per_label", {}).get(label, {}).get(key, 0.0), 4)
                    for key in sorted(
                        set(real.get("style_per_label", {}).get(label, {}))
                        | set(generated.get("style_per_label", {}).get(label, {}))
                    )
                }
                for label in ("HAM", "SPAM")
            },
            "note": "signals re-detected by regex on generated text; measures "
                    "detector-visible distribution match, not semantic spamminess.",
        }

    def get_evaluators(self) -> list[str]:
        return self._config["evaluators"]

    def get_evaluator_fns(self) -> dict:
        from framework.evaluators.classification.accuracy import compute_accuracy
        from framework.evaluators.classification.precision import compute_precision
        from framework.evaluators.classification.recall import compute_recall
        from framework.evaluators.classification.f1 import compute_f1
        from framework.evaluators.classification.fpr import compute_fpr
        return {
            "accuracy":  compute_accuracy,
            "precision": functools.partial(compute_precision, positive_label="SPAM"),
            "recall":    functools.partial(compute_recall, positive_label="SPAM"),
            "f1":        functools.partial(compute_f1, positive_label="SPAM"),
            "fpr":       functools.partial(compute_fpr, positive_label="SPAM"),
        }

    def get_model(self, model_config: dict):
        model_type = model_config["type"]
        params = self._config["models"].get(model_type, {})
        merged = {**model_config, **params}

        if model_type == "roberta":
            from framework.models.spam.roberta import RobertaSpamModel
            return RobertaSpamModel(merged)
        elif model_type in ("bert_tiny", "bert"):
            # Both are AutoModelForSequenceClassification fine-tunes; only the
            # label_map / max_length from spam.json["models"] differ per type.
            from framework.models.spam.bert_tiny import BertTinySpamModel
            return BertTinySpamModel(merged)
        else:
            raise ValueError(
                f"Unsupported spam model type: '{model_type}'. "
                f"Add it to configs/spam/spam.json and tasks/spam/task.py."
            )
        
    def get_label(self, result: dict) -> str:
        # Paraphrased samples remain legitimate (HAM); all other error types produce SPAM.
        if result.get("error_type") == "paraphrase":
            return "HAM"
        return "SPAM"

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        """Class-conditional records already carry text+label (both classes are
        first-class generated samples), so no expansion is needed."""
        return [{**item, "text": item["text"], "label": item["label"]} for item in synthetic]

    def parse_row(self, row: dict) -> dict | None:
        label = str(row.get("label", "")).lower()
        if label in ("spam", "1"):
            return None
        text = row.get("text") or row.get("message") or row.get("sms")
        if not text:
            return None
        return {"incorrect": text}

    def get_task_name(self) -> str:
        return "spam"