import json
import os
import functools
from ..base_task import BaseTask
from framework.profiling.spam_profiler import (
    DEFAULT_SPAM_DATASET,
    DEFAULT_SPAM_SPLIT,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tasks", "spam.json")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


class SpamTask(BaseTask):

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

    def get_generation_strategy(self) -> str:
        return "class_conditional"

    def get_ham_generation_prompt(self) -> str:
        return self._config["ham_generation_prompt"]

    def get_error_descriptions(self) -> dict[str, str]:
        return self._config.get("inverse_error_descriptions", {})

    def profile_error_distribution(self, real_data, count_max=5, config=None):
        """Empirical inverse-mode distribution over spam signals. real_data is
        HAM-only (parse_row drops SPAM), so load the SPAM subset separately via
        the spam profiler and count which signals fire per SPAM message."""
        from framework.profiling import spam_profiler
        from framework.profiling.spam_distribution import profile_spam_distribution

        from framework.data_loading import iter_local_rows, resolve_dataset_config

        cfg = config or {}
        ds = resolve_dataset_config(cfg.get("dataset") or {})
        inv = ((cfg.get("generation") or {}).get("inverse") or {})
        profile_size = inv.get("profile_size", 200)

        if ds["source"] == "local":
            # Profile SPAM signals from the same local file the run evaluates
            # against (parse_row drops SPAM rows, so re-read the raw file).
            rows = []
            for raw in iter_local_rows(ds["path"], ds["format"]):
                parsed = spam_profiler.normalize_spam_row(raw)
                if parsed is not None:
                    rows.append(parsed)
                if len(rows) >= profile_size:
                    break
        else:
            hf_token = ds.get("hf_token") or (cfg.get("api_keys") or {}).get("huggingface")
            rows = spam_profiler.load_spam_rows(
                sample_size=profile_size,
                streaming=ds.get("streaming", False),
                hf_token=hf_token,
                dataset_name=ds.get("name") or DEFAULT_SPAM_DATASET,
                split=ds.get("split") or DEFAULT_SPAM_SPLIT,
            )
        spam_rows = [r for r in rows if r["label"] == "SPAM"]
        supported = set(self.get_error_descriptions().keys())
        return profile_spam_distribution(spam_rows, supported, count_max=count_max)

    def profile_dataset(self, rows: list[dict]) -> dict:
        """Class balance + per-signal fire rate + normalized signal / count
        distributions over the SPAM messages, using the same detectors the
        generator injects (so real and generated are measured identically)."""
        from collections import Counter
        from framework.profiling.spam_profiler import detect_signals

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
        return {
            "n": n,
            "class_balance": {
                "HAM": ham, "SPAM": spam,
                "spam_fraction": (spam / n if n else 0.0),
            },
            "signal_rate": signal_rate,
            "signal_type_dist": signal_type_dist,
            "signal_count_dist": signal_count_dist,
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
        else:
            raise ValueError(
                f"Unsupported spam model type: '{model_type}'. "
                f"Add it to configs/tasks/spam.json and tasks/spam/task.py."
            )
        
    def get_label(self, result: dict) -> str:
        # Paraphrased samples remain legitimate (HAM); all other error types produce SPAM.
        if result.get("error_type") == "paraphrase":
            return "HAM"
        return "SPAM"

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        """Score the corrupted message, and for genuine SPAM items also score the
        clean source as a HAM negative so precision/recall/f1/fpr stay meaningful."""
        out = []
        for item in synthetic:
            label = self.get_label(item)
            out.append({**item, "text": item["corrupted"], "label": label})
            original = item.get("original")
            if label == "SPAM" and original and original.strip() != item["corrupted"].strip():
                out.append({
                    "text": original, "label": "HAM", "error_type": "clean",
                    "corrupted": original, "original": original,
                })
        return out

    def parse_row(self, row: dict) -> dict | None:
        # Skip spam rows — the generator needs legitimate (HAM) messages as input.
        if str(row.get("label", "")).lower() in ("spam", "1"):
            return None
        text = row.get("text") or row.get("message") or row.get("sms")
        if not text:
            return None
        return {"incorrect": text}

    def get_task_name(self) -> str:
        return "spam"