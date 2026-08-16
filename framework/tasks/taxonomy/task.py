"""Task representation for Taxonomy Induction / Subclass Axiom Induction.

This task is structured: benchmark rows are whole taxonomies, not corrupted
text examples. BaseTask still has a few corruption-oriented abstract methods,
so this class implements those with empty values or clear errors while keeping
the actual taxonomy representation separate.
"""

from __future__ import annotations

import json
import os
from typing import Any

from framework.evaluators.taxonomy.metrics import (
    compute_diagnostics,
    compute_f1,
    compute_precision,
    compute_recall,
    normalize_relation_set,
)
from framework.tasks.base_task import BaseTask

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "taxonomy", "taxonomy.json"
)


def _load_config() -> dict[str, Any]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def taxonomy_model_input(domain: str, classes: list[str]) -> dict[str, Any]:
    """Return the only fields an evaluated model is allowed to see."""
    return {"domain": domain, "classes": classes}


def serialize_taxonomy_model_input(domain: str, classes: list[str]) -> str:
    """Stable JSON prompt payload containing no gold subclass information."""
    return json.dumps(taxonomy_model_input(domain, classes), ensure_ascii=False, sort_keys=True)


class TaxonomyTask(BaseTask):
    """Structured task for predicting direct subclass relations."""

    def __init__(self):
        self._config = _load_config()

    def get_task_name(self) -> str:
        return "taxonomy"

    def get_error_types(self) -> list[str]:
        """Taxonomy induction has no corruption/error-type vocabulary."""
        return []

    def get_prompt_instruction(self) -> str:
        raise NotImplementedError(
            "TaxonomyTask does not support text corruption prompts. "
            "Structured taxonomy generation is intentionally out of scope for Phase 2."
        )

    def get_evaluators(self) -> list[str]:
        return self._config["evaluators"]

    def get_evaluator_fns(self) -> dict[str, Any]:
        return {
            "precision": compute_precision,
            "recall": compute_recall,
            "f1": compute_f1,
            "diagnostics": compute_diagnostics,
        }

    def get_model(self, model_config: dict):
        raise NotImplementedError(
            "Taxonomy model wrappers are not implemented in Phase 2. "
            "Use taxonomy evaluators directly with parsed predictions."
        )

    def parse_row(self, row: dict) -> dict | None:
        """Parse one normalized taxonomy row while preserving its structure."""
        domain = row.get("domain")
        classes = row.get("classes")
        if not isinstance(domain, str) or not domain.strip():
            return None
        if not isinstance(classes, list) or not classes:
            return None

        normalized_classes = [str(name).strip() for name in classes if str(name).strip()]
        if not normalized_classes:
            return None

        subclass_axioms = sorted(normalize_relation_set(row.get("subclass_axioms") or []))
        return {
            "ontology_id": row.get("ontology_id"),
            "domain": domain.strip(),
            "classes": normalized_classes,
            "subclass_axioms": [[child, parent] for child, parent in subclass_axioms],
            "metadata": row.get("metadata") or {},
        }

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        """Build eval rows whose model input excludes gold subclass axioms."""
        return [self._eval_sample(row) for row in synthetic]

    def get_real_eval_samples(self, config: dict, real_data: list[dict]) -> list[dict]:
        return [self._eval_sample(row) for row in real_data]

    def _eval_sample(self, row: dict) -> dict:
        domain = row["domain"]
        classes = list(row["classes"])
        model_input = taxonomy_model_input(domain, classes)
        return {
            "ontology_id": row.get("ontology_id"),
            "domain": domain,
            "classes": classes,
            "model_input": model_input,
            "text": serialize_taxonomy_model_input(domain, classes),
            "subclass_axioms": row.get("subclass_axioms", []),
        }
