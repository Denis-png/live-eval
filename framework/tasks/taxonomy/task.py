"""Task representation for Taxonomy Induction / Subclass Axiom Induction.

This task is structured: benchmark rows are whole taxonomies, not corrupted
text examples. BaseTask still has a few corruption-oriented abstract methods,
so this class implements those with empty values or clear errors while keeping
the actual taxonomy representation separate.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from framework.evaluators.taxonomy.metrics import (
    compute_diagnostics,
    compute_f1,
    compute_precision,
    compute_recall,
    normalize_relation_pair,
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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating fenced responses if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _has_cycle(classes: set[str], edges: set[tuple[str, str]]) -> bool:
    children_by_parent: dict[str, set[str]] = {name: set() for name in classes}
    for child, parent in edges:
        children_by_parent.setdefault(parent, set()).add(child)
    state = {name: 0 for name in classes}

    def visit(node: str) -> bool:
        state[node] = 1
        for child in sorted(children_by_parent.get(node, set())):
            if state[child] == 1:
                return True
            if state[child] == 0 and visit(child):
                return True
        state[node] = 2
        return False

    return any(state[name] == 0 and visit(name) for name in sorted(classes))


class TaxonomyTask(BaseTask):
    """Structured task for predicting direct subclass relations."""

    def __init__(self):
        self._config = _load_config()

    def get_task_name(self) -> str:
        return "taxonomy"

    def get_error_types(self) -> list[str]:
        """Taxonomy induction has no corruption/error-type vocabulary."""
        return []

    def get_generation_strategy(self) -> str:
        return "structured"

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
        model_type = model_config["type"]
        params = self._config.get("models", {}).get(model_type, {})
        merged = {**model_config, **params}

        if model_type == "llm":
            from framework.models.taxonomy import TaxonomyLLMModel
            return TaxonomyLLMModel(merged)
        raise ValueError(
            f"Unsupported taxonomy model type: '{model_type}'. "
            "Supported MVP type: llm."
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

    def build_structured_generation_prompt(self, profile: dict, rng=None) -> str:
        """Build a taxonomy generation prompt from structural profile targets only."""
        spec = self._generation_spec_from_profile(profile, rng=rng)
        return self._config["structured_generation_prompt"].format(
            spec_json=json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False)
        )

    def parse_structured_generation(self, text: str) -> dict | None:
        """Parse and validate one generated taxonomy JSON object."""
        payload = _extract_json_object(text)
        if payload is None:
            return None
        domain = payload.get("domain")
        raw_classes = payload.get("classes")
        raw_axioms = payload.get("subclass_axioms")
        if not isinstance(domain, str) or not domain.strip():
            return None
        if not isinstance(raw_classes, list) or not raw_classes:
            return None
        if not isinstance(raw_axioms, list):
            return None

        classes = []
        seen_classes = set()
        for value in raw_classes:
            if not isinstance(value, str) or not value.strip():
                return None
            normalized = value.strip()
            if normalized in seen_classes:
                return None
            seen_classes.add(normalized)
            classes.append(normalized)

        class_set = set(classes)
        edges: set[tuple[str, str]] = set()
        for value in raw_axioms:
            relation = normalize_relation_pair(value)
            if relation is None or not relation[0] or not relation[1]:
                return None
            child, parent = relation
            if child not in class_set or parent not in class_set:
                return None
            if child == parent:
                return None
            edges.add(relation)
        if _has_cycle(class_set, edges):
            return None

        return {
            "domain": domain.strip(),
            "classes": classes,
            "subclass_axioms": [[child, parent] for child, parent in sorted(edges)],
            "generation_diagnostics": {
                "n_classes": len(classes),
                "n_subclass_axioms": len(edges),
            },
        }

    def _generation_spec_from_profile(self, profile: dict, rng=None) -> dict[str, Any]:
        taxonomies = profile.get("taxonomies") or []
        if not taxonomies:
            raise RuntimeError("Taxonomy structured generation requires a taxonomy profile.")
        source = taxonomies[0]
        keys = [
            "domain",
            "n_classes",
            "n_subclass_axioms",
            "n_roots",
            "n_leaves",
            "max_depth",
            "mean_depth",
            "depth_distribution",
            "parent_count_distribution",
            "child_count_distribution",
            "multiple_parent_fraction",
        ]
        return {key: source.get(key) for key in keys if key in source}

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
