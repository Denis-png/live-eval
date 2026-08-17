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


_RAW_PREVIEW_LIMIT = 800


def _raw_preview(text: Any) -> str | None:
    """Bounded provider-output preview for diagnostics only."""
    if text is None:
        return None
    preview = text if isinstance(text, str) else repr(text)
    return preview[:_RAW_PREVIEW_LIMIT]


def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON object, tolerating fenced responses if present."""
    if not isinstance(text, str):
        return None, "non_string_response"
    stripped = text.strip()
    if not stripped:
        return None, "empty_response"
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None, "malformed_json"
    if not isinstance(payload, dict):
        return None, "malformed_json"
    return payload, None


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

    def build_structured_generation_prompt(self, profile: dict, rng=None, feedback: dict | None = None) -> str:
        """Build a taxonomy generation prompt from structural profile targets only."""
        spec = self._generation_spec_from_profile(profile, rng=rng)
        return self._config["structured_generation_prompt"].format(
            spec_json=json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False),
            feedback_section=self._format_feedback_section(feedback),
        )

    def parse_structured_generation(self, text: str) -> dict | None:
        """Parse and validate one generated taxonomy JSON object."""
        return self.parse_structured_generation_with_diagnostics(text)["artifact"]

    def parse_structured_generation_with_diagnostics(self, text: str) -> dict[str, Any]:
        """Parse a generated taxonomy and return a structured rejection reason.

        The public parse_structured_generation() method keeps the legacy
        artifact-or-None contract; this companion method uses the same parsing
        path while preserving bounded diagnostics for real smoke runs.
        """
        payload, reason = _extract_json_object(text)
        if payload is None:
            return self._structured_parse_result(None, reason, text)
        domain = payload.get("domain")
        raw_classes = payload.get("classes")
        raw_axioms = payload.get("subclass_axioms")
        if not isinstance(domain, str) or not domain.strip():
            return self._structured_parse_result(None, "missing_or_invalid_domain", text)
        if not isinstance(raw_classes, list) or not raw_classes:
            return self._structured_parse_result(None, "missing_or_invalid_classes", text)
        if not isinstance(raw_axioms, list):
            return self._structured_parse_result(None, "malformed_subclass_axiom", text)

        classes = []
        seen_classes = set()
        for value in raw_classes:
            if not isinstance(value, str) or not value.strip():
                return self._structured_parse_result(None, "missing_or_invalid_classes", text)
            normalized = value.strip()
            if normalized in seen_classes:
                return self._structured_parse_result(None, "duplicate_classes", text)
            seen_classes.add(normalized)
            classes.append(normalized)

        class_set = set(classes)
        edges: set[tuple[str, str]] = set()
        for value in raw_axioms:
            relation = normalize_relation_pair(value)
            if relation is None or not relation[0] or not relation[1]:
                return self._structured_parse_result(None, "malformed_subclass_axiom", text)
            child, parent = relation
            if child not in class_set or parent not in class_set:
                return self._structured_parse_result(None, "unknown_class_endpoint", text)
            if child == parent:
                return self._structured_parse_result(None, "self_loop", text)
            edges.add(relation)
        if _has_cycle(class_set, edges):
            return self._structured_parse_result(None, "cycle", text)

        artifact = {
            "domain": domain.strip(),
            "classes": classes,
            "subclass_axioms": [[child, parent] for child, parent in sorted(edges)],
            "generation_diagnostics": {
                "n_classes": len(classes),
                "n_subclass_axioms": len(edges),
            },
        }
        return self._structured_parse_result(artifact, None, text)

    def _structured_parse_result(
        self,
        artifact: dict | None,
        rejection_reason: str | None,
        raw_text: Any,
    ) -> dict[str, Any]:
        diagnostic = {"valid": artifact is not None}
        if rejection_reason is not None:
            diagnostic["rejection_reason"] = rejection_reason
            diagnostic["raw_preview"] = _raw_preview(raw_text)
        return {"artifact": artifact, "diagnostic": diagnostic}

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

    def profile_dataset(self, rows: list[dict]) -> dict:
        """Profile taxonomy artifacts with the same structural profiler for all sides.

        The run-level artifact intentionally strips class-name-bearing debug
        fields (roots, leaves, class_depths) so fidelity reporting cannot expose
        real ontology class identifiers or URI provenance.
        """
        from framework.profiling.taxonomy_fidelity import sanitize_taxonomy_profile
        from framework.profiling.taxonomy_profiler import profile_taxonomy_rows

        return sanitize_taxonomy_profile(profile_taxonomy_rows(rows))

    def compare_profiles(self, real: dict, generated: dict) -> dict:
        """Real-vs-synthetic structural fidelity for taxonomy profiles."""
        from framework.profiling.taxonomy_fidelity import compare_taxonomy_profiles

        return compare_taxonomy_profiles(real, generated)

    def get_feedback_config(self, generation_config: dict | None = None) -> dict:
        """Return taxonomy feedback settings, letting run config override defaults."""
        default = self._config.get("feedback") or {}
        override = (generation_config or {}).get("feedback") or {}
        tolerances = {
            **(default.get("tolerances") or {}),
            **(override.get("tolerances") or {}),
        }
        return {
            **default,
            **override,
            "tolerances": tolerances,
        }

    def build_structural_feedback(
        self,
        real_profile: dict,
        synthetic_taxonomy: dict,
        generation_config: dict | None = None,
    ) -> dict[str, Any]:
        """Profile one generated taxonomy and derive structural feedback."""
        from framework.profiling.taxonomy_fidelity import (
            build_generation_feedback,
            compare_taxonomy_profiles,
        )

        synthetic_profile = self.profile_dataset([synthetic_taxonomy])
        comparison = compare_taxonomy_profiles(real_profile, synthetic_profile)
        per_taxonomy = comparison["comparisons"][0] if comparison["comparisons"] else {}
        reference = comparison["real_profile"]
        synthetic = comparison["synthetic_profiles"][0] if comparison["synthetic_profiles"] else {}
        feedback_cfg = self.get_feedback_config(generation_config)
        feedback = build_generation_feedback(
            reference,
            synthetic,
            per_taxonomy,
            feedback_cfg.get("tolerances") or {},
        )
        return {
            "synthetic_profile": synthetic,
            "comparison": per_taxonomy,
            "feedback": feedback,
        }

    def _format_feedback_section(self, feedback: dict | None) -> str:
        if not feedback or not feedback.get("messages"):
            return ""
        payload = {
            "within_tolerance": feedback.get("within_tolerance", False),
            "adjustments": feedback.get("adjustments", []),
            "messages": feedback.get("messages", []),
        }
        return (
            "\nFeedback from previous generation:\n"
            "Use this structural feedback as guidance while keeping the original "
            "target profile authoritative. Do not copy any real ontology content.\n\n"
            f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}\n\n"
        )

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
