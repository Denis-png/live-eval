"""Task representation for Taxonomy Induction / Subclass Axiom Induction.

This task is structured: benchmark rows are whole taxonomies, not corrupted
text examples. BaseTask still has a few corruption-oriented abstract methods,
so this class implements those with empty values or clear errors while keeping
the actual taxonomy representation separate.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from framework.evaluators.taxonomy.diagnostics import compute_diagnostics
from framework.evaluators.taxonomy.f1 import compute_f1
from framework.evaluators.taxonomy.precision import compute_precision
from framework.evaluators.taxonomy.recall import compute_recall
from framework.evaluators.taxonomy.relations import (
    normalize_relation_pair,
    normalize_relation_set,
)
from framework.tasks.base_task import BaseTask
from framework.generators.base_generator import extract_json_object, preview_response
from framework.tasks.taxonomy.graph_ops import (
    EDIT_OPERATORS, matches_structure, sample_subtrees,
)

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


def _counts_from_fractions(fractions: dict, total: int) -> dict[str, int]:
    """Class counts summing exactly to `total`, by largest remainder, keyed by
    the string form the profiler writes. Empty bins are dropped."""
    weights = {str(k): max(0.0, float(v)) for k, v in (fractions or {}).items()}
    mass = sum(weights.values())
    if total <= 0 or mass <= 0:
        return {}
    exact = {k: total * w / mass for k, w in weights.items()}
    counts = {k: int(x) for k, x in exact.items()}
    shortfall = total - sum(counts.values())
    for k in sorted(exact, key=lambda k: (exact[k] - counts[k], k), reverse=True)[:shortfall]:
        counts[k] += 1
    return {k: counts[k] for k in sorted(counts, key=int) if counts[k] > 0}


_RAW_PREVIEW_LIMIT = 800


def _raw_preview(text: Any) -> str | None:
    """Bounded provider-output preview for diagnostics only. Keeps the start AND
    the end: a reasoning model's answer sits after its <think> block."""
    return preview_response(text, limit=_RAW_PREVIEW_LIMIT)


def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the model's JSON answer out of a response that may carry reasoning.

    Delegates to the extractor the evaluation parser also uses, so generation and
    evaluation read a reasoning model's output identically.
    """
    return extract_json_object(text)


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
        if model_type == "lexical":
            from framework.models.taxonomy import LexicalHeadMatchModel
            return LexicalHeadMatchModel(merged)
        if model_type == "star":
            from framework.models.taxonomy import StarModel
            return StarModel(merged)
        raise ValueError(
            f"Unsupported taxonomy model type: '{model_type}'. "
            "Supported types: llm, lexical, star."
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

    def build_structured_generation_prompt(
        self, profile: dict, rng=None, feedback: dict | None = None,
        mode: str = "inverse",
    ) -> str:
        """Build a taxonomy generation prompt.

        `inverse` imposes a structural target sampled from the real profile (and
        carries feedback from the previous round). `forward` supplies only the
        domain — every count, depth and branching property emerges, which is what
        makes a forward-vs-inverse comparison measure what targeting buys.
        """
        spec = self._generation_spec_from_profile(profile, rng=rng)
        if mode == "forward":
            return self._config["structured_forward_prompt"].format(
                domain=spec.get("domain", ""),
            )
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

    def get_seed_pool(self, config: dict, real_data: list[dict], mode: str,
                      *, seed_weights: dict | None = None, rng=None) -> list[dict]:
        """Subtrees of the real ontology, indexed by their own max depth.

        One ontology has to supply the whole pool, so it is sampled rather than
        used whole. `seed_weights` (a calibration artifact's, keyed by
        `str(max_depth)`) switches the pool to `sample_size` weighted draws
        with replacement; without it the whole pool is returned.

        Each subtree is stamped with `pool_index`, its position in the pool: ONE
        numbering over all of `real_data`. sample_subtrees enumerates in sorted
        order and ignores `rng`, so every caller -- generation with its rng, the
        real reference with none -- sees the same index on the same subtree.
        It is also stamped with its row's `ontology_id` and `domain`, so the
        real reference can take the whole pool in one call and still give each
        item its own ontology's id and domain.
        """
        opts = ((config.get("generation") or {}).get("seed_pool") or {})
        pool: list[dict] = []
        for row in real_data:
            for subtree in sample_subtrees(
                row.get("classes") or [], row.get("subclass_axioms") or [],
                max_depth=int(opts.get("max_depth", 4)),
                min_classes=int(opts.get("min_classes", 5)),
                rng=rng,
            ):
                pool.append({**subtree,
                             "ontology_id": row.get("ontology_id"),
                             "domain": row.get("domain"),
                             "pool_index": len(pool)})
        if not seed_weights:
            return pool
        # Calibrated: draw sample_size seeds -- a max-depth bucket by weight, then
        # a subtree uniformly within it, with replacement. With the whole pool
        # drawn every run, repeats are the only way a weight can shift the mix.
        from framework.calibration.seeds import draw_weighted_seeds

        index: dict[str, list[int]] = {}
        for i, subtree in enumerate(pool):
            index.setdefault(str(subtree.get("max_depth")), []).append(i)
        size = int((config.get("generation") or {}).get("sample_size", len(pool)))
        return draw_weighted_seeds(pool, index, seed_weights, size, rng or random.Random())

    def _target_domain(self, config_domains, rng) -> str:
        return rng.choice(list(config_domains)) if config_domains else "general knowledge"

    def build_seeded_artifact(self, seed: dict, mode: str,
                              profile: dict | None, config: dict, rng) -> dict:
        classes = list(seed["classes"])
        axioms = [list(a) for a in seed["subclass_axioms"]]

        if mode == "inverse":
            if not profile:
                raise RuntimeError(
                    f"{self.get_task_name()} inverse+seeded generation requires a "
                    "profile: the structural target it imposes is sampled from one."
                )
            # The spec is what inverse IMPOSES. It is sampled from the real
            # profile by the same accessor the seedless inverse cell uses, so
            # both inverse cells target the same distribution.
            self._generation_spec_from_profile(profile, rng=rng)   # validates the profile
            # Try operators in a shuffled order until one applies; an operator
            # returns None when the graph offers it nothing to do. Exactly ONE
            # edit is applied — that is all it takes to make inverse structurally
            # differ from forward, which is what this cell has to demonstrate.
            # Editing toward an exact target depth is targeting, and targeting is
            # a calibration concern, not this spec's.
            names = sorted(EDIT_OPERATORS)
            rng.shuffle(names)
            for name in names:
                edited = EDIT_OPERATORS[name](classes, axioms, rng)
                if edited is not None:
                    classes, axioms = edited
                    break
            if (sorted(classes), sorted(axioms)) == (
                    sorted(seed["classes"]), sorted(seed["subclass_axioms"])):
                raise RuntimeError(
                    f"{self.get_task_name()}: no edit operator could alter this "
                    f"seed (root {seed.get('root')!r}, {len(seed['classes'])} "
                    "classes). Raise generation.seed_pool.min_classes."
                )

        # Run config wins over taxonomy.json's default list, the same
        # precedence get_feedback_config already uses. One config block,
        # generation.seed_pool, owns every seeded setting.
        run_opts = ((config.get("generation") or {}).get("seed_pool") or {})
        domains = (run_opts.get("domains")
                   or (self._config.get("seed_pool") or {}).get("domains")
                   or ["general knowledge"])
        return {
            "domain": self._target_domain(domains, rng),
            "classes": classes,
            "subclass_axioms": axioms,
            "source_max_depth": seed.get("max_depth"),
            # Which pool subtree this gold came from, as an INTEGER: it pairs
            # the accepted record with its real subtree in the archive, and no
            # real class name or ontology id ever has to cross over to do it.
            "source_pool_index": seed.get("pool_index"),
        }

    def build_seeded_generation_prompt(self, gold: dict) -> str:
        """Anonymise every class before showing the structure.

        The model must reproduce the SHAPE, not translate the names. Sending the
        real identifiers would both leak the source ontology into the benchmark
        and invite the model to recall it rather than build from the structure.

        The anonymised order IS gold["classes"]'s order, because
        matches_structure maps positionally -- sorting here would silently
        reject every honest answer whenever gold's classes were not already
        sorted.
        """
        import json

        order = {name: f"C{i}" for i, name in enumerate(gold["classes"])}
        structure = {
            "classes": [order[c] for c in gold["classes"]],
            "subclass_axioms": sorted([order[c], order[p]]
                                      for c, p in gold["subclass_axioms"]),
        }
        template = self._config["seeded_generation_prompt"]
        return template.format(
            domain=gold["domain"],
            structure_json=json.dumps(structure, ensure_ascii=False, indent=2),
        )

    def verify_structured_match(self, gold: dict, parsed: dict) -> bool:
        return matches_structure(gold["classes"], gold["subclass_axioms"],
                                 parsed.get("classes") or [],
                                 parsed.get("subclass_axioms") or [])

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        """Build eval rows whose model input excludes gold subclass axioms."""
        return [self._eval_sample(row) for row in synthetic]

    def get_real_eval_samples(self, config: dict, real_data: list[dict]) -> list[dict]:
        """The real side of the comparison, drawn the way this cell's synthetic
        side is drawn.

        The reference used to be the whole ontology, as one item, for every
        cell. Seeded cells evaluate on subtrees of it, and F1 on a small graph
        is far easier than on a large one, so a seeded benchmark looked easier
        than the real one purely because of size -- with n=1 on the real side.
        The same reference feeds the structural fidelity profile, so the size
        confound hit that comparison too.

        Seeded: the real subtrees of the SAME seed pool generation draws from,
        with their real names and domain. The pool is used exactly as it is --
        never edited -- because the real side is real data.

        Per ITEM, a forward+seeded synthetic item is one pool subtree's
        structure under new names. Per SESSION the sides are not paired: the
        real side is the whole pool, scored once, while each run's synthetic
        side is what that run drew and what passed verification. So a seeded
        session's gap mixes the change of vocabulary with draw and verification
        attrition. The pairing is recoverable from the archive -- each real item
        carries `pool_index`, each accepted record the `source_pool_index` of its
        subtree -- but per-run paired scoring is not implemented.

        Seedless: the whole ontology, which is what seedless generation targets.
        """
        # Imported here, not at module level: a task must not load the pipeline
        # merely to be imported, and get_model defers its imports the same way.
        from framework.pipeline import resolve_mode, resolve_seedless

        strategy = self.get_generation_strategy()
        if resolve_seedless(config, strategy):
            return [self._eval_sample(row) for row in real_data]
        mode = resolve_mode(config, strategy)
        # ONE call over all of real_data: per-row calls would restart the pool
        # numbering at 0 for each ontology. Each subtree carries its own row's
        # ontology_id and domain.
        return [
            self._eval_sample({
                "ontology_id": subtree["ontology_id"],
                "domain": subtree["domain"],
                "classes": subtree["classes"],
                "subclass_axioms": subtree["subclass_axioms"],
                "pool_index": subtree["pool_index"],
            })
            for subtree in self.get_seed_pool(config, real_data, mode)
        ]

    def paired_real_indices(self, real_reference: list[dict],
                            synthetic: list[dict]) -> list[int] | None:
        """Seeded records carry `source_pool_index`, real items `pool_index`:
        each record pairs with the real subtree it was re-verbalised from.

        A seedless artifact comes from no particular real item, so a run whose
        records lack the index is not paired. An index that matches no real item
        means the session's files disagree, and raises rather than guessing."""
        if not synthetic or any(type(r.get("source_pool_index")) is not int
                                for r in synthetic):
            return None
        position = {item.get("pool_index"): i for i, item in enumerate(real_reference or [])}
        indices = []
        for record in synthetic:
            index = record["source_pool_index"]
            if index not in position:
                raise ValueError(f"source_pool_index {index} matches no real "
                                 "reference item")
            indices.append(position[index])
        return indices

    def build_fidelity_profile(self, rows: list[dict]) -> dict:
        """Profile taxonomy artifacts with the same structural profiler for all sides.

        The run-level artifact intentionally strips class-name-bearing debug
        fields (roots, leaves, class_depths) so fidelity reporting cannot expose
        real ontology class identifiers or URI provenance.
        """
        from framework.profiling.taxonomy_fidelity import (
            sanitize_taxonomy_profile,
            structure_measurements,
        )
        from framework.profiling.taxonomy_profiler import profile_taxonomy_rows

        profile = sanitize_taxonomy_profile(profile_taxonomy_rows(rows))
        # Top-level class measurements for calibration. The fidelity comparison
        # and its plots read `taxonomies`, never these.
        profile.update(structure_measurements(profile["taxonomies"]))
        return profile

    def compare_fidelity_profiles(self, real: dict, generated: dict) -> dict:
        """Real-vs-synthetic structural fidelity for taxonomy profiles."""
        from framework.profiling.taxonomy_fidelity import compare_taxonomy_profiles

        return compare_taxonomy_profiles(real, generated)

    def get_calibration_keys(self) -> dict[str, str]:
        # The framework's two control slots, reused: per-class depth and
        # per-class child count, the two structural distributions inverse+
        # seedless imposes. Parent count (multiple inheritance) is left out.
        return {"type_dist": "depth_dist", "count_dist": "child_count_dist"}

    def get_seed_calibration_key(self) -> str:
        # Seeded cells steer which subtrees are drawn, by max-depth bucket, so
        # what they measure is the delivered mix of those same buckets.
        return "max_depth_mix"

    def apply_calibrated_structure(self, profile: dict, request: dict) -> dict:
        """A copy of the benchmark profile imposing a calibration request.

        inverse+seedless imposes taxonomies[0]'s structure, through both the
        prompt and the feedback loop. Calibration replaces its depth and
        child-count distributions with the request -- class counts summing to
        n_classes -- and recomputes mean_depth and n_leaves so the imposed spec
        stays self-consistent. Every other field is kept; `profile` is not
        mutated."""
        import copy

        out = copy.deepcopy(profile)
        target = out["taxonomies"][0]
        n = int(target.get("n_classes") or 0)
        if request.get("type_dist"):
            depth = _counts_from_fractions(request["type_dist"], n)
            target["depth_distribution"] = depth
            target["mean_depth"] = (round(sum(int(d) * c for d, c in depth.items()) / n, 4)
                                    if n else 0.0)
        if request.get("count_dist"):
            children = _counts_from_fractions(request["count_dist"], n)
            target["child_count_distribution"] = children
            target["n_leaves"] = children.get("0", 0)
        return out

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
        profile: dict,
        artifact: dict,
        generation_config: dict | None = None,
    ) -> dict[str, Any]:
        """Profile one generated taxonomy and derive structural feedback.

        Parameter names follow BaseTask's contract (`profile`, `artifact`) so
        this is a true override and a keyword call against the declared
        signature works; here they are the real taxonomy profile and one
        generated taxonomy.
        """
        from framework.profiling.taxonomy_fidelity import (
            build_generation_feedback,
            compare_taxonomy_profiles,
        )

        synthetic_profile = self.build_fidelity_profile([artifact])
        comparison = compare_taxonomy_profiles(profile, synthetic_profile)
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
        sample = {
            "ontology_id": row.get("ontology_id"),
            "domain": domain,
            "classes": classes,
            "model_input": model_input,
            "text": serialize_taxonomy_model_input(domain, classes),
            "subclass_axioms": row.get("subclass_axioms", []),
        }
        # A seeded reference item's seed-pool position, so real_sample.json
        # records which subtree each real item is.
        if "pool_index" in row:
            sample["pool_index"] = row["pool_index"]
        return sample
