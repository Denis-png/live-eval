"""Structural fidelity comparison for taxonomy GET benchmarks.

This module compares a real taxonomy profile against one or more generated
taxonomy profiles. It reuses the structural profile shape from
taxonomy_profiler; it does not inspect class names, URI provenance, text, or
semantic labels.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from framework.profiling.fidelity import jensen_shannon_divergence

SCALAR_KEYS = [
    "n_classes",
    "n_subclass_axioms",
    "n_roots",
    "n_leaves",
    "max_depth",
    "mean_depth",
    "multiple_parent_fraction",
]

DISTRIBUTION_KEYS = [
    "depth_distribution",
    "parent_count_distribution",
    "child_count_distribution",
]

DEFAULT_FEEDBACK_TOLERANCES = {
    # Count targets are approximate; 15% keeps the loop from chasing small noise.
    "count_relative": 0.15,
    # Depth is integer-ish but mean_depth can be fractional. A half-level gap is
    # enough to be worth nudging without demanding exact equality.
    "depth_absolute": 0.5,
    # multiple_parent_fraction is a rate in [0, 1].
    "rate_absolute": 0.05,
    # JSD is in [0, 1], where lower means closer.
    "distribution_jsd": 0.1,
}

STRUCTURAL_PROFILE_KEYS = [
    "domain",
    *SCALAR_KEYS,
    *DISTRIBUTION_KEYS,
    "has_cycle",
    "validation",
]


def feedback_tolerances(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Merge caller tolerances with conservative taxonomy defaults."""
    merged = dict(DEFAULT_FEEDBACK_TOLERANCES)
    for key, value in (config or {}).items():
        if key in merged and isinstance(value, (int, float)):
            merged[key] = float(value)
    return merged


def sanitize_taxonomy_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a structural-only taxonomy profile safe for fidelity artifacts.

    The Phase 1 profiler keeps roots, leaves, and class_depths to aid debugging.
    Those contain class identifiers, so run-level fidelity artifacts strip them
    and retain only aggregate structure.
    """
    taxonomies = [
        {key: taxonomy.get(key) for key in STRUCTURAL_PROFILE_KEYS if key in taxonomy}
        for taxonomy in profile.get("taxonomies") or []
    ]
    return {
        "profile_type": profile.get("profile_type", "taxonomy_structure"),
        "profile_version": profile.get("profile_version", 1),
        "num_taxonomies": len(taxonomies),
        "taxonomies": taxonomies,
        "summary": profile.get("summary", {}),
    }


def select_reference_taxonomy_profile(
    profile: dict[str, Any],
    ontology_id: str | None = None,
) -> dict[str, Any]:
    """Select the real taxonomy profile used as the fidelity reference.

    The Pizza MVP uses one real ontology. If a future profile contains multiple
    real ontologies, callers must name the reference explicitly instead of
    silently comparing against the first unrelated ontology.
    """
    taxonomies = profile.get("taxonomies") or []
    if not taxonomies:
        raise ValueError("Taxonomy fidelity requires at least one real taxonomy profile.")
    if ontology_id is not None:
        matches = [t for t in taxonomies if t.get("ontology_id") == ontology_id]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"No unique taxonomy profile found for ontology_id={ontology_id!r}.")
    if len(taxonomies) == 1:
        return taxonomies[0]
    raise ValueError(
        "Taxonomy fidelity received multiple real taxonomy profiles; pass an "
        "explicit ontology_id to select the reference."
    )


def _relative_difference(real_value: float | int | None, absolute_difference: float) -> float | None:
    if real_value is None:
        return None
    real = float(real_value)
    if real == 0.0:
        return 0.0 if absolute_difference == 0.0 else None
    return round(absolute_difference / abs(real), 4)


def compare_scalar(real: dict[str, Any], synthetic: dict[str, Any], key: str) -> dict[str, Any]:
    """Compare one scalar characteristic.

    Relative difference is absolute_difference / abs(real). When the real value
    is zero and synthetic differs, relative difference is not meaningful and is
    reported as None.
    """
    real_value = real.get(key)
    synthetic_value = synthetic.get(key)
    if real_value is None or synthetic_value is None:
        absolute = None
    else:
        absolute = round(float(synthetic_value) - float(real_value), 4)
    return {
        "real": real_value,
        "synthetic": synthetic_value,
        "absolute_difference": absolute,
        "relative_difference": (
            _relative_difference(real_value, abs(absolute))
            if absolute is not None else None
        ),
    }


def compare_distribution(real: dict[str, Any], synthetic: dict[str, Any], key: str) -> dict[str, Any]:
    """Compare one count distribution with Jensen-Shannon divergence.

    Missing bins are aligned with zero counts by the JSD helper. Inputs are
    normalized to probabilities before comparison. Lower divergence means the
    synthetic distribution is more similar to the real distribution.
    """
    return {
        "real": real.get(key, {}),
        "synthetic": synthetic.get(key, {}),
        "jensen_shannon_divergence": round(
            jensen_shannon_divergence(real.get(key, {}), synthetic.get(key, {})), 6
        ),
        "interpretation": "lower is more similar; 0 means identical distributions",
    }


def compare_taxonomy_profiles(
    real_profile: dict[str, Any],
    synthetic_profile: dict[str, Any],
    reference_ontology_id: str | None = None,
) -> dict[str, Any]:
    """Compare one real taxonomy profile against one or more synthetic profiles."""
    sanitized_real = sanitize_taxonomy_profile(real_profile)
    sanitized_synthetic = sanitize_taxonomy_profile(synthetic_profile)
    reference = select_reference_taxonomy_profile(real_profile, reference_ontology_id)
    reference = sanitize_taxonomy_profile(
        {"taxonomies": [reference], "profile_type": "taxonomy_structure", "profile_version": 1}
    )["taxonomies"][0]

    comparisons = [
        _compare_one(reference, synthetic)
        for synthetic in sanitized_synthetic.get("taxonomies", [])
    ]
    return {
        "profile_type": "taxonomy_structural_fidelity",
        "profile_version": 1,
        "real_profile": reference,
        "synthetic_profiles": sanitized_synthetic.get("taxonomies", []),
        "comparisons": comparisons,
        "aggregate": aggregate_comparisons(comparisons),
        "notes": {
            "scalar_relative_difference": (
                "absolute_difference / abs(real); null when the real value is zero "
                "and the synthetic value differs"
            ),
            "distribution_distance": (
                "Jensen-Shannon divergence over normalized count distributions; "
                "lower is more similar"
            ),
        },
    }


def _compare_one(real: dict[str, Any], synthetic: dict[str, Any]) -> dict[str, Any]:
    return {
        "scalar_characteristics": {
            key: compare_scalar(real, synthetic, key) for key in SCALAR_KEYS
        },
        "distribution_characteristics": {
            key: compare_distribution(real, synthetic, key) for key in DISTRIBUTION_KEYS
        },
        "validation": {
            "real": real.get("validation", {}),
            "synthetic": synthetic.get("validation", {}),
            "real_has_cycle": real.get("has_cycle"),
            "synthetic_has_cycle": synthetic.get("has_cycle"),
        },
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(float(mean(values)), 4),
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
    }


def aggregate_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fidelity metrics across independent synthetic taxonomies."""
    scalar_summary = {}
    for key in SCALAR_KEYS:
        values = [
            c["scalar_characteristics"][key]["absolute_difference"]
            for c in comparisons
            if c["scalar_characteristics"][key]["absolute_difference"] is not None
        ]
        rel_values = [
            c["scalar_characteristics"][key]["relative_difference"]
            for c in comparisons
            if c["scalar_characteristics"][key]["relative_difference"] is not None
        ]
        synthetic_values = [
            c["scalar_characteristics"][key]["synthetic"]
            for c in comparisons
            if c["scalar_characteristics"][key]["synthetic"] is not None
        ]
        scalar_summary[key] = {
            "synthetic": _summary([float(v) for v in synthetic_values]),
            "absolute_difference": _summary([float(v) for v in values]),
            "relative_difference": _summary([float(v) for v in rel_values]),
        }

    distribution_summary = {}
    for key in DISTRIBUTION_KEYS:
        divergences = [
            c["distribution_characteristics"][key]["jensen_shannon_divergence"]
            for c in comparisons
        ]
        distribution_summary[key] = {
            "jensen_shannon_divergence": _summary([float(v) for v in divergences])
        }

    return {
        "n_synthetic_taxonomies": len(comparisons),
        "scalar_characteristics": scalar_summary,
        "distribution_characteristics": distribution_summary,
    }


def build_generation_feedback(
    reference: dict[str, Any],
    synthetic: dict[str, Any],
    comparison: dict[str, Any],
    tolerances: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic structural feedback for one generated taxonomy.

    Feedback contains only aggregate structural numbers and directions. It does
    not include class identifiers, edges, URI provenance, roots/leaves by name,
    examples, or metadata.
    """
    tol = feedback_tolerances(tolerances)
    adjustments: list[dict[str, Any]] = []

    scalar = comparison.get("scalar_characteristics") or {}
    for key in ("n_classes", "n_subclass_axioms", "n_roots", "n_leaves"):
        item = scalar.get(key) or {}
        _maybe_add_count_adjustment(adjustments, key, item, tol["count_relative"])

    for key in ("max_depth", "mean_depth"):
        item = scalar.get(key) or {}
        _maybe_add_depth_adjustment(adjustments, key, item, tol["depth_absolute"])

    _maybe_add_rate_adjustment(
        adjustments,
        "multiple_parent_fraction",
        (scalar.get("multiple_parent_fraction") or {}),
        tol["rate_absolute"],
    )

    distributions = comparison.get("distribution_characteristics") or {}
    for key in DISTRIBUTION_KEYS:
        item = distributions.get(key) or {}
        divergence = item.get("jensen_shannon_divergence")
        if isinstance(divergence, (int, float)) and divergence > tol["distribution_jsd"]:
            adjustments.append({
                "metric": key,
                "status": "needs_adjustment",
                "direction": "match_distribution_more_closely",
                "current_jsd": round(float(divergence), 6),
                "tolerance": tol["distribution_jsd"],
                "message": _distribution_message(key),
            })

    return {
        "within_tolerance": not adjustments,
        "tolerances": tol,
        "adjustments": adjustments,
        "messages": [item["message"] for item in adjustments],
    }


def _maybe_add_count_adjustment(
    adjustments: list[dict[str, Any]],
    key: str,
    item: dict[str, Any],
    tolerance: float,
) -> None:
    real = item.get("real")
    synthetic = item.get("synthetic")
    if not isinstance(real, (int, float)) or not isinstance(synthetic, (int, float)):
        return
    diff = float(synthetic) - float(real)
    if real == 0:
        outside = diff != 0
    else:
        outside = abs(diff) / abs(float(real)) > tolerance
    if not outside:
        return
    direction = "decrease" if diff > 0 else "increase"
    messages = {
        "n_classes": {
            "increase": "The previous taxonomy had too few classes. Aim for more classes.",
            "decrease": "The previous taxonomy had too many classes. Aim for fewer classes.",
        },
        "n_subclass_axioms": {
            "increase": (
                "The previous taxonomy had too few direct subclass relations. "
                "Increase connectivity while keeping relations direct."
            ),
            "decrease": (
                "The previous taxonomy had too many direct subclass relations. "
                "Reduce connectivity while keeping relations direct."
            ),
        },
        "n_roots": {
            "increase": "The previous taxonomy had too few root classes. Aim for more roots.",
            "decrease": "The previous taxonomy had too many root classes. Aim for fewer roots.",
        },
        "n_leaves": {
            "increase": "The previous taxonomy had too few leaf classes. Aim for more leaves.",
            "decrease": "The previous taxonomy had too many leaf classes. Aim for fewer leaves.",
        },
    }
    adjustments.append({
        "metric": key,
        "status": "needs_adjustment",
        "direction": direction,
        "target": real,
        "current": synthetic,
        "tolerance": tolerance,
        "message": messages[key][direction],
    })


def _maybe_add_depth_adjustment(
    adjustments: list[dict[str, Any]],
    key: str,
    item: dict[str, Any],
    tolerance: float,
) -> None:
    real = item.get("real")
    synthetic = item.get("synthetic")
    if not isinstance(real, (int, float)) or not isinstance(synthetic, (int, float)):
        return
    diff = float(synthetic) - float(real)
    if abs(diff) <= tolerance:
        return
    direction = "decrease" if diff > 0 else "increase"
    messages = {
        "increase": "The previous hierarchy was shallower than the target. Increase hierarchical depth.",
        "decrease": "The previous hierarchy was deeper than the target. Reduce hierarchical depth.",
    }
    adjustments.append({
        "metric": key,
        "status": "needs_adjustment",
        "direction": direction,
        "target": real,
        "current": synthetic,
        "tolerance": tolerance,
        "message": messages[direction],
    })


def _maybe_add_rate_adjustment(
    adjustments: list[dict[str, Any]],
    key: str,
    item: dict[str, Any],
    tolerance: float,
) -> None:
    real = item.get("real")
    synthetic = item.get("synthetic")
    if not isinstance(real, (int, float)) or not isinstance(synthetic, (int, float)):
        return
    diff = float(synthetic) - float(real)
    if abs(diff) <= tolerance:
        return
    direction = "decrease" if diff > 0 else "increase"
    messages = {
        "increase": (
            "Multiple inheritance was below the target. Use slightly more classes "
            "with more than one direct parent."
        ),
        "decrease": (
            "Multiple inheritance was above the target. Use fewer classes with "
            "more than one direct parent."
        ),
    }
    adjustments.append({
        "metric": key,
        "status": "needs_adjustment",
        "direction": direction,
        "target": real,
        "current": synthetic,
        "tolerance": tolerance,
        "message": messages[direction],
    })


def _distribution_message(key: str) -> str:
    messages = {
        "depth_distribution": (
            "Make the hierarchy depth distribution closer to the structural target."
        ),
        "parent_count_distribution": (
            "Make the parent-count distribution closer to the structural target; "
            "adjust root frequency and multiple-parent frequency as needed."
        ),
        "child_count_distribution": (
            "Make the child-count/branching distribution closer to the structural target; "
            "adjust branching without adding cycles."
        ),
    }
    return messages[key]
