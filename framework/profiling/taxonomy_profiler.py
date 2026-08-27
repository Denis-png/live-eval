"""Structural profiling for normalized taxonomy benchmark records.

Expected input rows are dictionaries shaped like:

{
  "ontology_id": "example",
  "domain": "pizza",
  "classes": ["Pizza", "VegetarianPizza"],
  "subclass_axioms": [["VegetarianPizza", "Pizza"]]
}

Depth is defined over direct subclass edges as the longest distance from any
root class, where roots are classes with no valid named parent. For a forest,
each root has depth 0. Multiple inheritance is valid; a child with several
parents receives the maximum parent depth plus one. Cyclic graphs are reported
with has_cycle=true and depth values are set to None to avoid misleading stats.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from statistics import mean
from typing import Any, Iterable


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _mean(values: list[int]) -> float:
    return round(float(mean(values)), 4) if values else 0.0


def _normalized_axioms(row: dict[str, Any]) -> list[tuple[str, str]]:
    axioms = []
    for axiom in row.get("subclass_axioms") or []:
        if isinstance(axiom, (list, tuple)) and len(axiom) == 2:
            child, parent = axiom
            axioms.append((str(child), str(parent)))
    return axioms


def _has_cycle(classes: set[str], children_by_parent: dict[str, set[str]]) -> bool:
    state = {name: 0 for name in classes}

    def visit(node: str) -> bool:
        state[node] = 1
        for child in children_by_parent.get(node, set()):
            if state[child] == 1:
                return True
            if state[child] == 0 and visit(child):
                return True
        state[node] = 2
        return False

    return any(state[name] == 0 and visit(name) for name in sorted(classes))


def _depths_for_dag(
    classes: set[str],
    parents_by_child: dict[str, set[str]],
    children_by_parent: dict[str, set[str]],
) -> dict[str, int]:
    """Return longest-root-distance depths for an acyclic taxonomy forest."""
    indegree = {name: len(parents_by_child.get(name, set())) for name in classes}
    depths = {name: 0 for name in classes if indegree[name] == 0}
    queue = deque(sorted(depths))

    while queue:
        parent = queue.popleft()
        for child in sorted(children_by_parent.get(parent, set())):
            depths[child] = max(depths.get(child, 0), depths[parent] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return depths


def profile_taxonomy(row: dict[str, Any]) -> dict[str, Any]:
    """Compute structural statistics for one normalized taxonomy.

    Unknown-class edges reference a class that is absent from the row's classes
    list. Self-loops are invalid direct subclass axioms. Both are excluded from
    root, leaf, branching, parent-count, cycle, and depth calculations.
    """
    classes = {str(name) for name in row.get("classes") or []}
    raw_axioms = _normalized_axioms(row)
    unique_axioms = sorted(set(raw_axioms))

    unknown_edges = [
        (child, parent)
        for child, parent in unique_axioms
        if child not in classes or parent not in classes
    ]
    self_loops = [
        (child, parent)
        for child, parent in unique_axioms
        if child == parent and child in classes
    ]
    valid_axioms = [
        (child, parent)
        for child, parent in unique_axioms
        if child in classes and parent in classes and child != parent
    ]

    parents_by_child: dict[str, set[str]] = defaultdict(set)
    children_by_parent: dict[str, set[str]] = defaultdict(set)
    for child, parent in valid_axioms:
        parents_by_child[child].add(parent)
        children_by_parent[parent].add(child)

    parent_counts = [len(parents_by_child.get(name, set())) for name in classes]
    child_counts = [len(children_by_parent.get(name, set())) for name in classes]
    roots = sorted(name for name in classes if not parents_by_child.get(name))
    leaves = sorted(name for name in classes if not children_by_parent.get(name))
    has_cycle = _has_cycle(classes, children_by_parent) if classes else False

    if has_cycle:
        depths = {}
        max_depth = None
        mean_depth = None
        depth_distribution = {}
    else:
        depths = _depths_for_dag(classes, parents_by_child, children_by_parent)
        depth_values = list(depths.values())
        max_depth = max(depth_values) if depth_values else 0
        mean_depth = _mean(depth_values)
        depth_distribution = _distribution(depth_values)

    multiple_parent_count = sum(1 for count in parent_counts if count > 1)

    return {
        "ontology_id": row.get("ontology_id"),
        "domain": row.get("domain"),
        "n_classes": len(classes),
        "n_subclass_axioms": len(valid_axioms),
        "n_roots": len(roots),
        "n_leaves": len(leaves),
        "roots": roots,
        "leaves": leaves,
        "max_depth": max_depth,
        "mean_depth": mean_depth,
        "depth_distribution": depth_distribution,
        "parent_count_distribution": _distribution(parent_counts),
        "child_count_distribution": _distribution(child_counts),
        "multiple_parent_fraction": (
            round(multiple_parent_count / len(classes), 4) if classes else 0.0
        ),
        "has_cycle": has_cycle,
        "validation": {
            "unknown_class_edges": len(unknown_edges),
            "self_loops": len(self_loops),
            "duplicate_subclass_axioms": len(raw_axioms) - len(unique_axioms),
        },
        "class_depths": dict(sorted(depths.items())),
    }


def profile_taxonomy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile a collection of normalized taxonomy records."""
    taxonomies = [profile_taxonomy(row) for row in rows]
    return {
        "profile_type": "taxonomy_structure",
        "profile_version": 1,
        "num_taxonomies": len(taxonomies),
        "taxonomies": taxonomies,
        "summary": {
            "n_classes": {
                "min": min((t["n_classes"] for t in taxonomies), default=0),
                "max": max((t["n_classes"] for t in taxonomies), default=0),
                "mean": _mean([t["n_classes"] for t in taxonomies]),
            },
            "n_subclass_axioms": {
                "min": min((t["n_subclass_axioms"] for t in taxonomies), default=0),
                "max": max((t["n_subclass_axioms"] for t in taxonomies), default=0),
                "mean": _mean([t["n_subclass_axioms"] for t in taxonomies]),
            },
            "cycles": sum(1 for t in taxonomies if t["has_cycle"]),
            "unknown_class_edges": sum(
                t["validation"]["unknown_class_edges"] for t in taxonomies
            ),
            "self_loops": sum(t["validation"]["self_loops"] for t in taxonomies),
        },
    }
