"""Pure structural operations on a taxonomy graph.

Every function here takes and returns `(classes, subclass_axioms)` and nothing
else — no domain, no config, no I/O, no LLM. The seeded generation cells compute
their ground truth with these functions and only ever COMPARE model output to
that result, so a wrong answer here is a wrong benchmark, not a failed sample.

An axiom is `[child, parent]`, matching TaxonomyTask.parse_row.
"""
from __future__ import annotations

import random


def _children_index(axioms) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for child, parent in axioms:
        index.setdefault(parent, []).append(child)
    for kids in index.values():
        kids.sort()
    return index


def _depth_of(classes, axioms) -> int:
    """Longest root-to-leaf edge count. 0 for a single class or a forest of them."""
    parents = {child: parent for child, parent in axioms}
    best = 0
    for cls in classes:
        depth, cur, seen = 0, cls, set()
        while cur in parents and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
            depth += 1
        best = max(best, depth)
    return best


def sample_subtrees(classes, axioms, *, max_depth: int = 4,
                    min_classes: int = 5, rng=None) -> list[dict]:
    """Every rooted subtree of `classes`, truncated at `max_depth` edges and
    kept only when it holds at least `min_classes` classes.

    One ontology has to supply a whole seed pool, so the pool is built by
    rooting at each class rather than by taking the graph whole. Truncation is
    what creates structural VARIATION across seeds: without it every subtree of
    a shallow ontology would have the same depth, seed choice would carry no
    information, and the seeded cells would have no control input at all.

    Returns [] rather than raising when nothing qualifies — callers own the
    framework's error vocabulary, this layer does not.
    """
    rng = rng or random.Random()
    children = _children_index(axioms)
    out = []
    for root in sorted(classes):
        kept, frontier = [root], [(root, 0)]
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for kid in children.get(node, []):
                kept.append(kid)
                frontier.append((kid, depth + 1))
        if len(kept) < min_classes:
            continue
        names = set(kept)
        sub_axioms = sorted([c, p] for c, p in axioms if c in names and p in names)
        out.append({
            "classes": sorted(kept),
            "subclass_axioms": sub_axioms,
            "root": root,
            "max_depth": _depth_of(kept, sub_axioms),
        })
    return out
