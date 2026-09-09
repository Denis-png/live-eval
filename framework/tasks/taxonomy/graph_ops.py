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
    """Longest root-to-node edge count. Ontologies are DAGs — a class may have
    several parents — so this is a longest path over every parent, not a walk up
    a single chain. Memoised, so it stays O(V+E)."""
    parents_of: dict[str, list[str]] = {}
    for child, parent in axioms:
        parents_of.setdefault(child, []).append(parent)
    memo: dict[str, int] = {}

    def height(node, stack):
        if node in memo:
            return memo[node]
        if node in stack:            # defensive: a cycle has no finite depth
            return 0
        best = 0
        for parent in parents_of.get(node, []):
            best = max(best, 1 + height(parent, stack | {node}))
        memo[node] = best
        return best

    return max((height(c, frozenset()) for c in classes), default=0)


def sample_subtrees(classes, axioms, *, max_depth: int = 4,
                    min_classes: int = 5, rng=None) -> list[dict]:
    """Every rooted subtree of `classes`, truncated at `max_depth` edges and
    kept only when it holds at least `min_classes` classes.

    One ontology has to supply a whole seed pool, so the pool is built by
    rooting at each class rather than by taking the graph whole. Truncation is
    what creates structural VARIATION across seeds: without it every subtree of
    a shallow ontology would have the same depth, seed choice would carry no
    information, and the seeded cells would have no control input at all.

    `max_depth` bounds how far the traversal DISCOVERS from the root, not the
    depth of the graph it returns. The result is an INDUCED subgraph — every
    axiom between kept classes is retained — so on a DAG a class discovered via
    a short path may also sit on a longer one, and the returned "max_depth"
    field (the true depth of the returned axioms) can exceed this parameter.
    Dropping those edges to force the bound would make the seed stop being a
    subgraph of the real ontology, which is the worse failure: these seeds exist
    to perturb a REAL artifact.

    `rng` is accepted for signature compatibility; enumeration is exhaustive and
    ordered, so the pool does not depend on it.

    Returns [] rather than raising when nothing qualifies — callers own the
    framework's error vocabulary, this layer does not.
    """
    rng = rng or random.Random()
    children = _children_index(axioms)
    out = []
    for root in sorted(classes):
        kept, frontier, seen = [root], [(root, 0)], {root}
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for kid in children.get(node, []):
                if kid in seen:
                    continue
                seen.add(kid)
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
