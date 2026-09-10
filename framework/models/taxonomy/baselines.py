"""Deterministic taxonomy baselines.

Neither calls a model, so both are free and exactly reproducible. More
importantly for a GET benchmark, neither can MEMORISE: an LLM may score on a
canonical ontology by recalling it, while these read nothing but class names.
That makes them a control for the real-vs-synthetic comparison -- a gap that
opens for the LLMs but not for these points at recall, not induction.

Both emit the same payload as TaxonomyLLMModel and reuse its input parser and
payload builder, so the scorer cannot tell them apart from an LLM.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from framework.evaluators.taxonomy.metrics import parse_prediction_relations
from framework.models.base_model import BaseModel
from framework.models.taxonomy.llm import TaxonomyLLMModel

# Split CamelCase (keeping acronyms whole: "HTTPServer" -> HTTP, Server) and any
# run of non-alphanumerics. Re-verbalised synthetic taxonomies are named by a
# model rather than an ontology engineer, so spaces and underscores matter as
# much as CamelCase.
_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def name_tokens(name: str) -> list[str]:
    """Lowercase word tokens of a class name."""
    return [t.lower() for t in _TOKEN_RE.findall(name or "")]


def lexical_parents(classes) -> dict[str, str]:
    """Map each class to the class whose name is its longest proper token-suffix.

    `VegetarianPizza` -> `Pizza`. The LONGEST matching suffix wins, because the
    most specific named ancestor is the direct parent and a shorter suffix is a
    grandparent (a false positive for a direct-subclass metric). A class with no
    matching suffix is left unattached rather than guessed -- named pizzas like
    `Margherita` do not end in `Pizza`, and that limit is precisely what an LLM
    is measured against.

    Ties on suffix length break alphabetically, so the result does not depend on
    input order.
    """
    by_tokens: dict[tuple[str, ...], str] = {}
    for name in sorted(classes):
        by_tokens.setdefault(tuple(name_tokens(name)), name)

    parents: dict[str, str] = {}
    for name in sorted(classes):
        tokens = name_tokens(name)
        for start in range(1, len(tokens)):          # proper suffixes, longest first
            candidate = by_tokens.get(tuple(tokens[start:]))
            if candidate is not None and candidate != name:
                parents[name] = candidate
                break
    return parents


def _star_root(classes) -> str:
    """The class that is the lexical head of the most others; alphabetical-first
    when names encode no hierarchy at all. Chosen from names alone, so the floor
    never peeks at gold."""
    heads = Counter(lexical_parents(classes).values())
    if heads:
        best = max(heads.values())
        return sorted(c for c, n in heads.items() if n == best)[0]
    return sorted(classes)[0]


class _BaselineModel(BaseModel):
    """Shared plumbing: parse the model input, emit an LLM-shaped payload."""

    def load_model(self, model_config: dict):
        self.model_name = model_config["name"]

    def predict(self, texts: list[str]) -> list[dict]:
        return [self._predict_one(text) for text in texts]

    def _predict_one(self, text: str) -> dict:
        payload = TaxonomyLLMModel._parse_model_input(text)
        axioms = self._axioms(payload["classes"])
        # Serialised and re-parsed on purpose: the baseline's output goes through
        # the exact validation an LLM's does, so a baseline bug would surface as
        # an invalid relation rather than slip past the scorer.
        raw = json.dumps({"subclass_axioms": axioms})
        parsed = parse_prediction_relations(raw, payload["classes"])
        return TaxonomyLLMModel._prediction_payload(raw, parsed)

    def _axioms(self, classes: list[str]) -> list[list[str]]:
        raise NotImplementedError


class LexicalHeadMatchModel(_BaselineModel):
    """Parent = the class whose name is the longest proper token-suffix."""

    def _axioms(self, classes):
        return sorted([child, parent] for child, parent in lexical_parents(classes).items())


class StarModel(_BaselineModel):
    """Every class attached to one root: the standard taxonomy-induction floor.

    It tells you how much F1 is achievable by imposing no structure at all, which
    is the reference any other model's score has to be read against.
    """

    def _axioms(self, classes):
        if len(classes) < 2:
            return []
        root = _star_root(classes)
        return sorted([c, root] for c in classes if c != root)
