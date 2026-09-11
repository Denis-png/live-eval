"""Prepare normalized taxonomy JSONL records from OWL/RDF ontologies.

This utility keeps ontology parsing outside the framework's generic runtime
loader. It extracts named classes, direct named rdfs:subClassOf axioms, and
the parents named by owl:equivalentClass intersection definitions, ignoring
anonymous restriction nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from rdflib import Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS

_NON_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def _local_name(uri: URIRef) -> str:
    """Return a readable local identifier for a URI."""
    text = str(uri)
    if "#" in text:
        local = text.rsplit("#", 1)[1]
    else:
        local = text.rstrip("/").rsplit("/", 1)[-1]
    local = unquote(local).strip()
    local = _NON_NAME_CHARS.sub("_", local).strip("_")
    return local or "Class"


def _stable_id(uri: URIRef, collisions: set[str]) -> str:
    """Return a deterministic readable class id, disambiguating collisions."""
    local = _local_name(uri)
    if local not in collisions:
        return local
    digest = hashlib.sha1(str(uri).encode("utf-8")).hexdigest()[:8]
    return f"{local}__{digest}"


def _class_ids(uris: set[URIRef]) -> dict[URIRef, str]:
    names = {_local_name(uri): 0 for uri in uris}
    counts = Counter(_local_name(uri) for uri in uris)
    collisions = {name for name, count in counts.items() if count > 1}
    ids = {uri: _stable_id(uri, collisions) for uri in uris}
    if len(set(ids.values())) != len(ids):
        raise ValueError("Could not build unique deterministic class identifiers.")
    return ids


def _named_class_uris(graph: Graph) -> set[URIRef]:
    classes: set[URIRef] = set()
    for class_type in (OWL.Class, RDFS.Class):
        classes.update(
            subject for subject in graph.subjects(RDF.type, class_type)
            if isinstance(subject, URIRef)
        )
    for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(child, URIRef):
            classes.add(child)
        if isinstance(parent, URIRef):
            classes.add(parent)
    classes.discard(OWL.Thing)
    classes.discard(OWL.Nothing)
    return classes


def _direct_named_subclass_axioms(graph: Graph, ids: dict[URIRef, str]) -> set[tuple[str, str]]:
    axioms: set[tuple[str, str]] = set()
    for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
            continue
        if child not in ids or parent not in ids:
            continue
        child_id = ids[child]
        parent_id = ids[parent]
        if child_id != parent_id:
            axioms.add((child_id, parent_id))
    return axioms


def _definitional_parents(graph: Graph, ids: dict[URIRef, str]) -> set[tuple[str, str]]:
    """Parents a class's own definition states: A ≡ B ⊓ X entails A ⊑ B.

    Ontologies routinely define classes rather than subclass them --
    `VegetarianPizza ≡ Pizza ⊓ ¬∃hasTopping.MeatTopping` -- and reading only
    rdfs:subClassOf leaves such a class with no parent at all, so a model that
    correctly predicts VegetarianPizza -> Pizza is scored as wrong.

    Only the NAMED members of an owl:intersectionOf are taken. Excluded:
      * anonymous members (restrictions, complements): not named classes;
      * owl:unionOf: the reverse direction -- A ≡ B ⊔ C means B ⊑ A;
      * owl:oneOf: an enumeration of individuals, with no named superclass;
      * named ≡ named: A ⊑ B and B ⊑ A, a two-cycle.
    owl:equivalentClass is symmetric, so the named class may sit on either side.
    Nested intersections are not unwrapped: incomplete, never unsound; Pizza has none.
    """
    edges: set[tuple[str, str]] = set()
    for left, _, right in graph.triples((None, OWL.equivalentClass, None)):
        for named, expr in ((left, right), (right, left)):
            if not isinstance(named, URIRef) or named not in ids:
                continue
            if isinstance(expr, URIRef):
                continue                                  # named ≡ named
            head = graph.value(expr, OWL.intersectionOf)
            if head is None:
                continue                                  # union, oneOf, complement
            for member in Collection(graph, head):
                if isinstance(member, URIRef) and member in ids:
                    child, parent = ids[named], ids[member]
                    if child != parent:
                        edges.add((child, parent))
    return edges


def _merge_without_cycles(
    base: set[tuple[str, str]], extra: set[tuple[str, str]]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Add each edge of `extra` to `base` unless it would close a cycle.

    Returns (kept, skipped). An edge already in `base` is neither: it was
    asserted, so it is not a recovered definitional edge. A cyclic record would
    be rejected downstream, so an edge that would close one is skipped and
    reported instead.
    """
    parents_of: dict[str, set[str]] = {}
    for child, parent in base:
        parents_of.setdefault(child, set()).add(parent)

    def climbs_to(start: str, target: str) -> bool:
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(parents_of.get(node, ()))
        return False

    kept: set[tuple[str, str]] = set()
    skipped: set[tuple[str, str]] = set()
    for child, parent in sorted(extra):
        if (child, parent) in base:
            continue
        if climbs_to(parent, child):       # parent already sits below child
            skipped.add((child, parent))
            continue
        kept.add((child, parent))
        parents_of.setdefault(child, set()).add(parent)
    return kept, skipped


def taxonomy_record_from_graph(
    graph: Graph,
    ontology_id: str,
    domain: str,
) -> dict[str, Any]:
    """Convert an RDF graph to one normalized taxonomy benchmark record.

    The output preserves multiple inheritance. Its subclass axioms are the
    asserted direct named rdfs:subClassOf edges PLUS the parents a class's own
    owl:equivalentClass definition names (see _definitional_parents), which are
    listed separately in metadata so the gold's provenance stays visible. It
    does not infer transitive subclass relations.
    """
    class_uris = _named_class_uris(graph)
    if not class_uris:
        raise ValueError("No usable named classes found in ontology.")

    ids = _class_ids(class_uris)
    asserted = _direct_named_subclass_axioms(graph, ids)
    definitional, skipped = _merge_without_cycles(
        asserted, _definitional_parents(graph, ids))
    axioms = asserted | definitional
    classes = sorted(ids.values())
    class_uri_map = {
        class_id: str(uri)
        for uri, class_id in sorted(ids.items(), key=lambda item: item[1])
    }

    return {
        "ontology_id": ontology_id,
        "domain": domain,
        "classes": classes,
        "subclass_axioms": [[child, parent] for child, parent in sorted(axioms)],
        "metadata": {
            "class_uri_map": class_uri_map,
            "definitional_axioms": [[c, p] for c, p in sorted(definitional)],
            "skipped_definitional_axioms": [[c, p] for c, p in sorted(skipped)],
        },
    }


def prepare_taxonomy_record(
    input_path: str | Path,
    ontology_id: str,
    domain: str,
    rdf_format: str | None = None,
) -> dict[str, Any]:
    """Parse an OWL/RDF file and return one normalized taxonomy record."""
    graph = Graph()
    graph.parse(str(input_path), format=rdf_format)
    return taxonomy_record_from_graph(graph, ontology_id=ontology_id, domain=domain)


def write_jsonl_record(record: dict[str, Any], output_path: str | Path) -> str:
    """Write one normalized taxonomy record as JSONL and return the path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OWL/RDF ontology into normalized taxonomy JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to the OWL/RDF ontology file")
    parser.add_argument("output", help="Output JSONL path")
    parser.add_argument("--ontology-id", required=True, help="Stable ontology identifier")
    parser.add_argument("--domain", required=True, help="Human-readable domain name")
    parser.add_argument("--rdf-format", help="Optional rdflib parser format, e.g. xml or turtle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = prepare_taxonomy_record(
        args.input,
        ontology_id=args.ontology_id,
        domain=args.domain,
        rdf_format=args.rdf_format,
    )
    output = write_jsonl_record(record, args.output)
    print("Taxonomy benchmark preparation summary")
    print("=" * 38)
    print(f"Ontology ID       : {record['ontology_id']}")
    print(f"Domain            : {record['domain']}")
    print(f"Classes           : {len(record['classes'])}")
    print(f"Subclass axioms   : {len(record['subclass_axioms'])}")
    print(f"Output            : {output}")


if __name__ == "__main__":
    main()
