"""Parents a class's own OWL definition states must reach the gold.

Fourteen of Pizza's 99 classes had no parent in the benchmark. They are DEFINED
classes -- VegetarianPizza ≡ Pizza ⊓ ¬∃hasTopping.MeatTopping -- and the prep
script read only rdfs:subClassOf. A model that correctly predicted
VegetarianPizza -> Pizza was scored as a false positive; on the real benchmark
8 of the lexical baseline's 10 false positives were exactly that.

A ≡ B ⊓ X entails A ⊑ B, so each NAMED conjunct is a parent. Every other shape
of definition is excluded, and each exclusion below has a reason.
"""
import os
import tempfile
import textwrap
import unittest

from scripts.prepare_taxonomy_benchmark import prepare_taxonomy_record

_PREFIXES = textwrap.dedent("""\
    @prefix ex: <http://example.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
""")


def _record(body):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ttl", delete=False, encoding="utf-8")
    try:
        tmp.write(_PREFIXES + textwrap.dedent(body))
        tmp.close()
        return prepare_taxonomy_record(tmp.name, ontology_id="t", domain="d",
                                       rdf_format="turtle")
    finally:
        os.unlink(tmp.name)


def _edges(record):
    return {tuple(a) for a in record["subclass_axioms"]}


_DEFINED = """\
    ex:Food a owl:Class .
    ex:Pizza a owl:Class ; rdfs:subClassOf ex:Food .
    ex:Topping a owl:Class .
    ex:VegPizza a owl:Class ;
        owl:equivalentClass [ a owl:Class ;
            owl:intersectionOf ( ex:Pizza
                                 [ a owl:Restriction ; owl:onProperty ex:has ;
                                   owl:someValuesFrom ex:Topping ] ) ] .
"""


class RecoveredParentTests(unittest.TestCase):
    def test_the_named_conjunct_of_a_definition_becomes_a_parent(self):
        self.assertIn(("VegPizza", "Pizza"), _edges(_record(_DEFINED)))

    def test_a_restriction_member_never_becomes_a_parent(self):
        # Topping is the restriction's FILLER (∃has.Topping) -- the class a
        # wrong implementation would take. VegPizza ⊑ ∃has.Topping does not
        # make VegPizza a Topping. (This used to check that "Restriction" was
        # not a class name, which it never is, so it could not fail.)
        self.assertNotIn(("VegPizza", "Topping"), _edges(_record(_DEFINED)))

    def test_recovered_edges_are_listed_as_provenance(self):
        record = _record(_DEFINED)
        self.assertEqual(record["metadata"]["definitional_axioms"], [["VegPizza", "Pizza"]])
        self.assertEqual(record["metadata"]["skipped_definitional_axioms"], [])

    def test_the_intersection_may_be_on_either_side(self):
        # owl:equivalentClass is symmetric.
        record = _record("""\
            ex:Pizza a owl:Class .
            ex:Hot a owl:Class .
            ex:HotPizza a owl:Class .
            [ a owl:Class ; owl:intersectionOf ( ex:Pizza ex:Hot ) ]
                owl:equivalentClass ex:HotPizza .
        """)
        self.assertTrue({("HotPizza", "Pizza"), ("HotPizza", "Hot")} <= _edges(record))


class ExcludedDefinitionTests(unittest.TestCase):
    def test_a_complemented_member_is_not_a_parent(self):
        # NonVegPizza ≡ Pizza ⊓ ¬VegPizza. The complement is the OPPOSITE of a
        # parent -- and the lexical baseline predicts exactly this wrong edge.
        record = _record(_DEFINED + """\
            ex:NonVegPizza a owl:Class ;
                owl:equivalentClass [ a owl:Class ;
                    owl:intersectionOf ( ex:Pizza [ owl:complementOf ex:VegPizza ] ) ] .
        """)
        self.assertIn(("NonVegPizza", "Pizza"), _edges(record))
        self.assertNotIn(("NonVegPizza", "VegPizza"), _edges(record))

    def test_a_union_is_not_read_backwards(self):
        # Dish ≡ Pizza ⊔ Pasta means Pizza ⊑ Dish -- NOT Dish ⊑ Pizza.
        record = _record("""\
            ex:Pizza a owl:Class .
            ex:Pasta a owl:Class .
            ex:Dish a owl:Class ;
                owl:equivalentClass [ a owl:Class ; owl:unionOf ( ex:Pizza ex:Pasta ) ] .
        """)
        self.assertNotIn(("Dish", "Pizza"), _edges(record))
        self.assertNotIn(("Dish", "Pasta"), _edges(record))

    def test_an_enumeration_has_no_named_parent(self):
        # Italy and France are declared classes so that a wrong reading of the
        # oneOf list WOULD emit an edge: undeclared, they are not in the class
        # index, and even an implementation reading oneOf members as parents
        # emitted nothing -- the test could not fail.
        record = _record("""\
            ex:Italy a owl:Class .
            ex:France a owl:Class .
            ex:Country a owl:Class ;
                owl:equivalentClass [ a owl:Class ; owl:oneOf ( ex:Italy ex:France ) ] .
        """)
        self.assertNotIn(("Country", "Italy"), _edges(record))
        self.assertNotIn(("Country", "France"), _edges(record))
        self.assertEqual(record["metadata"]["definitional_axioms"], [])

    def test_a_named_to_named_equivalence_adds_no_edge(self):
        # Pie ≡ Pizza means Pie ⊑ Pizza AND Pizza ⊑ Pie: a two-cycle.
        record = _record("""\
            ex:Pizza a owl:Class .
            ex:Pie a owl:Class ; owl:equivalentClass ex:Pizza .
        """)
        self.assertNotIn(("Pie", "Pizza"), _edges(record))
        self.assertNotIn(("Pizza", "Pie"), _edges(record))

    def test_an_edge_that_would_close_a_cycle_is_skipped_and_recorded(self):
        # A ≡ B ⊓ R gives A ⊑ B, but B ⊑ A is already asserted.
        record = _record("""\
            ex:A a owl:Class .
            ex:B a owl:Class ; rdfs:subClassOf ex:A .
            ex:A owl:equivalentClass [ a owl:Class ;
                owl:intersectionOf ( ex:B [ a owl:Restriction ; owl:onProperty ex:p ;
                                            owl:someValuesFrom ex:B ] ) ] .
        """)
        self.assertNotIn(("A", "B"), _edges(record))
        self.assertEqual(record["metadata"]["skipped_definitional_axioms"], [["A", "B"]])

    def test_an_edge_already_asserted_is_not_listed_as_definitional(self):
        record = _record("""\
            ex:Pizza a owl:Class .
            ex:Veg a owl:Class ; rdfs:subClassOf ex:Pizza ;
                owl:equivalentClass [ a owl:Class ;
                    owl:intersectionOf ( ex:Pizza [ a owl:Restriction ;
                        owl:onProperty ex:p ; owl:someValuesFrom ex:Pizza ] ) ] .
        """)
        self.assertEqual(record["metadata"]["definitional_axioms"], [])
        self.assertEqual(sorted(record["subclass_axioms"]).count(["Veg", "Pizza"]), 1)


class NoDefinitionsTests(unittest.TestCase):
    def test_an_ontology_without_definitions_is_unchanged(self):
        record = _record("""\
            ex:Food a owl:Class .
            ex:Pizza a owl:Class ; rdfs:subClassOf ex:Food .
        """)
        self.assertEqual(record["subclass_axioms"], [["Pizza", "Food"]])
        self.assertEqual(record["metadata"]["definitional_axioms"], [])


if __name__ == "__main__":
    unittest.main()
