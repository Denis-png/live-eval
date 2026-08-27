import os
import tempfile
import textwrap
import unittest

from scripts.prepare_taxonomy_benchmark import prepare_taxonomy_record


TINY_TTL = textwrap.dedent("""\
    @prefix ex: <http://example.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    ex:Thing a owl:Class .
    ex:Food a owl:Class .
    ex:Pizza a owl:Class ;
        rdfs:subClassOf ex:Food .
    ex:VegetarianPizza a owl:Class ;
        rdfs:subClassOf ex:Pizza ,
                        ex:VegetarianFood ,
                        [ a owl:Restriction ;
                          owl:onProperty ex:hasTopping ;
                          owl:someValuesFrom ex:Vegetable ] .
    ex:VegetarianFood a owl:Class ;
        rdfs:subClassOf ex:Food .
    ex:MargheritaPizza a owl:Class ;
        rdfs:subClassOf ex:VegetarianPizza .
    ex:VegetarianPizza rdfs:subClassOf ex:Pizza .
""")


COLLISION_TTL = textwrap.dedent("""\
    @prefix ex1: <http://example.org/one/> .
    @prefix ex2: <http://example.org/two/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    ex1:Class a owl:Class .
    ex2:Class a owl:Class .
    ex1:Child a owl:Class ;
        rdfs:subClassOf ex1:Class .
    ex2:Child a owl:Class ;
        rdfs:subClassOf ex2:Class .
""")


def _write(content):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ttl", delete=False, encoding="utf-8")
    try:
        tmp.write(content)
        return tmp.name
    finally:
        tmp.close()


class TaxonomyPreparationTests(unittest.TestCase):
    def test_extracts_named_classes_and_direct_subclass_axioms(self):
        path = _write(TINY_TTL)
        try:
            record = prepare_taxonomy_record(
                path, ontology_id="tiny", domain="food", rdf_format="turtle"
            )
        finally:
            os.unlink(path)

        self.assertEqual(record["ontology_id"], "tiny")
        self.assertEqual(record["domain"], "food")
        self.assertIn("Pizza", record["classes"])
        self.assertIn(["Pizza", "Food"], record["subclass_axioms"])
        self.assertIn(["MargheritaPizza", "VegetarianPizza"], record["subclass_axioms"])

    def test_preserves_multiple_inheritance(self):
        path = _write(TINY_TTL)
        try:
            record = prepare_taxonomy_record(
                path, ontology_id="tiny", domain="food", rdf_format="turtle"
            )
        finally:
            os.unlink(path)

        self.assertIn(["VegetarianPizza", "Pizza"], record["subclass_axioms"])
        self.assertIn(["VegetarianPizza", "VegetarianFood"], record["subclass_axioms"])

    def test_ignores_anonymous_restrictions_and_deduplicates_relations(self):
        path = _write(TINY_TTL)
        try:
            record = prepare_taxonomy_record(
                path, ontology_id="tiny", domain="food", rdf_format="turtle"
            )
        finally:
            os.unlink(path)

        flattened = [value for axiom in record["subclass_axioms"] for value in axiom]
        self.assertNotIn("Restriction", flattened)
        self.assertEqual(
            record["subclass_axioms"].count(["VegetarianPizza", "Pizza"]),
            1,
        )

    def test_disambiguates_local_name_collisions_with_uri_provenance(self):
        path = _write(COLLISION_TTL)
        try:
            record = prepare_taxonomy_record(
                path, ontology_id="collision", domain="test", rdf_format="turtle"
            )
        finally:
            os.unlink(path)

        class_names = record["classes"]
        self.assertEqual(len(class_names), len(set(class_names)))
        self.assertTrue(any(name.startswith("Class__") for name in class_names))
        self.assertTrue(any(name.startswith("Child__") for name in class_names))
        self.assertEqual(len(record["metadata"]["class_uri_map"]), len(class_names))

    def test_fails_clearly_when_no_usable_classes_found(self):
        path = _write("@prefix ex: <http://example.org/> . ex:a ex:b ex:c .\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                prepare_taxonomy_record(
                    path, ontology_id="empty", domain="empty", rdf_format="turtle"
                )
        finally:
            os.unlink(path)
        self.assertIn("No usable named classes", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
