import unittest

from framework.profiling.taxonomy_profiler import profile_taxonomy, profile_taxonomy_rows


class TaxonomyProfilerTests(unittest.TestCase):
    def test_roots_leaves_depth_and_distributions(self):
        profile = profile_taxonomy({
            "ontology_id": "shape",
            "domain": "demo",
            "classes": ["A", "B", "C", "D"],
            "subclass_axioms": [
                ["B", "A"],
                ["C", "A"],
                ["D", "B"],
                ["D", "C"],
            ],
        })

        self.assertEqual(profile["n_classes"], 4)
        self.assertEqual(profile["n_subclass_axioms"], 4)
        self.assertEqual(profile["roots"], ["A"])
        self.assertEqual(profile["leaves"], ["D"])
        self.assertEqual(profile["max_depth"], 2)
        self.assertEqual(profile["mean_depth"], 1.0)
        self.assertEqual(profile["depth_distribution"], {"0": 1, "1": 2, "2": 1})
        self.assertEqual(profile["parent_count_distribution"], {"0": 1, "1": 2, "2": 1})
        self.assertEqual(profile["child_count_distribution"], {"0": 1, "1": 2, "2": 1})
        self.assertEqual(profile["multiple_parent_fraction"], 0.25)
        self.assertFalse(profile["has_cycle"])

    def test_validation_counts_and_duplicate_handling(self):
        profile = profile_taxonomy({
            "ontology_id": "invalid",
            "domain": "demo",
            "classes": ["A", "B"],
            "subclass_axioms": [
                ["B", "A"],
                ["B", "A"],
                ["A", "A"],
                ["B", "Missing"],
            ],
        })

        self.assertEqual(profile["n_subclass_axioms"], 1)
        self.assertEqual(profile["validation"]["duplicate_subclass_axioms"], 1)
        self.assertEqual(profile["validation"]["self_loops"], 1)
        self.assertEqual(profile["validation"]["unknown_class_edges"], 1)

    def test_cycle_detection_avoids_depth_stats(self):
        profile = profile_taxonomy({
            "ontology_id": "cycle",
            "domain": "demo",
            "classes": ["A", "B"],
            "subclass_axioms": [["A", "B"], ["B", "A"]],
        })

        self.assertTrue(profile["has_cycle"])
        self.assertIsNone(profile["max_depth"])
        self.assertIsNone(profile["mean_depth"])
        self.assertEqual(profile["depth_distribution"], {})
        self.assertEqual(profile["class_depths"], {})

    def test_collection_summary(self):
        profile = profile_taxonomy_rows([
            {
                "ontology_id": "one",
                "domain": "demo",
                "classes": ["A", "B"],
                "subclass_axioms": [["B", "A"]],
            },
            {
                "ontology_id": "two",
                "domain": "demo",
                "classes": ["A", "B"],
                "subclass_axioms": [["A", "B"], ["B", "A"]],
            },
        ])

        self.assertEqual(profile["profile_type"], "taxonomy_structure")
        self.assertEqual(profile["num_taxonomies"], 2)
        self.assertEqual(profile["summary"]["n_classes"]["mean"], 2.0)
        self.assertEqual(profile["summary"]["cycles"], 1)


if __name__ == "__main__":
    unittest.main()
