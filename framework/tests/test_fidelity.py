import unittest

from framework.profiling.fidelity import jensen_shannon_divergence as jsd


class JSDTests(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertAlmostEqual(jsd({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}), 0.0)

    def test_disjoint_is_one(self):
        self.assertAlmostEqual(jsd({"a": 1.0}, {"b": 1.0}), 1.0, places=6)

    def test_bounded_and_symmetric(self):
        p, q = {"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.8}
        d1, d2 = jsd(p, q), jsd(q, p)
        self.assertAlmostEqual(d1, d2)
        self.assertTrue(0.0 <= d1 <= 1.0)

    def test_empty_is_zero(self):
        self.assertEqual(jsd({}, {}), 0.0)


if __name__ == "__main__":
    unittest.main()
