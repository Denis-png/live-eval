import unittest
from unittest import mock

from framework.profiling.syntax_stats import syntax_profile


class FakeToken:
    def __init__(self, pos, dep, head=None):
        self.pos_ = pos
        self.dep_ = dep
        self.head = head if head is not None else self


class FakeDoc:
    def __init__(self, tokens, n_sents=1):
        self._tokens = tokens
        self.sents = [object()] * n_sents

    def __iter__(self):
        return iter(self._tokens)


class FakeNLP:
    def __init__(self, docs):
        self._docs = docs

    def pipe(self, texts):
        return iter(self._docs)


class FreshToken:
    """Token that mimics real spaCy: .head returns fresh objects on every access."""
    def __init__(self, i, pos, dep, head_i, doc):
        self.i = i
        self.pos_ = pos
        self.dep_ = dep
        self.head_i = head_i
        self.doc = doc

    @property
    def head(self):
        # Returns a fresh object every time (never cached)
        return self.doc[self.head_i]

    def __eq__(self, other):
        if not isinstance(other, FreshToken):
            return False
        # Compare by doc identity and index (not object identity)
        return self.doc is other.doc and self.i == other.i


class FreshDoc:
    """Doc whose __getitem__ builds fresh token instances on every call."""
    def __init__(self, specs):
        """specs: list of (pos, dep, head_i) tuples defining token properties."""
        self.specs = specs
        self.sents = [object()]

    def __iter__(self):
        return (FreshToken(i, pos, dep, head_i, self)
                for i, (pos, dep, head_i) in enumerate(self.specs))

    def __getitem__(self, idx):
        # Always construct a fresh token instance
        pos, dep, head_i = self.specs[idx]
        return FreshToken(idx, pos, dep, head_i, self)


def _simple_doc():
    root = FakeToken("VERB", "ROOT")            # depth 0, clause
    noun = FakeToken("NOUN", "nsubj", head=root)  # depth 1
    return FakeDoc([root, noun])


class SyntaxProfileTests(unittest.TestCase):
    def test_pos_dist_depth_and_clauses(self):
        profile = syntax_profile(["she runs"], nlp=FakeNLP([_simple_doc()]))
        self.assertEqual(profile["n_texts"], 1)
        self.assertAlmostEqual(profile["pos_dist"]["VERB"], 0.5)
        self.assertAlmostEqual(profile["pos_dist"]["NOUN"], 0.5)
        self.assertEqual(profile["parse_depth"]["max"], 1)
        self.assertEqual(profile["clauses_per_text"]["mean"], 1.0)
        self.assertEqual(profile["sentences_per_text"]["mean"], 1.0)

    def test_conj_verb_counts_as_clause(self):
        root = FakeToken("VERB", "ROOT")
        conj = FakeToken("VERB", "conj", head=root)
        profile = syntax_profile(["run and jump"], nlp=FakeNLP([FakeDoc([root, conj])]))
        self.assertEqual(profile["clauses_per_text"]["mean"], 2.0)

    def test_space_tokens_excluded_from_pos(self):
        root = FakeToken("VERB", "ROOT")
        space = FakeToken("SPACE", "dep", head=root)
        profile = syntax_profile(["run  "], nlp=FakeNLP([FakeDoc([root, space])]))
        self.assertEqual(set(profile["pos_dist"]), {"VERB"})

    def test_returns_none_when_no_nlp_available(self):
        with mock.patch(
            "framework.profiling.syntax_stats._load_nlp", return_value=None
        ):
            self.assertIsNone(syntax_profile(["hello"]))

    def test_token_depth_with_fresh_head_objects(self):
        """Regression: _token_depth must use equality (!=) not identity (is not).

        Real spaCy returns a fresh Token object on each .head access (never
        identity-equal). Old code with 'is not' would loop forever; new code
        with '!=' correctly uses __eq__ to detect the root.
        """
        # Create doc: root at i=0 (head_i=0), child at i=1 (head_i=0)
        doc = FreshDoc([
            ("VERB", "ROOT", 0),   # i=0: root token pointing to itself
            ("NOUN", "nsubj", 0),  # i=1: child token pointing to root
        ])

        # Sanity check: verify FreshToken mimics spaCy's fresh-object semantics
        token0 = doc[0]
        token0_head1 = token0.head
        token0_head2 = token0.head
        self.assertIsNot(token0_head1, token0_head2,
                         "Multiple .head accesses must return distinct objects")
        self.assertEqual(token0_head1, token0_head2,
                         "But they must compare equal via __eq__")

        # Now test that syntax_profile completes without hanging
        profile = syntax_profile(["verb noun"], nlp=FakeNLP([doc]))

        # Verify correctness
        self.assertIsNotNone(profile)
        self.assertEqual(profile["parse_depth"]["max"], 1)


if __name__ == "__main__":
    unittest.main()
