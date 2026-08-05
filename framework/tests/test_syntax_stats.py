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


class SpacyLikeToken:
    """Token that mimics real spaCy: returns fresh object on each .head access."""
    def __init__(self, i, pos, dep, head_i, doc=None):
        self.i = i
        self.pos_ = pos
        self.dep_ = dep
        self.head_i = head_i
        self.doc = doc

    @property
    def head(self):
        if self.doc is None:
            # Self-rooting fallback
            return SpacyLikeToken(self.i, self.pos_, self.dep_, self.i, None)
        return self.doc[self.head_i]

    def __eq__(self, other):
        if not isinstance(other, SpacyLikeToken):
            return False
        return self.i == other.i and self.head_i == other.head_i

    def __getitem__(self, idx):
        # Allow doc[idx] access; return self if idx matches, else placeholder
        return self if idx == self.i else SpacyLikeToken(idx, "NOUN", "dep", idx, None)


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

        Real spaCy returns a fresh Token object on each .head access.
        Old code with 'is not' would loop forever; new code with '!=' works.
        """
        # Create a simple doc: root (i=0, head_i=0) and child (i=1, head_i=0)
        root = SpacyLikeToken(0, "VERB", "ROOT", 0)
        child = SpacyLikeToken(1, "NOUN", "nsubj", 0)

        # Create a minimal doc that supports indexing
        class SimpleDoc:
            def __init__(self, tokens):
                self._tokens = tokens
                for t in tokens:
                    t.doc = self
                self.sents = [object()]

            def __iter__(self):
                return iter(self._tokens)

            def __getitem__(self, idx):
                return self._tokens[idx]

        doc = SimpleDoc([root, child])
        profile = syntax_profile(["verb noun"], nlp=FakeNLP([doc]))

        # Verify it completes and produces correct depth
        self.assertIsNotNone(profile)
        self.assertEqual(profile["parse_depth"]["max"], 1)


if __name__ == "__main__":
    unittest.main()
