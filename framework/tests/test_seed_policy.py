import random
import unittest

from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        return self.response


COMMON = dict(
    class_balance={"SPAM": 1.0, "HAM": 0.0}, labels=("SPAM", "HAM"),
    inverse_prompts={"SPAM": "inject {error_spec} into {sentence}",
                      "HAM": "rewrite {sentence}"},
    type_dist={"phishing_link": 1.0}, count_dist={1: 1.0},
    error_descriptions={"phishing_link": "insert a link"},
)


class SeedPolicyTests(unittest.TestCase):
    def test_impose_seeds_regardless_of_label(self):
        # impose's seed is purely index-based over real_seeds; it does not
        # depend on which label the draw happens to land on. The only real
        # seed here reads as ordinary chat, yet the forced SPAM draw is still
        # rendered through it.
        gen = FakeGenerator("Message: Win a FREE prize http://x.com now")
        out = gen.generate_class_conditional(
            real_seeds=[{"incorrect": "see you at lunch"}], seed_field="incorrect",
            sample_size=1, seed_policy="impose", rng=random.Random(0), **COMMON,
        )
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertIn("see you at lunch", gen.prompts[0])

    def test_impose_missing_seed_consumes_no_rng_draw(self):
        # Regression for a restructuring bug: an earlier version of this loop
        # drew the label (and, when it wanted signals, sampled categories)
        # BEFORE resolving/checking the seed, so a row with a falsy seed field
        # silently burned an rng draw even though that iteration was skipped.
        # That shifts every later draw and breaks impose's required
        # byte-for-byte equivalence with the pre-restructure implementation,
        # which resolved the seed — and skipped on a missing one — before
        # touching rng at all.
        #
        # Expressed as an invariant rather than a hardcoded threshold (the
        # label draw is now `rng.choices` over a balance vector, not
        # `rng.random() < class_prob`, so the old raw threshold arithmetic no
        # longer applies): a run that SKIPS a leading row with a missing seed
        # field must produce byte-identical output, under the same rng seed,
        # to a run that never included that row at all. If the skip consumed
        # a draw, the two runs would diverge.
        common = {**COMMON, "class_balance": {"SPAM": 0.5, "HAM": 0.5}}

        gen_with_gap = FakeGenerator("Rewritten: glad we could catch up again soon")
        seeds_with_gap = [{"incorrect": ""}, {"incorrect": "valid seed sentence here"}]
        out_with_gap = gen_with_gap.generate_class_conditional(
            real_seeds=seeds_with_gap, seed_field="incorrect", sample_size=2,
            seed_policy="impose", rng=random.Random(10), **common,
        )

        gen_without_gap = FakeGenerator("Rewritten: glad we could catch up again soon")
        seeds_without_gap = [{"incorrect": "valid seed sentence here"}]
        out_without_gap = gen_without_gap.generate_class_conditional(
            real_seeds=seeds_without_gap, seed_field="incorrect", sample_size=1,
            seed_policy="impose", rng=random.Random(10), **common,
        )

        self.assertEqual(len(out_with_gap), 1)
        self.assertEqual([r["label"] for r in out_with_gap],
                         [r["label"] for r in out_without_gap])

    def test_inherit_picks_a_seed_of_the_drawn_label(self):
        gen = FakeGenerator("Rewritten: CLAIM your FREE reward today")
        seeds = [{"text": "see you at lunch", "label": "HAM"},
                 {"text": "WIN cash now", "label": "SPAM"}]
        out = gen.generate_class_conditional(
            real_seeds=seeds, seed_field="text", sample_size=1,
            seed_policy="inherit", forward_prompts={"SPAM": "spam {sentence} :: {error_spec}", "HAM": "ham {sentence}"},
            rng=random.Random(0), **COMMON,
        )
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertIn("WIN cash now", gen.prompts[0])
        self.assertNotIn("see you at lunch", gen.prompts[0])

    def test_inherit_missing_label_in_pool_raises(self):
        gen = FakeGenerator("Rewritten: x")
        with self.assertRaises(RuntimeError) as ctx:
            gen.generate_class_conditional(
                real_seeds=[{"text": "hi", "label": "HAM"}], seed_field="text",
                sample_size=1, seed_policy="inherit",
                forward_prompts={"SPAM": "spam {sentence} :: {error_spec}", "HAM": "ham {sentence}"}, rng=random.Random(0), **COMMON,
            )
        self.assertIn("SPAM", str(ctx.exception))

    def test_none_policy_uses_specs_and_never_touches_seeds(self):
        gen = FakeGenerator("Message: FREE prize, click http://x.com")
        out = gen.generate_class_conditional(
            real_seeds=None, sample_size=1, seed_policy="none",
            seedless_prompts={"SPAM": "spam {spec} using {error_spec}",
                              "HAM": "ham {spec}"},
            specs_by_label={"SPAM": ["topic: prizes; roughly 12 words"],
                            "HAM": ["topic: chat; roughly 8 words"]},
            rng=random.Random(0), **COMMON,
        )
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertIn("topic: prizes", gen.prompts[0])
        self.assertIn("insert a link", gen.prompts[0])

    def test_none_policy_skips_judge_when_no_seed_to_compare(self):
        # IMPORTANT 3: seed_policy="none" has no seed ("source" is None), so
        # judge_prompt.format(sentence=text, correction=source) would render
        # "Counterpart: None" — spam's inverse_judge_prompt asks whether the
        # counterpart is a natural legitimate message, sees "None", and
        # rejects everything. Judging must be skipped entirely for this
        # policy (not called with correction=None) so the sample survives
        # and the judge is never invoked at all.
        gen = FakeGenerator("Message: FREE prize, click http://x.com")
        judge_calls = []

        def rejecting_judge(prompt):
            judge_calls.append(prompt)
            return "Redundancy: trivial\nCorrection: incorrect"

        out = gen.generate_class_conditional(
            real_seeds=None, sample_size=1, seed_policy="none",
            seedless_prompts={"SPAM": "spam {spec} using {error_spec}",
                              "HAM": "ham {spec}"},
            specs_by_label={"SPAM": ["topic: prizes; roughly 12 words"],
                            "HAM": ["topic: chat; roughly 8 words"]},
            judge_prompt="Is {sentence} vs counterpart {correction} legit?",
            judge_call=rejecting_judge,
            rng=random.Random(0), **COMMON,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertEqual(judge_calls, [])

    def test_none_policy_other_label_uses_its_own_prompt(self):
        gen = FakeGenerator("Message: are we still on for lunch tomorrow")
        common = {**COMMON, "class_balance": {"SPAM": 0.0, "HAM": 1.0}}
        out = gen.generate_class_conditional(
            real_seeds=None, sample_size=1, seed_policy="none",
            seedless_prompts={"SPAM": "spam {spec} {error_spec}", "HAM": "ham {spec}"},
            specs_by_label={"SPAM": ["x"], "HAM": ["topic: chat; roughly 8 words"]},
            rng=random.Random(0), **common,
        )
        self.assertEqual(out[0]["label"], "HAM")
        self.assertIn("topic: chat", gen.prompts[0])



class SeedlessPromptValidationTests(unittest.TestCase):
    """seed_policy="none" indexes seedless_prompts/specs_by_label by the drawn
    label inside the loop, so a task supplying only one label must fail before
    the first API call rather than KeyError-ing partway through a paid run."""

    def _kwargs(self, **over):
        kw = dict(COMMON, real_seeds=None, sample_size=4, seed_policy="none",
                  seedless_prompts={"SPAM": "s {spec} {error_spec}", "HAM": "h {spec}"},
                  specs_by_label={"SPAM": ["a"], "HAM": ["b"]},
                  rng=random.Random(0))
        kw.update(over)
        return kw

    def test_missing_prompt_for_a_class_fails_before_any_call(self):
        gen = FakeGenerator("Message: x y z")
        with self.assertRaises(RuntimeError) as ctx:
            gen.generate_class_conditional(
                **self._kwargs(seedless_prompts={"SPAM": "s {spec} {error_spec}"}))
        self.assertIn("seedless_prompts", str(ctx.exception))
        self.assertIn("HAM", str(ctx.exception))
        self.assertEqual(gen.prompts, [])          # nothing was generated

    def test_missing_specs_for_a_class_fails_before_any_call(self):
        gen = FakeGenerator("Message: x y z")
        with self.assertRaises(RuntimeError) as ctx:
            gen.generate_class_conditional(**self._kwargs(specs_by_label={"SPAM": ["a"]}))
        self.assertIn("specs_by_label", str(ctx.exception))
        self.assertIn("HAM", str(ctx.exception))
        self.assertEqual(gen.prompts, [])

    def test_complete_mappings_are_accepted(self):
        gen = FakeGenerator("Message: x y z")
        out = gen.generate_class_conditional(**self._kwargs(sample_size=1))
        self.assertEqual(len(out), 1)


class InheritTechniqueTests(unittest.TestCase):
    """inherit rewrites a seed of the drawn label. A signal-bearing label's
    template emphasises the sampled signal mix, so naming its technique after
    the sampled category is honest; a label whose template asks for no signals
    renders a plain same-label rewrite, still recorded as "imitation" — this is
    the forward+seeded cell, whose records must stay comparable with the
    archives written before the symmetric-inverse change."""

    def test_technique_reflects_what_actually_happened(self):
        seeds = [{"text": "see you at lunch soon", "label": "HAM"},
                 {"text": "WIN cash now today", "label": "SPAM"}]
        cases = (({"SPAM": 1.0, "HAM": 0.0}, "SPAM"), ({"SPAM": 0.0, "HAM": 1.0}, "HAM"))
        for class_balance, expected_label in cases:
            gen = FakeGenerator("Rewritten: brand new message here")
            out = gen.generate_class_conditional(
                **{**COMMON, "class_balance": class_balance},
                real_seeds=seeds, seed_field="text", sample_size=1,
                seed_policy="inherit", forward_prompts={"SPAM": "spam {sentence} :: {error_spec}", "HAM": "ham {sentence}"},
                rng=random.Random(0),
            )
            self.assertEqual(out[0]["label"], expected_label)
            # The signal-bearing label names its sampled category; the other
            # label injects nothing and stays "imitation".
            expected_technique = ("phishing_link" if expected_label == "SPAM"
                                  else "imitation")
            self.assertEqual(out[0]["technique"], expected_technique)



class ForwardSignalEmphasisTests(unittest.TestCase):
    """Forward mode used to inherit whatever signals its seed happened to carry,
    so it could not target the empirical distribution the way inverse does. A
    signal-bearing label now rewrites its seed emphasising a sampled signal mix."""

    SEEDS = [{"text": "see you at lunch soon", "label": "HAM"},
             {"text": "WIN cash now today", "label": "SPAM"}]
    PROMPTS = {"SPAM": "rewrite {sentence} emphasising {error_spec}",
               "HAM": "rewrite {sentence}"}

    def _run(self, class_balance):
        gen = FakeGenerator("Rewritten: a brand new message here")
        out = gen.generate_class_conditional(
            **{**COMMON, "class_balance": class_balance},
            real_seeds=self.SEEDS, seed_field="text", sample_size=1,
            seed_policy="inherit", forward_prompts=self.PROMPTS,
            rng=random.Random(0),
        )
        return gen.prompts[0], out[0]

    def test_signal_bearing_label_receives_the_sampled_signal_mix(self):
        prompt, record = self._run(class_balance={"SPAM": 1.0, "HAM": 0.0})
        self.assertIn("insert a link", prompt)          # rendered error_spec
        self.assertIn("WIN cash now today", prompt)     # its own-label seed
        self.assertEqual(record["technique"], "phishing_link")

    def test_other_label_gets_no_signal_mix(self):
        prompt, record = self._run(class_balance={"SPAM": 0.0, "HAM": 1.0})
        self.assertNotIn("insert a link", prompt)
        self.assertIn("see you at lunch soon", prompt)
        self.assertEqual(record["technique"], "imitation")

    def test_missing_prompt_for_a_label_fails_before_any_call(self):
        gen = FakeGenerator("Rewritten: x y z")
        with self.assertRaises(RuntimeError) as ctx:
            gen.generate_class_conditional(
                **COMMON, real_seeds=self.SEEDS, seed_field="text", sample_size=2,
                seed_policy="inherit", forward_prompts={"SPAM": "only {sentence} {error_spec}"},
                rng=random.Random(0),
            )
        self.assertIn("forward_prompts", str(ctx.exception))
        self.assertIn("HAM", str(ctx.exception))
        self.assertEqual(gen.prompts, [])


class NoSignalTechniqueNamingTests(unittest.TestCase):
    """`technique` for a label whose template asks for no signals is named
    per SEED POLICY, not one word for all three.

    Archives are the reason. "imitation" (inherit) and "paraphrase" (none) are
    what the forward and forward+seedless cells have always written, and those
    two cells are required to behave exactly as they did before the
    symmetric-inverse change — a single "rewrite" everywhere would have made
    post-change records non-comparable to the archive on that field for no
    benefit. `impose` is the one cell whose behaviour genuinely changed (its
    non-signal label now strips signals off a seed of any class rather than
    paraphrasing a seed of its own), so it says "rewrite" instead of restoring
    a "paraphrase" that would now be a lie."""

    HAM_ONLY = {**COMMON, "class_balance": {"SPAM": 0.0, "HAM": 1.0}}

    def test_impose_records_rewrite(self):
        gen = FakeGenerator("Message: are we still on for lunch tomorrow")
        out = gen.generate_class_conditional(
            real_seeds=[{"incorrect": "WIN a FREE prize now http://x.com"}],
            seed_field="incorrect", sample_size=1, seed_policy="impose",
            rng=random.Random(0), **self.HAM_ONLY,
        )
        self.assertEqual(out[0]["technique"], "rewrite")

    def test_inherit_records_imitation(self):
        gen = FakeGenerator("Rewritten: glad we could catch up again soon")
        out = gen.generate_class_conditional(
            real_seeds=[{"text": "see you at lunch soon", "label": "HAM"}],
            seed_field="text", sample_size=1, seed_policy="inherit",
            forward_prompts={"SPAM": "spam {sentence} {error_spec}",
                             "HAM": "ham {sentence}"},
            rng=random.Random(0), **self.HAM_ONLY,
        )
        self.assertEqual(out[0]["technique"], "imitation")

    def test_none_records_paraphrase(self):
        gen = FakeGenerator("Message: are we still on for lunch tomorrow")
        out = gen.generate_class_conditional(
            real_seeds=None, sample_size=1, seed_policy="none",
            seedless_prompts={"SPAM": "spam {spec} {error_spec}", "HAM": "ham {spec}"},
            specs_by_label={"SPAM": ["x"], "HAM": ["topic: chat; roughly 8 words"]},
            rng=random.Random(0), **self.HAM_ONLY,
        )
        self.assertEqual(out[0]["technique"], "paraphrase")

    def test_a_signal_bearing_label_still_names_its_categories(self):
        # The policy name is the fallback for "no signals drawn", never a
        # replacement for the sampled category list.
        gen = FakeGenerator("Message: Win a FREE prize http://x.com now")
        out = gen.generate_class_conditional(
            real_seeds=[{"incorrect": "see you at lunch"}], seed_field="incorrect",
            sample_size=1, seed_policy="impose", rng=random.Random(0), **COMMON,
        )
        self.assertEqual(out[0]["technique"], "phishing_link")


if __name__ == "__main__":
    unittest.main()
