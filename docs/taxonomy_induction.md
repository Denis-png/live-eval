# Taxonomy Induction

Taxonomy Induction / Subclass Axiom Induction evaluates whether a model can
infer direct subclass relationships from:

- a domain
- a list of class identifiers

The gold direct subclass axioms stay internal to the framework for scoring.

## Benchmark Preparation

The normalized benchmark flow is:

```text
OWL/RDF ontology
-> scripts/prepare_taxonomy_benchmark.py
-> normalized JSONL
```

The current Pizza MVP uses the canonical Protege Pizza ontology as the intended
real benchmark. The converter keeps:

- named classes
- direct named `rdfs:subClassOf` relations plus parents named by `owl:equivalentClass`
  intersection definitions (listed in `metadata.definitional_axioms`)
- multiple inheritance

It intentionally excludes:

- anonymous blank-node restrictions
- unions, enumerations, and complements from equivalence definitions
- inferred or transitive hierarchy
- reasoner-classified structure

Example preparation command:

```bash
python scripts/prepare_taxonomy_benchmark.py \
  --input /path/to/pizza.owl \
  --output framework/data/benchmarks/taxonomy/pizza.jsonl \
  --ontology-id pizza \
  --domain pizza
```

The framework consumes the normalized JSONL representation at runtime.

## Profiling

Run taxonomy profiling with:

```bash
python -m framework.profile_dataset \
  --task taxonomy \
  --config framework/configs/taxonomy/config.yaml
```

The structural profile includes:

- `n_classes`
- `n_subclass_axioms`
- roots and leaves
- hierarchy depth
- parent-count distribution
- child-count / branching distribution
- multiple-parent fraction
- cycle and validation fields

## Generation

Taxonomy uses:

```text
strategy = structured
```

One generated sample is one complete synthetic taxonomy, not one class or one edge.

Taxonomy generation has two modes:

- **Seedless** (the default when `seedless` is omitted; the shipped config runs
  seeded): profile-driven generation producing completely synthetic
  taxonomies. The model receives a structural target sampled from the real profile
  (inverse mode) or only a domain (forward mode).
- **Seeded**: the model receives a real benchmark subtree anonymised as `C0, C1, ...`
  and must re-verbalise it into a new domain. Gold structure is computed exactly
  (inherited for forward, edited toward a target for inverse) and verified by
  positional match.

In both modes, generation does not use corruption or error-type semantics. The
seedless cells do not use real benchmark class names or edges as seeds (generation
is driven by structural profiles). Structural targets in seedless inverse are
approximate targets, not exact constraints; seeded inverse uses exact computed targets.

## Structured Output

The generator is expected to return strict JSON:

```json
{
  "domain": "example",
  "classes": ["A", "B", "C"],
  "subclass_axioms": [
    ["A", "B"]
  ]
}
```

Subclass axioms are ordered pairs:

```text
child -> parent
```

Validation rules:

- duplicate classes are invalid
- unknown relation endpoints are invalid
- self-loops are invalid
- cycles are invalid
- multiple inheritance is allowed
- duplicate edges are normalized

## Feedback Loop

The feedback loop **is** the inverse-mode mechanism. `mode: inverse` imposes a
structural target sampled from the real profile, and the loop is how the artifact
is driven toward it. `mode: forward` imposes no target, so it has no loop. Setting
`generation.feedback.enabled` in the RUN config alongside it raises before any
API call, since that is a request the run made and silently ignoring it would be
wrong. `taxonomy.json`'s own `feedback` block is a task-level default tuned for
inverse — it is inherited, not set by the run, so forward simply skips the loop.

The bounded feedback loop is:

```text
initial synthetic taxonomy
-> structural profile
-> real-vs-synthetic comparison
-> deterministic structural feedback
-> optional bounded regeneration
```

The MVP default allows at most one feedback-informed regeneration. Feedback can
be disabled in config.

Feedback is structural only. It does not include real class names, real subclass
edges, ontology URIs, URI maps, or real hierarchy examples. Feedback is a guide;
it is not guaranteed to improve fidelity on every run.

## Evaluation

Evaluator-model input contains only:

```json
{
  "domain": "...",
  "classes": ["..."]
}
```

Gold `subclass_axioms` remain internal for scoring.

Evaluation uses exact ordered `(child, parent)` matching:

- precision
- recall
- F1

The MVP does not use fuzzy matching, synonym matching, semantic equivalence, or
transitive reasoning. Unknown-class valid prediction pairs count as false
positives. Malformed predictions are tracked as diagnostics and cannot become
valid predicted relations.

## Structural Fidelity

Structural fidelity compares the real profile with generated taxonomy profiles.
Scalar comparisons include:

- `n_classes`
- `n_subclass_axioms`
- `n_roots`
- `n_leaves`
- `max_depth`
- `mean_depth`
- `multiple_parent_fraction`

Distribution comparisons use Jensen-Shannon divergence for:

- depth distribution
- parent-count distribution
- child-count distribution

Lower JSD means more similar distributions. The framework does not create one
combined overall fidelity score.

A seeded session's real side is several subtrees of one ontology (see "What the
Real Baseline Is"). They are pooled into ONE reference profile, summarised with
the statistic the synthetic side is summarised with: each scalar is the mean over
the subtrees, and each distribution is the mean of the subtrees' normalised
distributions -- not their summed counts, which would let the largest subtree
dominate. `fidelity.real_profile.pooled_taxonomies` records how many subtrees the
reference pools. Real taxonomies from different ontologies are not pooled; the
fidelity step refuses them. Because each synthetic item is still compared with
the pool's mean, the per-item JSD is not zero even for a synthetic side identical
to the pool: it includes the subtrees' own spread around their mean.

## Plots

Taxonomy runs write:

```text
plots/taxonomy_fidelity.png
plots/taxonomy_fidelity_distributions.png
```

`taxonomy_fidelity.png` summarizes scalar structural differences and
distribution JSD values. `taxonomy_fidelity_distributions.png` shows normalized
real-vs-synthetic distribution shapes for depth, parent count, and child count.

## Example Run Configuration

Relevant fields of the shipped `framework/configs/taxonomy/config.yaml`:

```yaml
generation:
  mode: forward             # the shipped baseline cell; --mode / --seedless pick others
  seedless: false
  provider: openrouter
  model: minimax-m3
  num_runs: 3
  sample_size: 10           # whole taxonomies per run
  max_tokens: 32768         # a reasoning model thinks before it emits the artifact
  timeout: 600
  feedback:
    max_rounds: 1           # leave `enabled` to taxonomy.json's default: an explicit
                            # `enabled: true` makes three of the four cells refuse
  seed_pool:
    max_depth: 4
    min_classes: 5

task_models:
  - {name: lexical, type: lexical}
  - {name: star, type: star}
  - {name: minimax-m3, type: llm, provider: openrouter, max_tokens: 32768}
```

`sample_size: 10` and `num_runs: 3` mean ten synthetic taxonomies per run, across
three independent runs.

The taxonomy task config also contains feedback tolerances:

```json
{
  "feedback": {
    "enabled": true,
    "max_rounds": 1,
    "tolerances": {
      "count_relative": 0.15,
      "depth_absolute": 0.5,
      "rate_absolute": 0.05,
      "distribution_jsd": 0.1
    }
  }
}
```

Run the pipeline with the shipped config:

```bash
python -m framework.main --config framework/configs/taxonomy/config.yaml
```

## Validation Note

A live `forward+seeded` smoke run on 2026-09-11 (minimax-m3, generation and LLM
task model both at `max_tokens: 32768`, 2 samples, 1 run) verified both generated
taxonomies against their computed gold, and scored the LLM task model at real
F1 0.886 on the matched seed-pool reference (lexical 0.30, star 0.17). Every stage
-- generation, evaluation, fidelity and plots -- completed.

History: an earlier Xiaomi/MiMo smoke run at `generation.max_tokens = 4096` never
produced an artifact. Both attempts ended with `finish_reason = length` and
`message.content = None` after spending the whole budget on reasoning; the same
truncation later hit minimax at 16384 on the largest seeds, which is why both
limits now sit at 32768.

## Seeded Generation

`generation.seedless: false` samples a subtree of the real ontology and computes
its gold structure directly:

- **forward+seeded** inherits the subtree's own shape. No profile is consulted.
- **inverse+seeded** imposes a structural target sampled from the real profile,
  applying edit operators (`drop_leaf`, `reparent`, `collapse_level`,
  `add_sibling`) to move the subtree toward it.

The model then receives that structure with every class anonymised as `C0, C1,
...` and a target domain, and must return one new class identifier per
anonymised class with every relation preserved. The response is checked against
the gold by an exact positional match — the i-th returned name is taken to be
the i-th anonymised class, and the relabelled edge set must equal gold's
exactly. A mismatch is a counted skip.

Positional rather than isomorphic on purpose: ontologies are DAGs, and a
rooted-tree canonicalisation can accept two structurally different graphs, which
is the one failure this gate exists to prevent. It is also the stronger check —
it verifies the model preserved THE structure it was handed, not that it
produced some graph shaped like it.

Ground truth is therefore never parsed from model output. That is deliberate: a
drifting model loses its sample rather than redefining the reference. It also
keeps the source ontology's class names out of the generated benchmark, so a
model under test cannot score by recalling a canonical tutorial ontology.

Seeded cells run no feedback loop — the gold is exact, so there is nothing to
iterate toward.

## What the Real Baseline Is

Each session compares its synthetic benchmark against a real reference drawn the
same way:

| cell | real reference |
|---|---|
| seeded (`seedless: false`) | the real subtrees of the same seed pool, unedited |
| seedless | the whole ontology |

A seeded benchmark is made of subtrees, and precision and recall on a small graph
are far higher than on a large one. Comparing it against the whole ontology would
make it look easier purely because of size. The same reference also supplies the
real side of the structural fidelity profile, so it keeps that comparison
like-for-like too.

Per item, a `forward+seeded` synthetic item is one pool subtree's structure with
only the names and domain changed. Per session the two sides are not paired: the
real side is the whole pool, scored once, while each run's synthetic side is the
subtrees that run drew and that passed verification. Verification drops the
largest subtrees most often, and they dominate micro-averaged scores. A seeded
session's gap therefore mixes the change of vocabulary with draw and verification
attrition; it does not measure name reliance alone. The pairing can be recovered
from the archive -- each item in `real_sample.json` carries its `pool_index`, and
each generated record carries the `source_pool_index` of the subtree it
re-verbalises -- but per-run paired scoring is not implemented.

The pool's subtrees overlap: they are one ontology reweighted, not independent
taxonomies. On the current Pizza benchmark the 10 subtrees cover all 98 distinct
gold edges as 327 edge instances -- 92 of the 98 edges sit in more than one
subtree, 51 of them in four. So "n = 10" overstates independence. It does not
bias the comparison, since the synthetic side is drawn from the same subtrees
and shares the nesting, but it does shrink the effective n: ten overlapping
subtrees carry less independent evidence than ten separate taxonomies would.

Micro-averaged scores are the headline. `diagnostics` also carries
`macro_precision`, `macro_recall` and `macro_f1` -- each taxonomy's own score,
averaged -- so a report can check whether its conclusions depend on weighting
large taxonomies more heavily than small ones.

## Current MVP Limitations

- direct named subclass relations only: asserted `rdfs:subClassOf` edges plus the
  named conjuncts of `owl:equivalentClass` intersection definitions (listed in
  each record's `metadata.definitional_axioms`); no union, enumeration or
  complement is read as a parent
- anonymous OWL restrictions excluded
- no ontology reasoner or classification
- no transitive evaluation
- no semantic or fuzzy matching
- fidelity takes its real reference from one ontology: a seeded session's
  subtrees of it are pooled, and real taxonomies from different ontologies need
  an explicitly named reference
- model/provider structured-output behavior can affect real runs
