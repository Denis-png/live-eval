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

## Judge

The structural checks (parsing, and verification against gold in seeded cells) prove
an artifact's shape, not its meaning: a taxonomy can match gold and still call a
part or a property a subclass. The optional LLM judge (`judge:` in the config, off by
default and turned on per run with `--judge`) reads each accepted artifact's domain,
classes and axioms through `judge_prompt` in `taxonomy.json`. It drops a taxonomy whose
axioms are not a sensible is-a hierarchy of the domain, or whose classes are
degenerate (placeholder names, near-duplicates, classes outside the domain).

It judges the final artifact of every cell: after any feedback rounds, and after
gold verification in seeded cells. A dropped taxonomy is not regenerated, the same as
a dropped sentence in the other tasks, so a judged run can deliver fewer than
`sample_size` taxonomies. Each drop is archived with the judge's verdict in
`run_<N>_rejected.json`, and `meta.judge_stats` counts what the judge saw and dropped
per run.

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
  max_tokens: 65536         # a reasoning model thinks before it emits the artifact
  timeout: 1200
  feedback:
    max_rounds: 1           # leave `enabled` to taxonomy.json's default: an explicit
                            # `enabled: true` makes three of the four cells refuse
  seed_pool:
    max_depth: 4
    min_classes: 5

task_models:
  - {name: lexical, type: lexical}
  - {name: star, type: star}
  - {name: minimax-m3, type: llm, provider: openrouter, max_tokens: 65536}
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

Live smoke runs on 2026-09-14 raised both limits to 65536:

- inverse+seedless imposes all 99 of Pizza's classes. At 32768 it truncated on 9 of
  9 attempts. At 65536 it delivered 3 of 3 taxonomies, though one needed three
  attempts (one returned duplicate class names, one still truncated) and took 23
  minutes. The cell's 3 samples took 38 minutes in all.
- forward+seedless delivered 3 of 3 (52, 101 and 82 classes) in about 8 minutes.
- The LLM task model's answer on the whole 99-class ontology -- the seedless cells'
  real baseline -- truncated at 32768 in one session (real F1 0.0) and completed in
  another (0.722). At 65536, three independent calls all completed in 41-57 seconds
  (F1 0.839-0.895).

History: an earlier Xiaomi/MiMo smoke run at `generation.max_tokens = 4096` never
produced an artifact. Both attempts ended with `finish_reason = length` and
`message.content = None` after spending the whole budget on reasoning; the same
truncation later hit minimax at 16384 on the largest seeds, then at 32768 on
inverse+seedless and on the whole-ontology evaluation.

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

## Calibration

`python -m framework.calibrate --config framework/configs/taxonomy/config.yaml
--mode inverse --seedless` calibrates the one calibratable cell and writes an
artifact under `framework/data/profiles/taxonomy/`, which later runs of that cell
pick up automatically.

| cell | steered | measured against the real reference |
|---|---|---|
| inverse+seedless | the imposed depth and child-count distributions | the same distributions over every generated class |
| forward+seeded, inverse+seeded | nothing | refuses |
| forward+seedless | nothing | refuses |

inverse+seedless replaces the imposed distributions in the profile the prompt
and the feedback loop both read. An artifact measured against a different real
reference is ignored with a warning.

forward+seedless refuses because it imposes no structure: only a domain is
supplied. The seeded cells refuse because their only control input would be seed
weights over max-depth buckets. Seeds are drawn from a small pool of subtrees
(10 on Pizza), so bucket weights drawn with replacement add more structural
noise than the verification attrition they would correct. The weights are also
draw probabilities, while the measured mix is weighted by class count, so the
first round over-draws the deep buckets. Both refusals happen before any
generation. A seeded run still honours a seed-weight artifact if one exists
(for example, one written before seeded calibration was refused). Verification
attrition therefore still shifts a seeded benchmark's structural mix. Per-run
paired scoring (see "What the Real Baseline Is") keeps that attrition out of the
paired generated-vs-real gap, and the whole-reference gap still shows it.

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
only the names and domain changed. Per session, `real` is the whole pool, scored
once, while each run's synthetic side is the subtrees that run drew and that
passed verification. Verification drops the largest subtrees most often, and they
dominate micro-averaged scores, so the gap against `real` mixes the change of
vocabulary with draw and verification attrition.

Seeded sessions are therefore also scored per run against the real items each
run delivered. Each item in `real_sample.json` carries its `pool_index`, and each
generated record the `source_pool_index` of the subtree it re-verbalises.
`real_paired_runs[k]` is run k's task models scored on the matching real subtrees
(a subtree drawn twice counts twice), reusing the real predictions, so pairing
costs no model call. `real_paired` aggregates those as `mean ± std`, like
`generated`, and `meta.paired_real: true` marks the session. The paired gap
isolates generation fidelity on the items a run delivered; the gap against
`real`, which is kept unchanged, still includes what verification dropped. A
session pairs all its runs or none. Seedless sessions are not paired: a seedless
artifact comes from no particular real item.

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
