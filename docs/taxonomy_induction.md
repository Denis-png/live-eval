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
- direct named `rdfs:subClassOf` relations
- multiple inheritance

It intentionally excludes:

- anonymous blank-node restrictions
- inferred or transitive hierarchy
- reasoner-classified structure

Example preparation command:

```bash
python scripts/prepare_taxonomy_benchmark.py \
  --input /path/to/pizza.owl \
  --output framework/data/taxonomy/pizza.jsonl \
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

Generation is profile-driven and seedless by design. One generated sample is
one complete synthetic taxonomy, not one class or one edge.

Taxonomy generation does not use:

- inverse mode
- corruption or error-type semantics
- real benchmark class names as seeds
- real benchmark subclass edges as seeds

Structural targets are approximate targets, not exact constraints.

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

Relevant taxonomy config fields:

```yaml
generation:
  provider: openrouter
  model: xiaomi/mimo-v2.5
  num_runs: 3
  sample_size: 1
  profile_path: framework/data/profiles/taxonomy_profile.json
  max_parse_attempts: 2
  max_tokens: 4096
  feedback:
    enabled: true
    max_rounds: 1
```

`sample_size: 1` and `num_runs: 3` means one complete synthetic taxonomy per
run, across three independent runs.

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

Run the pipeline with a complete taxonomy YAML config:

```bash
python -m framework.main --config /path/to/taxonomy_run.yaml
```

## Validation Note

The final Xiaomi/MiMo OpenRouter smoke validation reached the provider and
passed prompt leakage checks, but did not produce a valid generated taxonomy
under:

```text
generation.max_tokens = 4096
```

Both bounded attempts returned `message.content = None`, had
`finish_reason = length`, consumed all 4096 completion tokens, and reported
reasoning output. No textual taxonomy JSON was produced, so evaluator and
fidelity stages were not reached in that smoke run.

This is treated as a model/configuration limitation rather than a taxonomy
validator failure. The validation note does not imply that increasing token
limits is proven to solve the issue.

## Current MVP Limitations

- asserted direct named subclass relations only
- anonymous OWL restrictions excluded
- no ontology reasoner or classification
- no transitive evaluation
- no semantic or fuzzy matching
- no seeded taxonomy generation
- no inverse taxonomy generation
- current Pizza MVP assumes one real reference ontology for fidelity selection
- model/provider structured-output behavior can affect real runs
