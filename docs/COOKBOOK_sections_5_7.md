---

## 5. Task Examples

Each task ships with a config under `framework/configs/<task>/` and a task class under
`framework/tasks/<task>/task.py`. All four generation cells (`forward/inverse ×
seeded/seedless`) are in principle available to every task; the shipped config runs
`forward + seeded` as the baseline.

---

### 5.1 GEC — Grammatical Error Correction

**Strategy:** `corruption` — the generator rewrites a real (or synthetic) correct
sentence by injecting one or more grammatical errors.

**Benchmark:** FCE corpus in M2 format (`framework/data/benchmarks/gec/fce.m2`).
Task models are seq2seq correctors; the framework feeds them the *corrupted* sentence
and checks whether their output matches the known *correct* version.

**Error types (selection):** verb tense, subject–verb agreement, article/determiner,
preposition, spelling — drawn from ERRANT operation+category codes and mapped to
human-readable phrases by `get_error_descriptions()`.

#### The four cells

| cell | what happens |
|---|---|
| `forward + seeded` *(shipped baseline)* | A real correct sentence is rewritten; the generator picks which error to inject. No distribution profile needed. |
| `forward + seedless` | No real sentence — a content spec (topic, length, style) from the benchmark profile seeds an original/corrupted pair. Needs `--topics` profile. |
| `inverse + seeded` | A real correct sentence has errors injected according to the benchmark's ERRANT distribution. Maintains realistic error-type mix. |
| `inverse + seedless` | A synthetic carrier sentence (from profile) has errors injected by the same ERRANT distribution. Needs `--topics` profile. |

#### Commands

Set `generation.mode` and `generation.seedless` in the config, then run:

```bash
python -m framework.main --config framework/configs/gec/config.yaml
```

Or override on the fly without touching the config:

```bash
# inverse + seeded
python -m framework.main --config framework/configs/gec/config.yaml --mode inverse

# forward + seedless (build profile first — see section 3)
python -m framework.main --config framework/configs/gec/config.yaml --seedless

# inverse + seedless
python -m framework.main --config framework/configs/gec/config.yaml --mode inverse --seedless
```

#### Task models

| model | type | notes |
|---|---|---|
| `vennify/t5-base-grammar-correction` | `t5` | T5 fine-tune |
| `prithivida/grammar_error_correcter_v1` | `gec_v1` | seq2seq |
| `grammarly/coedit-large` | `coedit` | instruction fine-tune |

#### Evaluators

| metric | measures |
|---|---|
| `gleu` | Fluency of the correction (sentence-level BLEU) |
| `errant` | Precision / Recall / F0.5 of edit operations |
| `errant_dist` | Distribution of edit categories (R:VERB:TENSE, M:DET, …) |
| `cola` | Linguistic acceptability of the model's output |
| `correction_extent` | Fraction of the input that was changed |
| `n_edits` | Raw edit count |

#### Config notes

```yaml
generation:
  max_tokens: 65536   # reasoning models think before emitting; 1024 truncated ~2/3 of calls
  temperature: 1.0    # CoT 3-step prompt benefits from higher temperature
  sample_size: 150    # per run; real pool loaded to match
```

GEC `forward + seeded` is the only cell where the generator freely picks the error
type — no empirical distribution is needed. Calibration for this cell therefore
reweights the *seed pool* (which seeds are fed) rather than any prompt parameter.

---

### 5.2 Spam Detection

**Strategy:** `class_conditional` — a target class (SPAM or HAM) is drawn first from
the class balance, then a message of that class is generated. Both classes are
first-class generated outputs.

**Benchmark:** SMS spam CSV (`framework/data/benchmarks/spam/sms_spam_ham_300.csv`).
`parse_row()` keeps **HAM rows only** as seeds (SPAM rows return `None`). SPAM seeds
for inverse mode are loaded separately from the full dataset via
`_load_reference_rows()`.

**Signal types:** explicit offers, urgency, prize claims, URL patterns, financial
keywords, imperatives — detected by regex in `spam_profiler.detect_signals()`.

#### The four cells

| cell | what happens |
|---|---|
| `forward + seeded` *(shipped baseline)* | Each class imitates itself: a real labeled seed is rewritten into a new message of the same class. |
| `forward + seedless` | Per-class content specs from the profile; no real seeds. Needs `--topics` profile. |
| `inverse + seeded` | Target label drawn independently and imposed on a seed of any class — a HAM message becomes SPAM by injecting a signal mix; a SPAM message is paraphrased clean. |
| `inverse + seedless` | Same as inverse+seeded but carriers synthesized from the profile. Needs `--topics` profile. |

#### Commands

Set `generation.mode` and `generation.seedless` in the config, then run:

```bash
python -m framework.main --config framework/configs/spam/config.yaml
```

Or override on the fly without touching the config:

```bash
# inverse + seeded
python -m framework.main --config framework/configs/spam/config.yaml --mode inverse

# seedless variants (need profile with --topics first)
python -m framework.main --config framework/configs/spam/config.yaml --seedless
python -m framework.main --config framework/configs/spam/config.yaml --mode inverse --seedless
```

To change the class balance, edit the config directly — there is no CLI flag for it:

```yaml
generation:
  class_balance: {SPAM: 0.4, HAM: 0.6}   # explicit | empirical | float
```

#### Task models

| model | type |
|---|---|
| `mshenoda/roberta-spam` | `roberta` |
| `mariagrandury/roberta-base-finetuned-sms-spam-detection` | `roberta` |
| `mrm8488/bert-tiny-finetuned-sms-spam-detection` | `bert_tiny` |
| `wesleyacheng/sms-spam-classification-with-bert` | `bert` |

#### Evaluators

| metric | notes |
|---|---|
| `accuracy` | Overall correct classification rate |
| `precision` | SPAM is the positive label |
| `recall` | SPAM is the positive label |
| `f1` | SPAM is the positive label |
| `fpr` | False-positive rate (legitimate messages flagged as spam) |

`fpr` is intentionally hidden from the charts (reads near 0.00 in practice) but
written to `results.json`.

#### Fidelity profile

Spam is a classification task, so every run writes `profile.json` with:

- **class balance** — real vs generated SPAM fraction
- **signal rates** — per signal type, fraction of SPAM messages that fire it
- **Jensen-Shannon divergences** — `type_dist_jsd`, `count_dist_jsd`

The fidelity chart (`plots/fidelity.png`) visualises real vs generated signal rates
and the class balance side by side.

#### Config notes

```yaml
generation:
  class_balance: empirical   # empirical | float | {SPAM: 0.3, HAM: 0.7}
  # empirical = match real dataset's P(SPAM); explicit value beats calibration
```

`class_balance: empirical` re-reads P(SPAM) from the loaded reference each run.
Set `reference_size` under `dataset.local` to cap how many rows are loaded as the
reference (default: entire split — can be slow on CPU).

---

### 5.3 Sentiment Analysis

**Strategy:** `corruption` — a real tweet is rewritten by injecting a sentiment
transformation (negation, sarcasm, intensity reduction, …), which implies a
deterministic ground-truth label the classifier is scored against.

**Benchmark:** TweetEval sentiment subset (`framework/data/benchmarks/sentiment/tweeteval_base_150.csv`).  
Build the CSV once per clone:

```bash
python -m scripts.prepare_sentiment_benchmark
```

#### Error types → labels

| error type | ground-truth label |
|---|---|
| `negation_insertion`, `sarcasm_injection`, `*negative*` | `NEGATIVE` |
| `*positive*` | `POSITIVE` |
| `intensity_reduction` | `NEUTRAL` |
| `paraphrase` | *(skipped — original sentiment unknown after rephrasing)* |

`get_label()` encodes this mapping. Samples whose `get_label()` returns `None` are
excluded from evaluation.

#### The four cells

| cell | what happens |
|---|---|
| `forward + seeded` *(shipped baseline)* | A real tweet is rewritten with a sentiment transformation; model picks the transformation. |
| `forward + seedless` | Synthetic tweet generated from a profile spec; error type drawn from the profile. Needs `--topics` profile. |
| `inverse + seeded` | Transformation drawn from the **real label balance** and imposed on a real tweet. |
| `inverse + seedless` | Synthetic carrier rewritten to a target transformation. Needs `--topics` profile. |

#### Commands

Set `generation.mode` and `generation.seedless` in the config, then run:

```bash
python -m framework.main --config framework/configs/sentiment/config.yaml
```

Or override on the fly without touching the config:

```bash
# inverse + seeded (match real label balance)
python -m framework.main --config framework/configs/sentiment/config.yaml --mode inverse

# seedless variants (build profile first)
python -m framework.main --config framework/configs/sentiment/config.yaml --seedless
python -m framework.main --config framework/configs/sentiment/config.yaml --mode inverse --seedless
```

#### Task models

| model | type |
|---|---|
| `finiteautomata/bertweet-base-sentiment-analysis` | `bertweet` |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | `multilingual` |

Both fine-tuned on TweetEval sentiment — same benchmark the seeds are drawn from.

#### Evaluators

| metric | notes |
|---|---|
| `accuracy` | Overall correct classification (3 classes) |
| `macro_precision` | Per-class precision, averaged over NEG/NEU/POS |
| `macro_recall` | Per-class recall, averaged |
| `macro_f1` | Mean of per-class F1 (sklearn `average="macro"`) |

#### Config notes

```yaml
generation:
  seedless: true           # the shipped config runs seedless forward
  model: "google/gemini-2.5-flash"   # via openrouter
  judge:
    enabled: true          # judge enabled in the shipped config
```

`forward + seeded` is the only Sentiment cell that cannot be calibrated — the
model picks its own transformation, and real tweets carry no transformation label
to aim a seed mix at.

---

### 5.4 Taxonomy Induction

**Strategy:** `structured` — one generated sample is a *complete synthetic taxonomy*
(domain, classes, subclass axioms), not a sentence. This means `sample_size: 10`
costs ten full LLM artifacts, each potentially several thousand tokens.

**Benchmark:** Pizza ontology in JSONL format (`framework/data/benchmarks/taxonomy/pizza.jsonl`).  
This file is **not shipped** — build it once from the OWL source:

```bash
python -m scripts.prepare_taxonomy_benchmark \
  --input /path/to/pizza.owl \
  --output framework/data/benchmarks/taxonomy/pizza.jsonl \
  --ontology-id pizza \
  --domain pizza
```

#### The four cells

| cell | what happens |
|---|---|
| `forward + seeded` *(shipped baseline)* | A real ontology subtree is anonymised (`C0, C1, …`) and the model re-verbalises it into a new domain. Gold structure verified by positional match. |
| `forward + seedless` | Only the domain is supplied; every structural property emerges freely. |
| `inverse + seeded` | A real subtree is edited toward a structural target (from the profile), then re-verbalised. Gold is the edited structure. |
| `inverse + seedless` | A structural target is sampled from the profile and imposed via a bounded feedback loop. |

#### Two-step setup for seedless cells

Seedless cells need a structural profile. Build it once:

```bash
# No --topics for taxonomy (not supported)
python -m framework.profile_dataset \
  --task taxonomy \
  --config framework/configs/taxonomy/config.yaml
# writes: framework/data/profiles/taxonomy/<benchmark>_<n>_taxonomy_profile.json
```

#### Commands

Set `generation.mode` and `generation.seedless` in the config, then run:

```bash
python -m framework.main --config framework/configs/taxonomy/config.yaml
```

Or override on the fly without touching the config:

```bash
# inverse + seeded
python -m framework.main --config framework/configs/taxonomy/config.yaml --mode inverse

# forward + seedless (build profile first)
python -m framework.main --config framework/configs/taxonomy/config.yaml --seedless

# inverse + seedless (feedback loop active; build profile first)
python -m framework.main --config framework/configs/taxonomy/config.yaml --mode inverse --seedless
```

#### Task models

| model | type | notes |
|---|---|---|
| `lexical` | `lexical` | Longest-suffix heuristic — cheap strong baseline, cannot memorise ontologies |
| `star` | `star` | Every class under one root — the structural floor |
| `minimax-m3` | `llm` | Same model as generation — report separately |

#### Evaluators

| metric | notes |
|---|---|
| `precision` | Exact `(child, parent)` pairs, micro-averaged |
| `recall` | Exact `(child, parent)` pairs, micro-averaged |
| `f1` | Harmonic mean of the above |
| `diagnostics` | Macro P/R/F1; malformed/invalid/unknown-class relations per run |

#### Fidelity

Taxonomy fidelity compares structural properties with Jensen-Shannon divergence:
depth distribution, parent-count distribution, child-count distribution; plus scalar
comparisons for `n_classes`, `n_roots`, `n_leaves`, `max_depth`, `mean_depth`.

Two fidelity plots per session:
- `plots/taxonomy_fidelity.png` — scalar diffs + JSD values
- `plots/taxonomy_fidelity_distributions.png` — normalised real vs synthetic
  depth / parent-count / child-count shapes

#### Config notes

```yaml
generation:
  sample_size: 10         # whole taxonomies per run — one order of magnitude below sentence tasks
  max_tokens: 65536       # a reasoning model thinks before emitting the full artifact;
                          # 32768 truncated on Pizza's 99 classes
  timeout: 1200           # a full 65536-token answer can approach 8 minutes
  feedback:
    max_rounds: 1         # leave `enabled` out — explicit enabled: true makes 3 cells refuse
  seed_pool:
    max_depth: 4          # truncate subtrees here to create structural variety
    min_classes: 5        # drop subtrees smaller than this
    domains: [...]        # target re-verbalisation domains; fixed = reproducible
```

**Calibration** is available only for `inverse + seedless`:

```bash
python -m framework.calibrate \
  --config framework/configs/taxonomy/config.yaml \
  --mode inverse --seedless
```

The other three cells refuse calibration (see README for the rationale).

---

## 6. Outputs & Troubleshooting

### 6.1 Where outputs are saved

Every run creates a session directory under `output.base_dir/<task>/`:

```
framework/data/runs/<task>/<session>/
  results.json              # provenance + scores; rewritten after every run
  generated/
    run_1.json              # synthetic data for run 1 (never reused for eval)
    run_2.json
    run_N.json
    run_N_rejected.json     # samples the judge dropped, with verdict
  real_sample.json          # real reference rows used for baseline + profiling
  profile.json              # {real, generated, fidelity} — classification tasks only
  plots/
    generated_vs_real_<model>.png
    run_variance_<model>.png
    fidelity.png            # spam
    sentiment_fidelity.png  # sentiment
    taxonomy_fidelity.png               # taxonomy
    taxonomy_fidelity_distributions.png # taxonomy
```

Session names encode the setup: `20260901_120000_forward_seeded` or
`20260901_120000_inverse_seedless`. A judged session appends nothing — that
information lives in `results.json`'s `meta.judge_stats`.

Override the root with `--output`:

```bash
python -m framework.main --config ... --output /tmp/my_runs
```

### 6.2 Reading results.json

```json
{
  "meta": {
    "timestamp": "2026-09-01T12:00:00",
    "task": "spam",
    "mode": "forward",
    "seedless": false,
    "provider": "openrouter",
    "model": "minimax-m3",
    "sample_size": 150,
    "real_baseline": true,
    "partial": false,
    "judge_stats": null
  },
  "results": {
    "mshenoda/roberta-spam": {
      "generated": {
        "f1": {"mean": 0.82, "std": 0.03},
        "accuracy": {"mean": 0.87, "std": 0.02}
      },
      "real": {
        "f1": 0.90,
        "accuracy": 0.93
      }
    }
  }
}
```

Key fields:

| field | meaning |
|---|---|
| `meta.partial: true` | Run is still in progress; some runs are not yet complete |
| `generated.<metric>.std` | Core GET signal — high std means instability on unseen data |
| `real.<metric>` | Single-pass score on the real benchmark (no variance) |
| `meta.judge_stats` | `{seen, dropped}` per run when judge was enabled; `null` otherwise |

For taxonomy seeded sessions, two extra fields appear:

| field | meaning |
|---|---|
| `results.<model>.real_paired` | `mean ± std` of each run scored against the real subtrees *it delivered* |
| `results.<model>.real_paired_runs` | One paired score per run |
| `meta.paired_real: true` | All runs carry paired scores |

### 6.3 Re-rendering plots

Plots are written automatically after every run. To re-render from a past session
without re-running:

```bash
python -m framework.plotting framework/data/runs/spam/20260901_120000_forward_seeded/

# write to a different directory
python -m framework.plotting framework/data/runs/gec/<session>/ --out /tmp/figs
```

### 6.4 Troubleshooting

---

#### Missing API key

```
[ERROR] No API key found for generator provider 'openrouter'.
Set OPENROUTER_API_KEY in .env or override api_keys.openrouter.
```

**Fix:** Copy `example.env` → `.env` in the repo root and fill in the key for the
provider you are using. The framework fails fast before any API call when a key is
missing — you will not burn credits discovering a missing key mid-run.

---

#### Empty API responses

```
[WARN] Run 2, sample 14: empty response from API — retrying (1/2)
```

Some providers (especially free-tier or heavily loaded endpoints) return empty bodies
intermittently. The generator retries up to `generation.max_retries` times with a
30-second sleep between attempts. If a sample still yields nothing, it is skipped and
logged. A run that produces zero usable samples aborts entirely rather than writing
all-zero scores.

**Fixes:**
- Increase `generation.max_retries` (2–5) and `generation.timeout` (600–1200 s)
- Switch to a more reliable provider for the same model (e.g. `openrouter` instead of a direct endpoint)
- Reduce `generation.request_delay` if you are on a paid plan with high rate limits, or increase it if you are hitting 429s

---

#### Token-limit truncation (TruncatedResponse)

```
[WARN] Run 1, sample 3: response truncated at max_tokens=1024 — skipping
```

The generator emitted output that hit the `max_tokens` cap before finishing. A
truncated response is **always skipped** — it is never partially parsed, because a
partially generated corrupt sentence (or taxonomy) is not a valid sample.

**Fixes:**
- Increase `generation.max_tokens`. The shipped configs use `65536` — never go below
  `16384` for reasoning models (they spend budget on `<think>` before the answer).
- Taxonomy is the most affected: `inverse + seedless` on Pizza's 99 classes
  truncated at `32768` on every attempt; `65536` was required.
- GEC with the 3-step CoT prompt measured completions of 1881–7376 tokens; `1024`
  truncated ~2/3 of all calls in early experiments.

---

#### Missing benchmark file

```
FileNotFoundError: framework/data/benchmarks/gec/fce.m2
```

Benchmark files are gitignored. You need to provide them yourself:

| task | file | how to get it |
|---|---|---|
| GEC | `framework/data/benchmarks/gec/fce.m2` | Download FCE corpus |
| Spam | `framework/data/benchmarks/spam/sms_spam_ham_300.csv` | Download SMS spam dataset |
| Sentiment | `framework/data/benchmarks/sentiment/tweeteval_base_150.csv` | `python -m scripts.prepare_sentiment_benchmark` |
| Taxonomy | `framework/data/benchmarks/taxonomy/pizza.jsonl` | `python -m scripts.prepare_taxonomy_benchmark --input pizza.owl ...` |

---

#### Missing benchmark profile (seedless runs)

```
RuntimeError: No profile found for task 'gec' in framework/data/profiles/gec/.
Build one with: python -m framework.profile_dataset --task gec --config ...
```

Profiles are gitignored. Build one once per clone before the first seedless run:

```bash
python -m framework.profile_dataset --task gec \
  --config framework/configs/gec/config.yaml \
  --topics --topic-sample-size 20
```

`--topics` is **required** for seedless generation. Without it the profile lacks
the `topics` block the content-spec sampler reads, and the framework fails fast
naming the missing block.

---

#### `python framework/main.py` not found

```
ModuleNotFoundError: No module named 'framework'
```

Run as a module from `live-eval/` (the parent of `framework/`):

```bash
# correct
cd live-eval
python -m framework.main --config ...

# wrong — do NOT do this
python framework/main.py --config ...
```

---

## 7. Adding a New Task

A new task requires at minimum:

1. A task class (`framework/tasks/<task>/task.py`)
2. A task config (`framework/configs/<task>/<task>.json`)
3. A run config (`framework/configs/<task>/config.yaml`)
4. Task models under `framework/models/<task>/`
5. Evaluator functions under `framework/evaluators/<task>/`
6. Registration in `framework/pipeline.py`

---

### Step 1 — Create the task class

Subclass `BaseTask` and implement the **seven required abstract methods**. These are
the minimum for `forward + seeded`:

```python
# framework/tasks/mytask/task.py
from framework.tasks.base_task import BaseTask

class MyTask(BaseTask):

    def get_task_name(self) -> str:
        return "mytask"

    def get_error_types(self) -> list[str]:
        # Return every label/transformation/signal the generator may produce
        return ["type_a", "type_b", "type_c"]

    def get_prompt_instruction(self) -> str:
        # Forward-mode prompt template; {text} and {error_type} are filled in
        return "Rewrite the following text to exhibit {error_type}:\n\n{text}"

    def get_evaluators(self) -> list[str]:
        # Names of evaluator functions returned by get_evaluator_fns()
        return ["accuracy", "f1"]

    def get_evaluator_fns(self) -> dict:
        from framework.evaluators.classification.accuracy import compute_accuracy
        from framework.evaluators.classification.f1 import compute_f1
        import functools
        return {
            "accuracy": compute_accuracy,
            "f1": functools.partial(compute_f1, positive_label="TYPE_A"),
        }

    def get_model(self, model_config: dict):
        from framework.models.mytask.my_model import MyModel
        return MyModel(model_config)

    def parse_row(self, row: dict) -> dict | None:
        text = row.get("text")
        if not text:
            return None
        return {"incorrect": text}
```

`parse_row()` returning `None` silently skips that row — use it to drop classes
you don't want as seeds (like SpamTask dropping SPAM rows).

---

### Step 2 — Add optional hooks

Each additional capability is a separate opt-in. An unsupported cell fails fast
naming the exact accessor it needed — no silent fallback.

| capability | methods to add |
|---|---|
| `get_label()` | Maps `result["error_type"]` → ground-truth label for evaluation; return `None` to skip a sample |
| **Inverse cells** | `get_inverse_prompt()`, `get_error_descriptions()`, `profile_error_distribution()` |
| **Seedless cells** | `get_carrier_prompt()` (inverse) or `get_seedless_forward_prompt()` (forward) + a `--topics` profile |
| **Real-vs-generated fidelity** | `build_fidelity_profile()`, `compare_fidelity_profiles()`, `get_real_eval_samples()` |
| **Calibration** | `get_calibration_keys()` — maps `{"type_dist": "<profile_key>", "count_dist": "<profile_key>"}` |
| **Class-conditional strategy** | `get_generation_strategy()` returning `"class_conditional"`, plus `get_class_labels()`, `get_inverse_class_prompts()`, `get_forward_prompts()`, `get_seedless_class_prompts()`, `get_seed_pool()` |
| **Profile-side selection** | `get_profile_side(mode)` — `"incorrect"` or `"correct"` per mode; default is `"incorrect"` |

---

### Step 3 — Add the JSON config

Create `framework/configs/mytask/mytask.json`:

```json
{
  "error_types": ["type_a", "type_b", "type_c"],
  "prompt": "Rewrite the following text to exhibit {error_type}:\n\n{text}",
  "judge_prompt": null,
  "inverse_prompt": null,
  "evaluators": ["accuracy", "f1"],
  "models": {
    "mymodel": {
      "label_map": {"TYPE_A": 0, "TYPE_B": 1}
    }
  }
}
```

Add optional keys (`judge_prompt`, `inverse_prompt`, `carrier_prompt`,
`seedless_forward_prompt`, `error_descriptions`) only when you implement the
corresponding method.

---

### Step 4 — Add the run config

Create `framework/configs/mytask/config.yaml` (copy from an existing task and edit):

```yaml
api_keys:
  openrouter: "${OPENROUTER_API_KEY}"
  # ...

dataset:
  source: local
  local:
    path: "framework/data/benchmarks/mytask/my_data.csv"
    format: csv

generation:
  mode: "forward"
  seedless: false
  provider: "openrouter"
  model: "minimax-m3"
  temperature: 1.0
  max_tokens: 65536
  num_runs: 3
  sample_size: 50

task:
  name: "mytask"

task_models:
  - name: "my-org/my-classifier"
    type: "mymodel"

compute:
  device: cpu

output:
  base_dir: "framework/data/runs"
  plots: true
```

---

### Step 5 — Register the task

Open `framework/pipeline.py` and add your task to `load_task()`:

```python
def load_task(name: str) -> BaseTask:
    if name == "gec":
        from framework.tasks.gec.task import GECTask
        return GECTask()
    if name == "spam":
        from framework.tasks.spam.task import SpamTask
        return SpamTask()
    # ...
    if name == "mytask":                          # ← add this
        from framework.tasks.mytask.task import MyTask
        return MyTask()
    raise ValueError(f"Unknown task: '{name}'")
```

---

### Step 6 — Add model and evaluator classes

**Model** (`framework/models/mytask/my_model.py`):

```python
from framework.models.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, config: dict):
        # load checkpoint, tokenizer, etc.
        pass

    def predict(self, samples: list[dict]) -> list[str]:
        # receives list of {"text": "..."} dicts
        # returns list of predicted label strings
        return ["TYPE_A"] * len(samples)
```

**Evaluator** functions live in `framework/evaluators/mytask/` and follow the same
signature as the classification evaluators: `(predictions, labels) -> float`.

---

### Step 7 — Run a smoke test

Start small to verify the full pipeline end-to-end before committing to a full run:

```bash
python -m framework.main \
  --config framework/configs/mytask/config.yaml \
  --runs 1 \
  --sample-size 5 \
  --no-real-baseline \
  --no-plots
```

This produces one run of five samples, no real-benchmark baseline, no plots — the
fastest possible check that generation, parsing, evaluation and results writing
all complete without error. Inspect `framework/data/runs/mytask/<session>/generated/run_1.json`
to verify the generated samples look correct before running at full scale.
