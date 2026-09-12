# GET Evaluation Framework

A flexible, task based framework for evaluating NLP models
using the GET (Generate → Evaluate → Trash) methodology.

---

## What is GET?

Instead of evaluating models on fixed public benchmarks
(which the model may have memorized during training),
this framework generates fresh synthetic data on every run.

This eliminates benchmark contamination and reveals
how stable a model really is on truly unseen data.

"Trash" means the synthetic data is never reused for evaluation.
Each run's data is archived under `framework/data/runs/<task>/<session>/generated/`
so it can be inspected later.

---

## Project Structure

    framework/
        main.py                  - entry point
        pipeline.py              - GET loop (Generate, Evaluate, Trash)
        data_loading.py          - dataset source resolution + local file loaders (m2/csv/tsv/jsonl)
        requirements.txt         - Python dependencies
        configs/                 - one directory per task: config.yaml (dataset,
                                   generator, task models, output) beside
                                   <task>.json (error types, prompts, evaluators,
                                   model params)
            gec/       config.yaml + gec.json
            spam/      config.yaml + spam.json
            sentiment/ config.yaml + sentiment.json
            taxonomy/  config.yaml + taxonomy.json
        tasks/
            base_task.py         - abstract task template (declares generation strategy)
            gec/task.py          - Grammatical Error Correction task (corruption: forward + inverse)
            spam/task.py         - Spam Detection task (class-conditional)
            sentiment/task.py    - Sentiment Analysis task (corruption)
            taxonomy/task.py     - Taxonomy Induction task (structured)
        generators/              - LLM that creates synthetic evaluation data
            base_generator.py    - shared generate() / generate_inverse() / generate_class_conditional() loops
            openai_generator.py  - OpenAI / Groq / OpenRouter / Mistral (OpenAI-compatible)
            anthropic_generator.py  - Anthropic / MiniMax (Anthropic-compatible)
            google_generator.py
        profiling/               - two distinct kinds of profile, do not confuse them:
                                   BENCHMARK profile (topics/length/style, built by
                                   `profile_dataset`, INPUT to seedless generation) and
                                   FIDELITY profile (task.build_fidelity_profile, a
                                   MEASUREMENT of real vs generated)
            errant_distribution.py  - ERRANT-based GEC error distribution
            spam_distribution.py  - spam-signal-based spam error distribution
            fidelity.py           - Jensen-Shannon divergence for distribution fidelity
            text_stats.py         - samplable length/style/vocabulary characteristics
            topics.py             - opt-in LLM topic profiling (profile_dataset --topics)
            syntax_stats.py       - spaCy-based GEC syntactic complexity
        plotting/            - figures from a run session (matplotlib, headless)
            plots.py         - pure figure builders (dict -> Figure)
            session.py       - load a session, render + save PNGs (fail-soft)
        models/gec/              - GEC models under evaluation
            seq2seq.py (t5/gec_v1/coedit), claude.py
        models/spam/             - Spam models under evaluation
            roberta.py, bert_tiny.py
        evaluators/              - scoring functions applied to model predictions
            gleu.py              - GLEU score
            gec/                 - errant, errant_dist, cola, correction_extent, n_edits
            classification/      - accuracy, precision, recall, f1, fpr (spam)
        data/                    - all gitignored (only .gitkeep is committed)
            benchmarks/<task>/   - source benchmark files (fce.m2, *.csv, *.jsonl)
            profiles/            - profile JSON from `python -m framework.profile_dataset`
            runs/<task>/<session>/ - per-session run artifacts; the session name
                                   carries the setup that produced it, e.g.
                                   20260827_120000_inverse_seedless
    docs/
        taxonomy_induction.md    - Taxonomy Induction task guide

---

## Setup

1. Install Python deps (run from `live-eval/`):

       pip install -r framework/requirements.txt
       python -m spacy download en_core_web_sm    # required by ERRANT

2. Copy `live-eval/example.env` → `live-eval/.env` and fill in the API keys
   you need. `main.py` loads it automatically. You only need the keys for
   providers you actually use (the generator's provider, plus Anthropic if
   you evaluate Claude as a task model).

3. Edit `framework/configs/<task>/config.yaml` (e.g. `framework/configs/gec/config.yaml`
   or `framework/configs/spam/config.yaml` — each task's config carries only the
   fields that task reads; there is no shared root config):
   - `dataset`         — `source` (huggingface | local). Per-source settings live in
                         nested blocks that can both stay filled in — switching source
                         is a one-field change:

         dataset:
           source: local            # huggingface | local
           huggingface: {name: "deysi/spam-detection-dataset", split: "train"}
           local: {path: "framework/data/benchmarks/spam/sms_spam_ham_300.csv", format: csv}

     Local formats: `m2` (GEC benchmarks like FCE/CoNLL-14, annotator 0's edits),
     `csv`, `tsv` (header row, fields matched by the task's `parse_row`, e.g.
     `label`/`text` for spam). `format` is optional when the file extension says it.
   - `generation`      — generator provider, model, temperature, `num_runs`, and
                         `sample_size` (**the single sample-size knob** — synthetic
                         samples generated per run; the real pool is loaded to match).
                         Two independent knobs select the generation cell for
                         **every** task, corruption or class-conditional — see
                         "Generation Strategies" below:
                         `mode` (`forward` | `inverse`) and `seedless`
                         (`true` | `false`, default `false`).
   - `class_balance`   — `empirical` (default: the real dataset's per-label
                         distribution), an explicit mapping, e.g. `{SPAM: 0.3, HAM: 0.7}`,
                         or a bare float, which is the two-label spelling of that mapping
                         (`0.3` means `{first label: 0.3, second: 0.7}`) and raises for a
                         task with three or more labels, where it cannot say what it means.
                         An explicit value is a user instruction and beats a calibrated
                         balance; calibration refines `empirical` only. Mapping values are
                         normalized, so relative weights are what matter. Naming a label
                         the task does not declare aborts before any API call.
   - `evaluation.real_baseline` — also score the task models on the real benchmark
                         (default `true`; see "Real baseline & fidelity").
   - `task.name`       — `gec` or `spam`
   - `task_models`     — list of models to evaluate
   - `output.base_dir` — root for per-session run artifacts (default
                         `framework/data/runs`); see "Results".

   > Sampling is deterministic: `load_real_data` takes the first `generation.sample_size`
   > matching rows (no shuffle/seed). As long as the dataset settings (source, HF
   > name/split or local path) and `sample_size` are unchanged, every invocation sees the
   > **same benchmark sample** — which is what makes cross-model comparison fair.

---

## How to Run

Run as a module from the `live-eval/` directory (the parent of `framework/`):

    cd live-eval
    python -m framework.main --config framework/configs/gec/config.yaml

`--config` is required — there is no default, since each task has its own
config file. CLI flags override values in the YAML:

    python -m framework.main \
        --config framework/configs/gec/config.yaml \
        --mode inverse \
        --seedless \
        --provider anthropic \
        --model claude-haiku-4-5 \
        --runs 3 \
        --sample-size 20 \
        --output framework/data/runs \
        --no-judge \
        --no-real-baseline

`--mode forward|inverse` and `--seedless`/`--no-seedless` select the generation cell
(see "Generation Strategies"); `--judge/--no-judge` toggles the LLM-as-judge filter;
`--real-baseline/--no-real-baseline` toggles the real-benchmark baseline;
`--output` sets `output.base_dir`. `--seedless` needs a profile built up front — see
"Seedless prerequisite: profiling" below.

> Note: `python framework/main.py` will NOT work — `framework` must be
> importable as a package, so use `python -m framework.main`.

The config is validated up front: missing required keys, `num_runs < 1`,
an unknown `generation.mode`, or a missing API key for the selected provider
all abort before any API call with an error naming the offending config path.

---

## Generation Strategies

Each task declares a **generation strategy** (`task.get_generation_strategy()`) — the
task *shape*, `corruption` (GEC) or `class_conditional` (Spam) — which the pipeline
dispatches on. Independently of that shape, **every task** reads two generation
knobs that together select one of four **cells**:

- **`generation.mode`** (`forward` | `inverse`) — *where the annotation comes from*.
  `inverse` draws it independently and **imposes** it on the source; `forward`
  **inherits** it from the source — the seed it was derived from, or the artifact
  just generated. It also explains a detail the framework treats as incidental:
  among the seeded text-level cells (corruption and class_conditional), GEC
  `forward` + seeded is the only one that loads no empirical distribution, because
  it is the only one that imposes nothing — the generator infers the error type
  from its seed. Every other seeded text-level cell imposes at least part of the
  annotation and needs a distribution to draw it from: spam `forward` inherits the
  label from its seed but still imposes the signal mix, and GEC `forward` +
  seedless draws the error type from the profile while the correction remains the
  model's own assertion. Taxonomy `forward` imposes nothing either, but structured
  generation never loads an empirical distribution in the first place — in either
  mode — so it sits outside this comparison entirely.

  The axis is task-shape-independent. Back-translation (generate the target, derive
  the source) and doc2query (pick a document, generate a query for it) are inverse
  under this definition; annotating text you just generated is forward.
- **`generation.seedless`** (`true` | `false`) — *whether real benchmark text ever
  reaches the generation prompt*. `false` ("seeded") passes a real sample from the
  dataset as a seed. `true` drops real seeds entirely: a
  benchmark **profile** (built once per clone by `profile_dataset`, see
  "Seedless prerequisite: profiling" below) is sampled instead to synthesize the
  content spec (topic, length, style) that goes into the prompt — the LLM invents
  the text from that spec rather than transforming a real sentence.

  Omitting the key means `false` (seeded) for every strategy except `structured`,
  which defaults to `true` because its seeded cells need a real-artifact corpus.
  `pipeline.resolve_seedless` is the single answer the session name, the
  generation dispatch, the results `meta` and `calibrate` all read.

`mode` and `seedless` combine freely; a task shape only needs to support the cells it
declares prompts for (see "Fail-fast" below).

### The four cells, per task shape

**`corruption` (GEC, Sentiment)** — corrupt a source text:

| | `seedless: false` (seeded) | `seedless: true` |
|---|---|---|
| **`mode: forward`** | `generate()` rewrites a real seed sentence into a corrupted variant; the generator picks the error type itself. | `generate_seedless_pairs()`: no real seed — a profile-sampled content spec drives the LLM to invent an original/corrupted pair directly. Needs `seedless_forward_prompt`. |
| **`mode: inverse`** | `generate_inverse()` corrupts the real benchmark's known-clean `correct` field according to an **empirical error distribution** (ERRANT-profiled) so the injected error mix matches the benchmark. | A carrier (clean sentence) is synthesized from a profile content spec via `generate_carriers()` (needs `carrier_prompt`), then fed through the same `generate_inverse()` in place of a real seed — the empirical error distribution still drives the injected errors. |

For Sentiment the "error types" are sentiment transformations (`sentiment_flip_negative`,
`sarcasm_injection`, …), each implying the label the sample is scored against. Real tweets
carry a label but no transformation, so its empirical distribution is the real **label
balance**, spread evenly over the transformations producing each label — one per sample.

**`class_conditional`** — classification is inherently label → text: draw a target
label from the balance (`class_balance`, default the real dataset's empirical
distribution), then synthesize an example of it. **Any number of labels**, not
just two. A label receives an injected signal mix iff its own prompt template
contains `{error_spec}`, so a task with no injectable features simply writes
prompts without it. Because every class is LLM-authored in every cell, a
classifier cannot separate them on "was this written by an LLM" artifacts.

| | `seedless: false` (seeded) | `seedless: true` |
|---|---|---|
| **`mode: inverse`** (default when `mode` is omitted) | The target label is drawn independently and **imposed** on a seed of any class, via `get_inverse_class_prompts()[label]`. A spam seed rewritten to HAM is a hard negative — spam-like topic, no spam signals. | The same, over carriers synthesized from the profile instead of real messages. |
| **`mode: forward`** | The seed's own label is **inherited**: each class rewrites a labeled seed of that class into a new one. Needs `forward_prompts`. | Per-label profile content specs, no real seed. Needs `seedless_class_prompts`. Note this cell draws its label from the balance, so it imposes rather than inherits — see Limitations. |

**`structured` (Taxonomy)** — one sample is a whole artifact. The `mode` and
`seedless` axes both vary, yielding four cells. Seeded cells verify against a
computed gold graph rather than parsing one from model output:

| | `seedless: false` (seeded) | `seedless: true` |
|---|---|---|
| **`mode: forward`** | Inherits the seed subtree's structure. No profile needed. | Only the domain is supplied. Size, depth and branching all emerge. |
| **`mode: inverse`** (default when `mode` is omitted) | Edits the seed toward a target sampled from the profile. | A structural target is sampled from the real profile and imposed; the feedback loop iterates toward it for a bounded number of rounds. |

### Setting it

    generation:
      mode: "inverse"        # forward | inverse — see the tables above
      seedless: false         # true = drop real seeds, generate from the profile only

CLI overrides: `--mode forward|inverse`, `--seedless`/`--no-seedless`. Every shipped
config runs the **forward + seeded** baseline with the same generator
(`openrouter` / `minimax-m3`); `framework/tests/test_shipped_configs.py` keeps the four
aligned.

### Fail-fast

A task that hasn't defined the prompt a requested cell needs (e.g. a corruption task
with no `seedless_forward_prompt`, or a classification task with no `carrier_prompt`)
raises `RuntimeError` before any API call, naming the task, the `(mode, seedless)` cell,
and the missing accessor — never a silent fallback to a different cell. Classification
forward mode also fails fast when the labeled seed pool is missing one of the classes.

### Seedless prerequisite: profiling

Every `seedless: true` cell reads a benchmark **profile** JSON instead of real text.
Profiles are **gitignored** (`framework/data/**/*.json`) — build one after every fresh
clone, before the first seedless run:

    python -m framework.profile_dataset --task gec \
      --config framework/configs/gec/config.yaml --topics --topic-sample-size 20 \
      # writes framework/data/profiles/gec/<benchmark>_<n>_gec_profile.json

`--topics` is required for seedless generation — it adds the LLM-driven `topics`
(GEC, Sentiment) / `topics_per_label` (Spam) block the content-spec sampler needs;
without it, loading the profile raises `RuntimeError` naming the missing block.
`--task` is one of `gec`, `spam`, `sentiment`, `taxonomy` (taxonomy takes no `--topics`),
and every task profiles the benchmark its config names — the rows the real baseline is
scored on. The profile lands in `framework/data/profiles/<task>/<benchmark>_<n>_<task>_profile.json`,
where the pipeline looks for it; override per-run with `generation.profile_path`.

| Task | strategy | cells |
|------|----------|------------|
| GEC  | `corruption` | forward / inverse × seeded / seedless (see table above) |
| Sentiment | `corruption` | forward / inverse × seeded / seedless; the inverse and seedless mix follows the real label balance |
| Spam | `class_conditional` | inverse / forward × seeded / seedless; class balance from `class_balance` |
| Taxonomy | `structured` | forward / inverse × seeded / seedless; seeded cells verify generated structure against computed gold; see [docs/taxonomy_induction.md](docs/taxonomy_induction.md) |

### Calibration (optional, improves fidelity)

The distributions generation samples from are a *request*. What the generator delivers is
measurably different, because it honours some categories more readily than others.
`framework.calibrate` measures that gap and corrects the request, writing a reusable
artifact beside the profile:

    python -m framework.calibrate --config framework/configs/spam/config.yaml \
        --rounds 3 --alpha 0.5 --tolerance 0.1 --sample-size 120
    # writes framework/data/profiles/spam/<benchmark>_<n>_<cell>_calibration.json

The task comes from `task.name` in the config; `--mode` and `--seedless/--no-seedless`
select the cell, `--output` overrides the artifact path. Runs then pick the artifact up
automatically for the same benchmark and cell, printing `Calibration: <path> (round N)`.
Every calibratable cell with no artifact prints a `[NOTE]` naming the command that would
build one. `structured` (taxonomy) reuses the two slots for class-level structure: `type_dist` is
the per-class depth distribution and `count_dist` the per-class child count. Only
inverse+seedless is calibrated: it steers the depth and branching it imposes. The other
three cells refuse. forward+seedless imposes nothing. The seeded cells draw their seeds
from a small pool of subtrees, so bucket weights drawn with replacement would add more
structural noise than the verification attrition they are meant to correct. Structured
calibration defaults to `3 * generation.sample_size` artifacts per round. Pin an artifact
with `generation.calibration_path`, or set that key to `null` to opt out.

A calibrated `class_prob` only ever corrects the **`empirical`** balance for differential
attrition: an explicit float or mapping in `generation.class_balance` is a user instruction
and always wins, calibration included.

Calibration is a **separate phase** on purpose. Steering runs *inside* a scored session
would make them non-i.i.d. and turn `results.json`'s `mean ± std` — the core GET
instability signal — into "generator noise plus controller settling". So calibration runs
once, emits a tuned spec, and the GET session runs unchanged at that fixed setting.
`meta.calibration` in `results.json` records which artifact produced a run: artifacts are
gitignored, so that is the only surviving provenance.

All settings are optional and have working defaults:

    calibration:
      rounds: 3          # extra rounds after round 0 (round 0 = the target, before correction)
      alpha: 0.5         # damping; lower is more conservative
      tolerance: 0.1     # per-dimension JSD at which the loop stops
      sample_size: 120   # default max(generation.sample_size, 100)

`sample_size` deliberately does **not** inherit `generation.sample_size`, which is tuned
for eval cost rather than estimation precision; a round measured on fewer than 50
informative samples warns that the update may be chasing noise.

The loop emits the **best** round, and round 0 always requests exactly the target
distribution, before any correction — so calibration can never make a benchmark worse than
round 0. For `type_dist`/`count_dist` cells that guarantee coincides with "never worse than
not calibrating", because round 0 there reproduces today's uncalibrated draw exactly. GEC
`forward + seeded` is the one exception: its target is the seed pool's own ERRANT profile,
and round 0 already draws seeds two-stage-weighted **with replacement** rather than the
uncalibrated first-N order — "never worse than round 0" still holds, but round 0 itself is
not the same draw as an uncalibrated run for that cell. Calibration is numeric only: no
wording is ever added to a generation prompt, which keeps it from teaching the generator
what the fidelity detectors look for.

**Per-cell control input.** `inverse` (seeded and seedless) and `forward + seedless`
steer `type_dist`/`count_dist`. GEC `forward + seeded` has no injectable distribution —
the generator picks its own error type — so calibration steers **which seeds are fed**
instead, reweighting the seed pool that is that cell's implicit error distribution; the
prompt is untouched and runs still draw seeds freshly each run. Sentiment steers
`type_dist`/`count_dist` in its imposed cells, measured on the transformations the
surviving samples carry, so it corrects per-transformation attrition from parsing and
judging; its `forward + seeded` cell cannot be calibrated (the model picks the
transformation, and real tweets carry none to aim a seed mix at) and says so. Spam steers
`type_dist`/`count_dist` in all four cells, plus `class_prob`, which is corrected in
closed form from per-class attrition (a model refuses to write spam far more often than
it refuses a benign paraphrase, so the surviving class balance drifts from the balance
that was asked for).

#### Limitations

Two bounds worth knowing before setting a tolerance, plus one unrelated to tolerance at all:

- **A concentrated signal mix may be unreachable.** `_sample_categories` draws without
  replacement, so one category's achievable share saturates near
  `1 / mean_signals_per_message`; measured directly, requesting a 0.830 share yielded
  0.353. Past that ceiling the request keeps rising while the measurement does not
  follow. The loop degrades safely — the best round wins, never worse than round 0 —
  but it cannot close that gap.
- **JSD has a floor above zero.** The target is Laplace-smoothed (so every supported
  category stays sample-able) while the measurement is raw, so even a perfectly compliant
  generator scores slightly above 0. Set `tolerance` with that floor in mind rather than
  chasing 0.
- **`mode` partly degenerates for `class_conditional` + `seedless`.** With no seed
  there is nothing to inherit, so `forward + seedless` draws its label from the
  balance — imposition, by the definition above. What actually separates the two
  seedless cells is one-step generation (write a SPAM message from a spec) versus
  two-step (synthesize a carrier, then impose a label on it). Both cells work and
  are reachable; the axis simply carries less meaning there.

## Real baseline & fidelity

By default (`evaluation.real_baseline: true`) every run also evaluates the same task
models on the **real benchmark** — a reference point for the generated scores. It's a
single deterministic pass (no runs/variance), scored with the same evaluators, and saved
alongside the generated scores (`results.<model>.real`). Disable with `--no-real-baseline`.

For classification tasks the run also writes a **fidelity profile** (`profile.json`)
comparing the real and generated datasets: class balance, per-signal rates, and a
**Jensen-Shannon divergence** (0 = identical, 1 = disjoint) over the signal distributions
— so you can check whether the generated benchmark actually matches the real one. The
generated side is measured by re-running the signal detectors on the generated text, so
JSD reflects detector-visible distribution match, not ground-truth semantics.

---

## Plots

Figures are rendered with matplotlib (installed by `framework/requirements.txt`, run
headless — no display needed).

### During a run (default)

Plots are on by default: every run writes its figures into that run's own
`<session>/plots/` directory.

    python -m framework.main                 # renders plots at the end of the run
    python -m framework.main --no-plots      # skip them

Or set it in `config.yaml`:

    output:
      plots: true    # false to disable

Plotting runs **after** `results.json` is already on disk and is **fail-soft**: if
matplotlib is missing or a figure fails to build, it warns and skips — it can never
cost you a run that already succeeded.

### Standalone (any past session)

Point the module at a session directory to (re)render it — no API calls, no re-run:

    python -m framework.plotting framework/data/runs/spam/20260708_172422/

    # write the PNGs somewhere else instead of <session>/plots/
    python -m framework.plotting framework/data/runs/gec/<session>/ --out /tmp/figs

It reads only that session's `results.json` (+ `profile.json` if present), so it
reproduces exactly the figures the run itself would have made. A bad path fails loudly
with a clear message.

### What you get

| File | Reads | Shows |
|------|-------|-------|
| `generated_vs_real_<model>.png` | `results.json` | Per evaluator: generated (mean ± std) beside the **real-benchmark baseline**. The headline chart — *is the synthetic benchmark a good proxy for real data?* |
| `run_variance_<model>.png` | `results.json` → `runs` | Each individual run's score per evaluator — run-to-run instability, the core GET signal. |
| `fidelity.png` | `profile.json` | Real vs generated spam-signal rates + class balance, titled with the Jensen-Shannon divergences. Classification tasks only. |

Figures whose data is absent are skipped, not errored: a GEC session has no
`profile.json`, so it simply gets no fidelity chart; sessions produced before per-run
scores were persisted get no variance chart.

### Reading them

- **Blue is always generated, orange is always real** — in every figure.
- Metrics on different scales (e.g. GEC's unbounded `n_edits` count vs 0–1 scores) are
  drawn in **separate panels**, never on a second y-axis.
- `fpr` is omitted from the figures (it reads a flat 0.00 against a 0.00 baseline — dead
  space). It is still computed and written to `results.json`; this only hides it from the
  charts. See `HIDDEN_METRICS` in `framework/plotting/plots.py`.

---

## Comparing Generation Models (same benchmark sample)

Use the multi-model driver to run several generation models over the identical
sample in one command:

    cd live-eval
    python -m scripts.compare_models --config framework/configs/gec/compare.yaml

A task's `compare.yaml` holds only `base_config: config.yaml` and the
`generation_models` list: every other setting is the run config's own, so a comparison
differs from a normal run only in the generation model. For another cell, point
`base_config` at an edited copy of `config.yaml`.

> **`generation_models` is read ONLY by `scripts.compare_models`.**
> `python -m framework.main` always runs the **single** model in `generation.provider` /
> `generation.model` and ignores the list — so adding `generation_models` and then
> running `framework.main` does *not* compare anything. `framework.main` now prints a
> `[NOTE]` at startup when it sees the list, naming the entries it is ignoring and the
> one model it is actually about to run.

Each model gets its own session under `output.base_dir/<task>/<provider>_<model>/`
(the same per-session layout as a normal run), plus a combined
`output.base_dir/<task>/comparison/comparison.json` and a printed comparison table
(generated `mean ± std` and, per model, the `real` baseline). The same benchmark sample
is guaranteed by deterministic first-N sampling, so keep `dataset.*` and `sample_size`
constant across entries.

The driver accepts only sample-shaping flags (`--config/--task/--runs/--sample-size`);
`--provider/--model` are rejected because per-model provider/model come from the
`generation_models` list. API keys for **all** listed providers are checked before
the first model runs.

---

## Cross-Session Analysis

`scripts.analyze_results` reads every session under one or more results roots and
writes the report-level figures plus `analysis.md` / `analysis.json`: generated-vs-real
fidelity and model ranking (Kendall tau-b), how much score variance the generation model
explains, the paired forward-vs-inverse mode effect, and each task's fidelity JSDs.

    python -m scripts.analyze_results framework/data/runs --since 2026-09-11 \
        --out framework/data/runs/analysis

It keeps one session per (task, cell, generation model): the one with the most completed
runs, then the newest. A run that consumed a calibration artifact is its own cell
(`<cell>+calibrated`), so a calibration ablation keeps both sides. `--since` limits the
analysis to one sweep — without it, an older session with more runs outranks a fresh one.

---

## How to Add a New Task

1. Create `framework/tasks/<task>/task.py` subclassing `BaseTask` and
   implement the seven abstract methods: `get_error_types`,
   `get_prompt_instruction`, `get_evaluators`, `get_evaluator_fns`,
   `get_model`, `get_task_name`, `parse_row`. That is enough for
   `forward + seeded`; every other capability is an opt-in hook and an
   unsupported cell fails fast naming the accessor it wanted:

   | To get | Add |
   |--------|-----|
   | `inverse` cells | `get_inverse_prompt`, `get_error_descriptions`, `profile_error_distribution` |
   | `seedless` cells | `get_carrier_prompt` / `get_seedless_forward_prompt` + a `--topics` profile |
   | real-vs-generated fidelity | `build_fidelity_profile`, `compare_fidelity_profiles`, `get_real_eval_samples` |
   | calibration | `get_calibration_keys` |
   | the `class_conditional` shape | `get_generation_strategy`, `get_class_labels`, `get_inverse_class_prompts`, `get_forward_prompts`, `get_seedless_class_prompts`, `get_seed_pool` |

   `get_class_labels()` returns the task's ordered label set — any number of
   labels, not just two — so the dispatcher takes the class names from the
   task, and a classification task is never generated under another task's
   vocabulary.
2. Create `framework/configs/<task>/<task>.json` with error types, prompts,
   evaluators list, and per-model inference params.
3. Register the task in `framework/pipeline.py::load_task()`.
4. Add model classes under `framework/models/<task>/` and evaluator
   functions under `framework/evaluators/<task>/`.
5. Create `framework/configs/<task>/config.yaml` with `task.name: <task>` and only
   the fields that task's generation strategy actually reads. `mode` and `seedless`
   apply to every task shape (see "Generation Strategies"); a config may omit
   `mode` and rely on its per-strategy default (`forward` for corruption,
   `inverse` for class-conditional — see `spam/config.yaml`'s comment).

---

## Results

Each run session gets its own directory under `output.base_dir/<task>/<session>/`:

    results.json       - {"meta": <provenance>, "results": <scores>}
    generated/
        run_1.json …   - each run's synthetic data (never reused for eval)
    real_sample.json   - the real reference sample
    profile.json       - {real, generated, fidelity}
    plots/             - generated_vs_real_<model>.png, run_variance_<model>.png, and the
                         task's fidelity figure (fidelity.png, sentiment_fidelity.png, or
                         taxonomy_fidelity*.png)

`results.json` has two top-level keys:

- `meta` — provenance: timestamp, task, mode, generator provider/model, dataset,
  `sample_size`, the number of samples actually **scored** per run, `real_baseline`,
  `class_balance`, and the judge used (or `null`). `meta.partial` is `true` while runs
  are still outstanding.
- `results` — per model, a `generated` block (each evaluator reports `mean ± std`
  across runs — high `std` reveals instability on unseen data) and, when the real
  baseline is on, a `real` block (single-pass point estimates on the real benchmark):

      "mshenoda/roberta-spam": {
        "generated": { "f1": {"mean": 0.82, "std": 0.03}, ... },
        "real":      { "f1": 0.90, ... }
      }

  A seeded taxonomy session also scores each run against the real items it delivered
  (see [docs/taxonomy_induction.md](docs/taxonomy_induction.md)); `real` stays the whole
  reference:

  - `real_paired` — `mean ± std` across runs of each run's score on the real items it delivered.
  - `real_paired_runs` — one paired score per run; entry k belongs to `runs[k]`.
  - `meta.paired_real` — `true` when the session carries both blocks (all runs are paired, or none).

`results.json` is rewritten after **every** run, so a crash or Ctrl-C in run N keeps the
aggregated results of runs 1..N-1. A run that generates zero usable samples aborts
instead of writing misleading all-zero scores.

The LLM-as-judge filter is **opt-in**: no `judge:` block (or `judge.enabled: false`)
means judging is skipped.

---

## Current Tasks

GEC (Grammatical Error Correction) — implemented (corruption: forward + inverse)
Spam Detection — implemented (class-conditional generation + real baseline + fidelity)
Taxonomy Induction — implemented (structured generation + subclass evaluation + structural fidelity); see [docs/taxonomy_induction.md](docs/taxonomy_induction.md)
Hate Speech Detection — planned
Sentiment Analysis — implemented (corruption: forward + inverse, seeded + seedless; label-balance fidelity)

## Current Evaluators (GEC)

GLEU — fluency of correction
ERRANT — precision / recall / F0.5 of edits
errant_dist — distribution of edit categories
CoLA — linguistic acceptability of the prediction
correction_extent — how much of the input was edited
n_edits — raw edit count

## Current Evaluators (Spam)

accuracy — overall correct classification rate
precision / recall / f1 — computed with SPAM as the positive label
fpr — false-positive rate (legitimate messages flagged as spam)

## Current Evaluators (Sentiment)

accuracy — overall correct classification rate
macro_precision / macro_recall — each class's own score, averaged over NEGATIVE / NEUTRAL / POSITIVE
macro_f1 — the mean of per-class F1 (sklearn's `average="macro"`)

## Current Evaluators (Taxonomy)

precision / recall / f1 — exact subclass relations, micro-averaged over taxonomies
diagnostics — macro precision / recall / F1, invalid and unknown-class relations, and
malformed vs failed predictions; see [docs/taxonomy_induction.md](docs/taxonomy_induction.md)
