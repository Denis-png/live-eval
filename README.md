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
        data_loading.py          - dataset source resolution + local file loaders (m2/csv/tsv)
        requirements.txt         - Python dependencies
        configs/
            config.yaml          - dataset, generator, task models, output
            tasks/
                gec.json         - GEC task config (error types, prompts, evaluators, model params)
            tasks/
                spam.json        - Spam task config (class-conditional generation)
        tasks/
            base_task.py         - abstract task template (declares generation strategy)
            gec/task.py          - Grammatical Error Correction task (corruption: forward + inverse)
            spam/task.py         - Spam Detection task (class-conditional)
        generators/              - LLM that creates synthetic evaluation data
            base_generator.py    - shared generate() / generate_inverse() / generate_class_conditional() loops
            openai_generator.py  - OpenAI / Groq / OpenRouter / Mistral (OpenAI-compatible)
            anthropic_generator.py  - Anthropic / MiniMax (Anthropic-compatible)
            google_generator.py
        profiling/               - empirical distribution profilers + real-vs-generated fidelity
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
        data/
            runs/                - per-session run artifacts, one dir per run (gitignored)

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
           local: {path: "framework/data/spam/sms_spam_ham_300.csv", format: csv}

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
                         (`true` | `false`, default `false`). Spam also reads
                         `class_balance` (`empirical` | float = P(SPAM)).
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

- **`generation.mode`** (`forward` | `inverse`) — *what real signal drives content*,
  regardless of whether that signal comes from a real seed or a synthesized one (see
  `seedless` below).
- **`generation.seedless`** (`true` | `false`, default `false`) — *whether real
  benchmark text ever reaches the generation prompt*. `false` ("seeded") passes a
  real sample from the dataset as a seed. `true` drops real seeds entirely: a
  benchmark **profile** (built once per clone by `profile_dataset`, see
  "Seedless prerequisite: profiling" below) is sampled instead to synthesize the
  content spec (topic, length, style) that goes into the prompt — the LLM invents
  the text from that spec rather than transforming a real sentence.

`mode` and `seedless` combine freely; a task shape only needs to support the cells it
declares prompts for (see "Fail-fast" below).

### The four cells, per task shape

**`corruption` (GEC)** — corrupt a source text:

| | `seedless: false` (seeded) | `seedless: true` |
|---|---|---|
| **`mode: forward`** | `generate()` rewrites a real seed sentence into a corrupted variant; the generator picks the error type itself. | `generate_seedless_pairs()`: no real seed — a profile-sampled content spec drives the LLM to invent an original/corrupted pair directly. Needs `seedless_forward_prompt`. |
| **`mode: inverse`** | `generate_inverse()` corrupts the real benchmark's known-clean `correct` field according to an **empirical error distribution** (ERRANT-profiled) so the injected error mix matches the benchmark. | A carrier (clean sentence) is synthesized from a profile content spec via `generate_carriers()` (needs `carrier_prompt`), then fed through the same `generate_inverse()` in place of a real seed — the empirical error distribution still drives the injected errors. |

**`class_conditional` (Spam)** — classification is inherently label → text: sample a
target class from the balance (`class_balance`, default the real dataset's empirical
`P(SPAM)`), then synthesize an example of that class. Because **both classes are
LLM-authored** in every cell, a classifier can't separate them on "was this written by
an LLM" artifacts.

| | `seedless: false` (seeded) | `seedless: true` |
|---|---|---|
| **`mode: inverse`** (default) | `seed_policy="cross_class"`: SPAM = inject an empirically-profiled mix of spam **signals** (link, money, ALL-CAPS, urgency, keywords) into a real HAM seed; HAM = paraphrase a real HAM seed. | Same cross-class flow, but the HAM seed is a carrier synthesized from the profile via `generate_carriers()` (needs `carrier_prompt`) instead of a real message. |
| **`mode: forward`** | `seed_policy="same_class"`: each class imitates within itself — rewrites a real labeled seed of that class (SPAM or HAM) into a new message of the same kind. Needs `forward_prompt`. | `seed_policy="none"`: per-label profile-sampled content specs, no real seed at all. Needs `seedless_class_prompts`. |

### Setting it

    generation:
      mode: "inverse"        # forward | inverse — see the tables above
      seedless: false         # true = drop real seeds, generate from the profile only

CLI overrides: `--mode forward|inverse`, `--seedless`/`--no-seedless`.

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
      --output framework/data/profiles/gec_profile.json

`--topics` is required for seedless generation — it adds the LLM-driven `topics`
(GEC) / `topics_per_label` (Spam) block the content-spec sampler needs; without it,
`_load_generation_profile` raises `RuntimeError` naming the missing block. The default
profile path is `framework/data/profiles/<task>_profile.json`; override per-run with
`generation.profile_path`.

| Task | strategy | cells |
|------|----------|------------|
| GEC  | `corruption` | forward / inverse × seeded / seedless (see table above) |
| Spam | `class_conditional` | inverse / forward × seeded / seedless; class balance from `class_balance` |

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
    # generation_models: lives in each task's compare.yaml (see its comments)
    python -m scripts.compare_models --config framework/configs/gec/compare.yaml

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

## How to Add a New Task

1. Create `framework/tasks/<task>/task.py` subclassing `BaseTask` and
   implement `get_error_types`, `get_prompt_instruction`, `get_evaluators`,
   `get_evaluator_fns`, `get_model` (and optionally `get_judge_prompt`).
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
    real_sample.json   - the real reference sample (classification tasks)
    profile.json       - {real, generated, fidelity} (classification tasks)
    plots/             - generated_vs_real_<model>.png, run_variance_<model>.png, fidelity.png (classification tasks)

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

`results.json` is rewritten after **every** run, so a crash or Ctrl-C in run N keeps the
aggregated results of runs 1..N-1. A run that generates zero usable samples aborts
instead of writing misleading all-zero scores.

The LLM-as-judge filter is **opt-in**: no `judge:` block (or `judge.enabled: false`)
means judging is skipped.

---

## Current Tasks

GEC (Grammatical Error Correction) — implemented (corruption: forward + inverse)
Spam Detection — implemented (class-conditional generation + real baseline + fidelity)
Hate Speech Detection — planned
Sentiment Analysis — planned

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
