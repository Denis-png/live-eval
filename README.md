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

3. Edit `framework/configs/config.yaml`:
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
                         `mode` (forward | inverse) applies to **corruption** tasks
                         (GEC); spam is **class-conditional** and ignores it. Spam
                         also reads `class_balance` (`empirical` | float = P(SPAM))
                         and `seed_field` (the real-data field seeded from).
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
    python -m framework.main

CLI flags override values in `config.yaml`:

    python -m framework.main \
        --task gec \
        --mode inverse \
        --provider anthropic \
        --model claude-haiku-4-5 \
        --runs 3 \
        --sample-size 20 \
        --output framework/data/runs \
        --no-judge \
        --no-real-baseline

Use `--config <path>` to point at a different YAML. `--judge/--no-judge`
toggles the LLM-as-judge filter; `--real-baseline/--no-real-baseline` toggles
the real-benchmark baseline; `--output` sets `output.base_dir`.

> Note: `python framework/main.py` will NOT work — `framework` must be
> importable as a package, so use `python -m framework.main`.

The config is validated up front: missing required keys, `num_runs < 1`,
an unknown `generation.mode`, or a missing API key for the selected provider
all abort before any API call with an error naming the offending config path.

---

## Generation Strategies

Each task declares a **generation strategy** (`task.get_generation_strategy()`); the
pipeline dispatches on it. `generation.mode` only applies to the corruption strategy.

- **`corruption`** (GEC) — corrupt a source text. `generation.mode` selects:
  - **`forward`** — the generator rewrites each source sentence into a corrupted
    variant, choosing an error type itself.
  - **`inverse`** — the generator corrupts a known-clean source
    (`inverse.source_field`) according to an **empirical error distribution**
    (ERRANT-profiled) so the injected error mix matches the benchmark.

- **`class_conditional`** (Spam) — spam detection is classification (text → label), so
  generation is inherently **label → text**: sample a target class from the balance
  (`class_balance`, default the real dataset's empirical `P(SPAM)`), then synthesize an
  example of that class. **SPAM** is produced by injecting an empirically-profiled mix
  of spam **signals** (link, money, ALL-CAPS, urgency, keywords) into a legitimate (HAM)
  seed; **HAM** is produced by paraphrasing a HAM seed. Because **both classes are
  LLM-authored**, a classifier can't separate them on "was this written by an LLM"
  artifacts. Spam **ignores `generation.mode`**.

| Task | strategy | generation |
|------|----------|------------|
| GEC  | `corruption` | forward (free) or inverse (ERRANT-profiled) |
| Spam | `class_conditional` | SPAM = signal injection, HAM = paraphrase; class balance from `class_balance` |

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

## Comparing Generation Models (same benchmark sample)

Use the multi-model driver to run several generation models over the identical
sample in one command:

    cd live-eval
    # add a `generation_models:` list to your config (see config.yaml comments)
    python -m scripts.compare_models --config framework/configs/config.yaml

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
2. Create `framework/configs/tasks/<task>.json` with error types, prompts,
   evaluators list, and per-model inference params.
3. Register the task in `framework/pipeline.py::load_task()`.
4. Add model classes under `framework/models/<task>/` and evaluator
   functions under `framework/evaluators/<task>/`.
5. Set `task.name: <task>` in `configs/config.yaml`.

---

## Results

Each run session gets its own directory under `output.base_dir/<task>/<session>/`:

    results.json       - {"meta": <provenance>, "results": <scores>}
    generated/
        run_1.json …   - each run's synthetic data (never reused for eval)
    real_sample.json   - the real reference sample (classification tasks)
    profile.json       - {real, generated, fidelity} (classification tasks)
    plots/             - reserved for figures

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
