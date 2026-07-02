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
Each run's data is archived under `framework/data/generated/<task>/`
so it can be inspected later.

---

## Project Structure

    framework/
        main.py                  - entry point
        pipeline.py              - GET loop (Generate, Evaluate, Trash)
        requirements.txt         - Python dependencies
        configs/
            config.yaml          - dataset, generator, task models, output
            tasks/
                gec.json         - GEC task config (error types, prompts, evaluators, model params)
            tasks/
                spam.json        - Spam task config (forward only — no inverse keys)
        tasks/
            base_task.py         - abstract task template
            gec/task.py          - Grammatical Error Correction task (forward + inverse)
            spam/task.py         - Spam Detection task (forward only)
        generators/              - LLM that creates synthetic corrupted sentences
            base_generator.py    - shared generate()/generate_inverse() loop
            openai_generator.py  - OpenAI / Groq / OpenRouter / Mistral (OpenAI-compatible)
            anthropic_generator.py  - Anthropic / MiniMax (Anthropic-compatible)
            google_generator.py
        profiling/               - empirical error-distribution profilers (inverse mode)
            errant_distribution.py  - ERRANT-based GEC error distribution
        models/gec/              - GEC models under evaluation
            seq2seq.py (t5/gec_v1/coedit), claude.py
        models/spam/             - Spam models under evaluation
            roberta.py, bert_tiny.py
        evaluators/              - scoring functions applied to model predictions
            gleu.py              - GLEU score
            gec/                 - errant, errant_dist, cola, correction_extent, n_edits
            classification/      - accuracy, precision, recall, f1, fpr (spam)
        data/
            generated/           - archived synthetic data, one JSON per run (gitignored)

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
   - `dataset`         — HuggingFace dataset name + split + `sample_size` (size of the
                         pool loaded from the benchmark)
   - `generation`      — `mode` (forward | inverse), generator provider, model,
                         temperature, `num_runs`, `sample_size` (samples actually
                         used per run; must be ≤ `dataset.sample_size`)
   - `task.name`       — `gec` or `spam`
   - `task_models`     — list of models to evaluate
   - `output.results_path` — where to write results JSON (overwritten each invocation —
                         see "Comparing generation models" below)
   - `output.generated_data_dir` — where each run's synthetic data is archived

   > Sampling is deterministic: `load_real_data` takes the first `dataset.sample_size`
   > matching rows (no shuffle/seed). As long as `dataset.name/split/sample_size` are
   > unchanged, every invocation sees the **same benchmark sample** — which is what makes
   > cross-model comparison fair.

---

## How to Run

Run as a module from the `live-eval/` directory (the parent of `framework/`):

    cd live-eval
    python -m framework.main

CLI flags override values in `config.yaml`:

    python -m framework.main \
        --task gec \
        --provider anthropic \
        --model claude-haiku-4-5 \
        --runs 3 \
        --sample-size 20

Use `--config <path>` to point at a different YAML.

> Note: `python framework/main.py` will NOT work — `framework` must be
> importable as a package, so use `python -m framework.main`.

> CLI flags currently cover `--task/--provider/--model/--runs/--sample-size`
> only. There is **no** flag for `output.results_path`, `generation.mode`, or the
> judge — set those in the YAML (or a per-run `--config`).

---

## Generation Modes

Set `generation.mode` in the config:

- **`forward`** (default) — the generator rewrites each source sentence into a
  corrupted variant, choosing an error type itself. Supported by **GEC and Spam**.
- **`inverse`** — the generator corrupts a known-clean source (`inverse.source_field`)
  according to an **empirical error distribution** profiled from the real data, so the
  injected error mix matches the benchmark. Supported by **GEC only** (see matrix below).

| Task | forward | inverse |
|------|:-------:|:-------:|
| GEC  | ✅ | ✅ (ERRANT-profiled distribution) |
| Spam | ✅ | ❌ not implemented |

> `mode: inverse` + `task: spam` will **fail fast** — `SpamTask` has no inverse prompt
> or error vocabulary, so `load_error_distribution` raises `ValueError`. Adding inverse
> spam requires an `inverse_prompt` + `error_descriptions` (and ideally a profiler) on
> the spam task; the generator/pipeline layers already support it.

---

## Comparing Generation Models (same benchmark sample)

A single invocation runs **one** generation model `num_runs` times over one fixed sample
and reports mean ± std. To compare **different generation models on the same sample**,
run the pipeline once per model. Because sampling is deterministic (first N rows), each
invocation evaluates the identical benchmark sample.

**Important:** `output.results_path` is overwritten on every invocation. Give each model
its own output file (or config) so results are not clobbered:

    cd live-eval

    # Model A
    python -m framework.main --task gec --provider anthropic \
        --model claude-haiku-4-5 --runs 3 --sample-size 20 \
        --config framework/configs/config.yaml       # writes output.results_path from YAML

    # Model B — use a separate config whose output.results_path differs
    python -m framework.main --config framework/configs/config.gpt.yaml

Keep `task`, `dataset.*`, `generation.mode`, and both `sample_size` values identical
across the invocations so every model is scored on the same sentences. The per-run
synthetic data for each model is archived under
`output.generated_data_dir/<task>/<session>_run<N>.json` for later inspection.

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

Results are written to the path in `output.results_path` (default
`results.json`) after all runs finish. Each evaluator reports `mean ± std`
across runs — high `std` reveals model instability on unseen data.

The synthetic data itself is archived per run under
`output.generated_data_dir` (default `framework/data/generated/`) as
`<task>/<session>_run<N>.json`. It is never reused for evaluation.

---

## Current Tasks

GEC (Grammatical Error Correction) — implemented (forward + inverse)
Spam Detection — implemented (forward only)
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
