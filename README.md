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
        data_loading.py          - dataset source resolution + local file loaders (m2/csv/tsv)
        requirements.txt         - Python dependencies
        configs/
            config.yaml          - dataset, generator, task models, output
            tasks/
                gec.json         - GEC task config (error types, prompts, evaluators, model params)
            tasks/
                spam.json        - Spam task config (forward + inverse)
        tasks/
            base_task.py         - abstract task template
            gec/task.py          - Grammatical Error Correction task (forward + inverse)
            spam/task.py         - Spam Detection task (forward + inverse)
        generators/              - LLM that creates synthetic corrupted sentences
            base_generator.py    - shared generate()/generate_inverse() loop
            openai_generator.py  - OpenAI / Groq / OpenRouter / Mistral (OpenAI-compatible)
            anthropic_generator.py  - Anthropic / MiniMax (Anthropic-compatible)
            google_generator.py
        profiling/               - empirical error-distribution profilers (inverse mode)
            errant_distribution.py  - ERRANT-based GEC error distribution
            spam_distribution.py  - spam-signal-based spam error distribution
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
   - `dataset`         — `source` (huggingface | local) + `sample_size` (size of the
                         pool loaded from the benchmark). Per-source settings live in
                         nested blocks that can both stay filled in — switching source
                         is a one-field change:

         dataset:
           source: local            # huggingface | local
           sample_size: 300
           huggingface: {name: "deysi/spam-detection-dataset", split: "train"}
           local: {path: "framework/data/spam/sms_spam_ham_300.csv", format: csv}

     Local formats: `m2` (GEC benchmarks like FCE/CoNLL-14, annotator 0's edits),
     `csv`, `tsv` (header row, fields matched by the task's `parse_row`, e.g.
     `label`/`text` for spam). `format` is optional when the file extension says it.
     For inverse spam, SPAM-signal profiling also reads the local file.
   - `generation`      — `mode` (forward | inverse), generator provider, model,
                         temperature, `num_runs`, `sample_size` (samples actually
                         used per run; must be ≤ `dataset.sample_size`)
   - `task.name`       — `gec` or `spam`
   - `task_models`     — list of models to evaluate
   - `output.results_path` — where to write results JSON (overwritten each invocation —
                         see "Comparing generation models" below)
   - `output.generated_data_dir` — where each run's synthetic data is archived

   > Sampling is deterministic: `load_real_data` takes the first `dataset.sample_size`
   > matching rows (no shuffle/seed). As long as the dataset settings (source, HF
   > name/split or local path, `sample_size`) are unchanged, every invocation sees the
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
        --output results.json \
        --no-judge

Use `--config <path>` to point at a different YAML. `--judge/--no-judge`
toggles the LLM-as-judge filter; `--output` sets `output.results_path`.

> Note: `python framework/main.py` will NOT work — `framework` must be
> importable as a package, so use `python -m framework.main`.

The config is validated up front: missing required keys, `num_runs < 1`,
`generation.sample_size > dataset.sample_size`, an unknown `generation.mode`,
or a missing API key for the selected provider all abort before any API call
with an error naming the offending config path.

---

## Generation Modes

Set `generation.mode` in the config:

- **`forward`** (default) — the generator rewrites each source sentence into a
  corrupted variant, choosing an error type itself. Supported by **GEC and Spam**.
- **`inverse`** — the generator corrupts a known-clean source (`inverse.source_field`)
  according to an **empirical error distribution** profiled from the real data, so the
  injected error mix matches the benchmark. Supported by **GEC and Spam** (see matrix below).

| Task | forward | inverse |
|------|:-------:|:-------:|
| GEC  | ✅ | ✅ (ERRANT-profiled distribution) |
| Spam | ✅ | ✅ (spam-signal-profiled distribution) |

> Inverse spam injects an empirically-profiled mix of spam **signals** (link,
> money, ALL-CAPS, urgency, keywords) into legitimate (HAM) messages, and also
> scores each clean source as a HAM negative so precision/recall/f1/fpr stay
> meaningful. Set `generation.inverse.source_field: "incorrect"` (the HAM text).

---

## Comparing Generation Models (same benchmark sample)

Use the multi-model driver to run several generation models over the identical
sample in one command:

    cd live-eval
    # add a `generation_models:` list to your config (see config.yaml comments)
    python -m scripts.compare_models --config framework/configs/config.yaml

It writes one `results/<task>_<mode>_<provider>_<model>.json` per model, a combined
`results/comparison_<task>_<mode>.json`, and prints a comparison table. The same
benchmark sample is guaranteed by deterministic first-N sampling, so keep
`dataset.*`, `generation.mode`, and both `sample_size` values constant across entries.

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

Results are written to the path in `output.results_path` (default
`results.json`). The file has two top-level keys:

- `meta` — provenance: timestamp, task, mode, generator provider/model,
  dataset, sample sizes, the number of samples actually **scored** per run
  (after generation skips/judging, plus any task-added negatives — e.g. spam's
  HAM negatives), and the judge used (or `null`). `meta.partial` is `true`
  while runs are still outstanding.
- `results` — per-model scores; each evaluator reports `mean ± std` across
  runs — high `std` reveals model instability on unseen data.

The file is rewritten after **every** run, so a crash or Ctrl-C in run N
keeps the aggregated results of runs 1..N-1. A run that generates zero usable
samples aborts instead of writing misleading all-zero scores.

The LLM-as-judge filter is **opt-in**: no `judge:` block (or
`judge.enabled: false`) means judging is skipped.

The synthetic data itself is archived per run under
`output.generated_data_dir` (default `framework/data/generated/`) as
`<task>/<session>_run<N>.json`. It is never reused for evaluation.

---

## Current Tasks

GEC (Grammatical Error Correction) — implemented (forward + inverse)
Spam Detection — implemented (forward + inverse)
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
