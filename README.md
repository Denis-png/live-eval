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
            base_task.py         - abstract task template
            gec/task.py          - Grammatical Error Correction task
        generators/              - LLM that creates synthetic corrupted sentences
            openai_generator.py  - OpenAI / Groq / OpenRouter / Mistral (OpenAI-compatible)
            anthropic_generator.py
            google_generator.py
        models/gec/              - models under evaluation
            t5.py, gec_v1.py, coedit.py, claude.py
        evaluators/              - scoring functions applied to model predictions
            gleu.py              - GLEU score
            gec/                 - errant, errant_dist, cola, correction_extent, n_edits
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
   - `dataset`         — HuggingFace dataset name + split + sample size
   - `generation`      — generator provider, model, temperature, num_runs, sample_size
   - `task.name`       — `gec` (only task currently implemented)
   - `task_models`     — list of models to evaluate
   - `output.results_path` — where to write results JSON
   - `output.generated_data_dir` — where each run's synthetic data is archived

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

GEC (Grammatical Error Correction) — implemented
Hate Speech Detection — planned
Spam Detection — planned
Sentiment Analysis — planned

## Current Evaluators (GEC)

GLEU — fluency of correction
ERRANT — precision / recall / F0.5 of edits
errant_dist — distribution of edit categories
CoLA — linguistic acceptability of the prediction
correction_extent — how much of the input was edited
n_edits — raw edit count
