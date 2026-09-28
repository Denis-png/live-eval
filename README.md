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

For everything below the basics — configuration reference, the four generation
cells, calibration, the LLM judge, plotting, comparing generation models,
cross-session analysis, output format, troubleshooting, project layout, and how
to add a new task — see **[docs/COOKBOOK.md](docs/COOKBOOK.md)**.

---

## Setup

1. Install Python deps (run from `live-eval/`):

       pip install -r framework/requirements.txt
       python -m spacy download en_core_web_sm    # required by ERRANT

2. Build the benchmark files. `framework/data/` is gitignored, so none of them ships
   with the repo; each script downloads its source and writes the file the task's
   config points at:

       python -m scripts.benchmarks.prepare_gec_benchmark        # FCE v2.1 test split
       python -m scripts.benchmarks.prepare_spam_benchmark       # 300 fixed SMS Spam Collection rows
       python -m scripts.benchmarks.prepare_sentiment_benchmark  # first 150 TweetEval test tweets
       python -m scripts.benchmarks.prepare_taxonomy_benchmark   # Pizza ontology, pinned commit

   (`prepare_taxonomy_benchmark` also converts any other OWL/RDF ontology — see the
   Cookbook.)

3. Copy `live-eval/example.env` → `live-eval/.env` and fill in the API keys you
   need. `main.py` loads it automatically — you only need the keys for providers
   you actually use.

4. Point at a task's config: `framework/configs/<task>/config.yaml`. Each task's
   config carries only the fields that task reads; there is no shared root config.
   The full field-by-field reference (dataset sources/formats, generation cells,
   class balance, evaluation flags) is in the Cookbook's **Configuration** section.

---

## How to Run

Run as a module from the `live-eval/` directory (the parent of `framework/`):

    cd live-eval
    python -m framework.main --config framework/configs/gec/config.yaml

`--config` is required. CLI flags override values in the YAML:

    python -m framework.main \
        --config framework/configs/gec/config.yaml \
        --mode inverse \
        --seedless \
        --runs 3 \
        --sample-size 20 \
        --no-judge \
        --no-real-baseline

> Note: `python framework/main.py` will NOT work — `framework` must be
> importable as a package, so use `python -m framework.main`.

The config is validated up front: missing required keys, `num_runs < 1`, an
unknown `generation.mode`, or a missing API key all abort before any API call.

Generation, evaluation, profiling, calibration and plotting can each be run as
separate standalone stages — see the Cookbook's **Running Stages Separately**.

---

## Current Tasks

GEC (Grammatical Error Correction) — implemented (corruption: forward + inverse)
Spam Detection — implemented (class-conditional generation + real baseline + fidelity)
Sentiment Analysis — implemented (corruption: forward + inverse, seeded + seedless; label-balance fidelity)
Taxonomy Induction — implemented (structured generation + subclass evaluation + structural fidelity)
Hate Speech Detection — planned

## Current Models (GEC)

vennify/t5-base-grammar-correction — T5 fine-tune
prithivida/grammar_error_correcter_v1 — seq2seq
grammarly/coedit-large — instruction fine-tune

## Current Models (Spam)

mshenoda/roberta-spam
mariagrandury/roberta-base-finetuned-sms-spam-detection
mrm8488/bert-tiny-finetuned-sms-spam-detection
wesleyacheng/sms-spam-classification-with-bert

## Current Models (Sentiment)

finiteautomata/bertweet-base-sentiment-analysis
lxyuan/distilbert-base-multilingual-cased-sentiments-student

## Current Models (Taxonomy)

lexical — longest-suffix heuristic baseline, no LLM call, cannot memorise ontologies
star — everything under one root, the structural floor
minimax-m3 — LLM task model (same model used for generation by default; reported separately)

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
malformed vs failed predictions
