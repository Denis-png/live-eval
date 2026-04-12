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

---

## Project Structure

    framework/
    config.yaml          - all parameters (model, dataset, metrics)
    main.py              - entry point, run this file
    pipeline.py          - GET loop (Generate, Evaluate, Trash)
    tasks/
        base_task.py     - abstract template
        gec_task.py      - Grammatical Error Correction
    generators/
        base_generator.py
        llm_generator.py - LLaMA via Groq API (default)
    evaluators/
        base_evaluator.py
        gec_evaluator.py - T5 and Gramformer
    metrics/
        gleu.py          - GLEU score
        errant_metric.py - ERRANT precision/recall/F0.5

---

## How to Use

1. Install dependencies:
    pip install transformers datasets torch errant spacy langchain-groq pyyaml numpy nltk
    python -m spacy download en_core_web_sm

2. Open framework/config.yaml and set:
    - Your API key
    - Dataset name
    - Generator model
    - Task models to evaluate
    - Number of runs

3. Run:
    python framework/main.py

---

## How to Add a New Task

Create a new file in framework/tasks/, for example hate_speech_task.py
Implement the three methods: get_error_types, get_prompt_instruction, get_metrics
Update config.yaml with the new task name

---

## Results

Results are saved to results.json after each run.
The std value is the key scientific output —
it shows how consistent the model is on truly unseen data.

---

## Current Tasks

GEC (Grammatical Error Correction) — implemented
Hate Speech Detection — ?
Spam Detection — ?
Sentiment Analysis — ?

## Current Metrics

GLEU — measures fluency of correction
ERRANT — measures error detection accuracy (precision, recall, F0.5)
Accuracy and F1 — planned for future classification tasks