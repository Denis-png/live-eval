# ============================================================
# Pipeline — GET Methodology
# ============================================================
# This is the core of the framework.
# It runs the full GET loop N times:
#
#   For each run:
#     1. GENERATE — create synthetic corrupted sentences
#     2. EVALUATE — run task models on synthetic data
#     3. TRASH    — delete synthetic data
#
# At the end, compute mean ± std across all runs.
# This reveals how stable each model really is on unseen data.
#
# Based on: Paulheim, H. (2025). Towards Evaluating Knowledge
# Graph Construction and Ontology Learning with LLMs without
# Test Data Leakage. Section 4.2, Page 8.
# ============================================================

import json
import numpy as np
from datasets import load_dataset
from framework.tasks.gec_task import GECTask
from framework.generators.llm_generator import LLMGenerator
from framework.evaluators.gec_evaluator import GECEvaluator
from framework.metrics.gleu import compute_gleu
from framework.metrics.errant_metric import compute_errant


def load_real_data(config: dict) -> list[dict]:
    """
    Load real sentences from HuggingFace dataset.
    These are used as source material for the generator.
    We do NOT evaluate on these directly.
    """
    print(f"Loading dataset: {config['dataset']['name']} ...")
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"]
    )
    samples = []
    for i in range(config["dataset"]["sample_size"]):
        row = dataset[i]
        samples.append({
            "incorrect": row["input"],
            "correct":   row["output"]
        })
    print(f"Loaded {len(samples)} real samples.")
    return samples


def verify(synthetic_data: list[dict]) -> list[dict]:
    """
    Filter out bad generations.
    Remove samples where LLM failed to introduce an error.
    """
    verified = []
    for item in synthetic_data:
        if not item["corrupted"] or not item["original"]:
            continue
        if item["corrupted"].strip() == item["original"].strip():
            print(f"[SKIP] Unchanged sentence.")
            continue
        if len(item["corrupted"].split()) < 3:
            print(f"[SKIP] Too short.")
            continue
        verified.append(item)
    print(f"Verified: {len(verified)}/{len(synthetic_data)} samples passed.")
    return verified


def aggregate(all_run_scores: list[dict]) -> dict:
    """
    Compute mean ± std across N runs.

    As noted in the paper (Page 8):
    'The standard deviation is often considerable, showing that
    the approaches are not very stable, that good results can
    also be the result of a lucky coincidence.'
    """
    final = {}
    for model_name in all_run_scores[0]:
        final[model_name] = {}
        for metric in all_run_scores[0][model_name]:
            raw = all_run_scores[0][model_name][metric]
            if isinstance(raw, dict):
                final[model_name][metric] = {}
                for sub in raw:
                    values = [run[model_name][metric][sub]
                              for run in all_run_scores]
                    final[model_name][metric][sub] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std":  round(float(np.std(values)),  4)
                    }
            else:
                values = [run[model_name][metric]
                          for run in all_run_scores]
                final[model_name][metric] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std":  round(float(np.std(values)),  4)
                }
    return final


def run_pipeline(config: dict) -> dict:
    """
    Run the full GET pipeline.

    Args:
        config: loaded from config.yaml

    Returns:
        final aggregated results (mean ± std per model per metric)
    """
    # Load real data once
    real_data = load_real_data(config)

    # Initialize task and generator
    task      = GECTask()
    generator = LLMGenerator(config["generation"])

    all_run_scores = []
    num_runs = config["generation"]["num_runs"]

    for run_idx in range(num_runs):
        print(f"\n{'='*50}")
        print(f"RUN {run_idx + 1} / {num_runs}")
        print(f"{'='*50}")

        # ── GENERATE ──────────────────────────────────────
        synthetic = generator.generate(
            real_samples=real_data,
            error_types=task.get_error_types(),
            prompt_instruction=task.get_prompt_instruction(),
            sample_size=config["generation"]["sample_size"]
        )
        synthetic = verify(synthetic)
        corrupted_sentences = [item["corrupted"] for item in synthetic]

        # ── EVALUATE ──────────────────────────────────────
        run_scores = {}
        for model_config in config["task_models"]:
            evaluator   = GECEvaluator(model_config)
            predictions = evaluator.predict(corrupted_sentences)

            results = [
                {**item, "prediction": pred}
                for item, pred in zip(synthetic, predictions)
            ]

            run_scores[model_config["name"]] = {
                "gleu":   compute_gleu(results),
                "errant": compute_errant(results)
            }
            print(f"  {model_config['name']}:")
            print(f"    GLEU   : {run_scores[model_config['name']]['gleu']}")
            print(f"    ERRANT : {run_scores[model_config['name']]['errant']}")

        all_run_scores.append(run_scores)

        # ── TRASH ─────────────────────────────────────────
        synthetic.clear()
        print("\n Synthetic data trashed.")

    # ── AGGREGATE ─────────────────────────────────────────
    final = aggregate(all_run_scores)

    with open(config["output"]["results_path"], "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n Results saved to {config['output']['results_path']}")

    return final