import json
import os
import numpy as np
from datasets import load_dataset
from framework.tasks.base_task import BaseTask


# ── Generator registry ───────────────────────────────────────
# Add new providers here as they are implemented.

# Anthropic-compatible providers — use the Anthropic SDK against a custom base_url.
_ANTHROPIC_BASE_URLS = {
    "minimax": "https://api.minimax.io/anthropic",
    # "anthropic" → None (default endpoint)
}


def load_generator(config: dict):
    provider = config["provider"]
    if provider in ("openai", "groq", "openrouter", "mistral"):
        from framework.generators.openai_generator import OpenAIGenerator
        return OpenAIGenerator(config)
    elif provider in ("anthropic", "minimax"):
        from framework.generators.anthropic_generator import AnthropicGenerator
        if not config.get("base_url") and provider in _ANTHROPIC_BASE_URLS:
            config = {**config, "base_url": _ANTHROPIC_BASE_URLS[provider]}
        return AnthropicGenerator(config)
    elif provider == "google":
        from framework.generators.google_generator import GoogleGenerator
        return GoogleGenerator(config)
    raise ValueError(
        f"Unknown provider: '{provider}'. "
        f"Supported: openai, groq, openrouter, mistral, anthropic, minimax, google."
    )


# ── Judge generator ──────────────────────────────────────────

def _build_judge_call(config: dict, main_generator):
    """
    Return a callable(prompt: str) -> str for the LLM-as-judge step.

    Resolution order:
      1. config.judge with enabled != false → load a separate generator
      2. otherwise → fall back to the main generator (current behavior)
      3. if config.judge.enabled is false → return None (judging skipped)
    """
    judge_cfg = config.get("judge")
    if judge_cfg is None:
        return main_generator._call_api
    if judge_cfg.get("enabled", True) is False:
        return None
    if not judge_cfg.get("provider") or not judge_cfg.get("model"):
        print("[WARN] judge block missing provider/model — falling back to main generator.")
        return main_generator._call_api
    print(f"Judge    : {judge_cfg['provider']} / {judge_cfg['model']}")
    judge_generator = load_generator(judge_cfg)
    return judge_generator._call_api


# ── Task registry ────────────────────────────────────────────
# Add new tasks here as they are implemented.

def load_task(task_name: str) -> BaseTask:
    if task_name == "gec":
        from framework.tasks.gec.task import GECTask
        return GECTask()
    raise ValueError(
        f"Unknown task: '{task_name}'. "
        f"Register it in pipeline.load_task() and add configs/tasks/{task_name}.json."
    )


# ── Dataset loading ──────────────────────────────────────────

def _get_field(row: dict, candidates: list[str]):
    """Return the first matching field from a dataset row, with fallback."""
    for key in candidates:
        if key in row and row[key]:
            return row[key]
    return next((v for v in row.values() if isinstance(v, str) and v), None)


def load_real_data(config: dict) -> list[dict]:
    """
    Load real sentences from a HuggingFace dataset or local file.
    Supports streaming (set dataset.streaming: true) for large datasets.
    Field names are auto-detected with configurable overrides.
    """
    ds_config = config["dataset"]
    sample_size = ds_config["sample_size"]

    if ds_config.get("source", "huggingface") == "local":
        raise NotImplementedError(
            "Local dataset loading is not yet implemented. "
            "Contribute it in pipeline.load_real_data()."
        )

    print(f"Loading dataset: {ds_config['name']} ...")
    hf_token = (
        ds_config.get("hf_token")
        or (config.get("api_keys") or {}).get("huggingface")
        or os.getenv("HF_TOKEN")
    )
    dataset = load_dataset(
        ds_config["name"],
        split=ds_config["split"],
        streaming=ds_config.get("streaming", False),
        token=hf_token or None,
    )

    # Configurable field names with schema-detection fallback
    input_field   = ds_config.get("input_field",   "input")
    correct_field = ds_config.get("correct_field", "output")

    samples = []
    for i, row in enumerate(dataset):
        if i >= sample_size:
            break
        incorrect = _get_field(row, [input_field,   "input", "text", "incorrect"])
        correct   = _get_field(row, [correct_field, "output", "correct", "target"])
        if incorrect and correct:
            samples.append({"incorrect": incorrect, "correct": correct})

    print(f"Loaded {len(samples)} real samples.")
    return samples


# ── Verification ─────────────────────────────────────────────

def verify(synthetic_data: list[dict]) -> list[dict]:
    """Filter out generations where the LLM failed to introduce an error."""
    verified = []
    for item in synthetic_data:
        if not item["corrupted"] or not item["original"]:
            continue
        if item["corrupted"].strip() == item["original"].strip():
            print("[SKIP] Unchanged sentence.")
            continue
        if len(item["corrupted"].split()) < 3:
            print("[SKIP] Too short.")
            continue
        verified.append(item)
    print(f"Verified: {len(verified)}/{len(synthetic_data)} samples passed.")
    return verified


# ── Aggregation ──────────────────────────────────────────────

def aggregate(all_run_scores: list[dict]) -> dict:
    """
    Compute mean ± std across N runs.
    High std reveals model instability on unseen data (see Paulreich 2025, p.8).
    """
    final = {}
    for model_name in all_run_scores[0]:
        final[model_name] = {}
        for metric in all_run_scores[0][model_name]:
            raw = all_run_scores[0][model_name][metric]
            if isinstance(raw, dict):
                final[model_name][metric] = {}
                for sub in raw:
                    values = [run[model_name][metric][sub] for run in all_run_scores]
                    final[model_name][metric][sub] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std":  round(float(np.std(values)),  4),
                    }
            else:
                values = [run[model_name][metric] for run in all_run_scores]
                final[model_name][metric] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std":  round(float(np.std(values)),  4),
                }
    return final


# ── Main pipeline ─────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """
    Run the full GET pipeline (Generate → Evaluate → Trash) N times,
    then aggregate results as mean ± std across runs.
    """
    real_data = load_real_data(config)

    task       = load_task(config["task"]["name"])
    generator  = load_generator(config["generation"])
    judge_call = _build_judge_call(config, generator)
    metric_fns = task.get_metric_fns()

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
            sample_size=config["generation"]["sample_size"],
            judge_prompt=task.get_judge_prompt() if judge_call else None,
            judge_call=judge_call,
        )
        synthetic = verify(synthetic)
        corrupted_sentences = [item["corrupted"] for item in synthetic]

        # ── EVALUATE ──────────────────────────────────────
        run_scores = {}
        for model_config in config["task_models"]:
            evaluator   = task.get_evaluator(model_config)
            predictions = evaluator.predict(corrupted_sentences)
            results = [
                {**item, "prediction": pred}
                for item, pred in zip(synthetic, predictions)
            ]
            run_scores[model_config["name"]] = {
                name: metric_fns[name](results) for name in task.get_metrics()
            }
            for name, score in run_scores[model_config["name"]].items():
                print(f"  {model_config['name']}  {name}: {score}")

        all_run_scores.append(run_scores)

        # ── TRASH ─────────────────────────────────────────
        synthetic.clear()
        print("\nSynthetic data trashed.")

    # ── AGGREGATE ─────────────────────────────────────────
    final = aggregate(all_run_scores)

    with open(config["output"]["results_path"], "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {config['output']['results_path']}")

    return final
