import json
import math
import os
from datetime import datetime

import numpy as np
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
        return main_generator.call_api
    if judge_cfg.get("enabled", True) is False:
        return None
    if not judge_cfg.get("provider") or not judge_cfg.get("model"):
        print("[WARN] judge block missing provider/model — falling back to main generator.")
        return main_generator.call_api
    print(f"Judge    : {judge_cfg['provider']} / {judge_cfg['model']}")
    judge_generator = load_generator(judge_cfg)
    return judge_generator.call_api


# ── Task registry ────────────────────────────────────────────
# Add new tasks here as they are implemented.

def load_task(task_name: str) -> BaseTask:
    if task_name == "gec":
        from framework.tasks.gec.task import GECTask
        return GECTask()
    elif task_name == "spam":
        from framework.tasks.spam.task import SpamTask
        return SpamTask()
    raise ValueError(
        f"Unknown task: '{task_name}'. "
        f"Register it in pipeline.load_task() and add configs/tasks/{task_name}.json."
    )


# ── Dataset loading ──────────────────────────────────────────

def _get_field(row: dict, candidates: list[str]):
    """Return the first non-empty candidate field from a dataset row.

    Returns None if no candidate matches — callers skip such rows. We do NOT
    fall back to "the first string column" because that silently pulls in the
    wrong field on an unexpected schema and corrupts the whole sample set."""
    for key in candidates:
        if key in row and row[key]:
            return row[key]
    return None


def load_real_data(config: dict, task: BaseTask) -> list[dict]:
    """
    Load real samples from a HuggingFace dataset or local file.
    Supports streaming (set dataset.streaming: true) for large datasets.
    Field parsing and row filtering is delegated to task.parse_row().
    """
    ds_config = config["dataset"]
    sample_size = ds_config["sample_size"]

    if ds_config.get("source", "huggingface") == "local":
        raise NotImplementedError(
            "Local dataset loading is not yet implemented. "
            "Contribute it in pipeline.load_real_data()."
        )

    from datasets import load_dataset  # lazy: keeps pipeline importable without HF deps

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

    samples = []
    for row in dataset:
        parsed = task.parse_row(row)
        if parsed is not None:
            samples.append(parsed)
        if len(samples) >= sample_size:
            break

    print(f"Loaded {len(samples)} real samples.")
    return samples


# ── Error distribution (PLACEHOLDER seam) ────────────────────
# NOTE: This is a temporary placeholder. The real benchmark-preprocessing
# module (separate work) will replace load_error_distribution's body to compute
# the empirical error distribution from `real_data`. The signature is fixed so
# the generator/task layers never change.

def _poisson_pmf(mean: float, n_min: int = 1, n_max: int = 5) -> dict[int, float]:
    """Normalized Poisson PMF restricted to n in [n_min, n_max]."""
    raw = {
        n: (mean ** n) * math.exp(-mean) / math.factorial(n)
        for n in range(n_min, n_max + 1)
    }
    total = sum(raw.values()) or 1.0
    return {n: p / total for n, p in raw.items()}


def load_error_distribution(config: dict, real_data: list[dict], task) -> dict:
    """Return {"type_dist": {key: prob}, "count_dist": {n: prob}} for inverse mode.

    Delegates to task.profile_error_distribution to derive an empirical
    distribution from real_data. Falls back to a uniform type distribution over
    the task's category vocabulary plus a Poisson errors-per-sentence
    distribution when the task has no empirical profiler or too little data."""
    pd_cfg = (
        ((config.get("generation") or {}).get("inverse") or {})
        .get("placeholder_distribution") or {}
    )
    count_max = pd_cfg.get("count_max", 5)

    empirical = task.profile_error_distribution(real_data, count_max=count_max, config=config)
    if empirical:
        return empirical

    keys = list(task.get_error_descriptions().keys())
    if not keys:
        raise ValueError(
            "Inverse mode requires task.get_error_descriptions() to be non-empty."
        )
    type_dist = {k: 1 / len(keys) for k in keys}
    count_dist = _poisson_pmf(
        mean=pd_cfg.get("count_mean", 1.5),
        n_min=1,
        n_max=count_max,
    )
    return {"type_dist": type_dist, "count_dist": count_dist}


# ── Aggregation ──────────────────────────────────────────────

def _mean_std(values: list[float]) -> dict:
    """Mean ± sample std (ddof=1) across runs. Std is 0.0 for a single run
    rather than NaN. Sample std is the right estimator when treating the runs
    as a sample of the model's behaviour on unseen data."""
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"mean": round(float(np.mean(values)), 4), "std": round(std, 4)}


def aggregate(all_run_scores: list[dict]) -> dict:
    """
    Compute mean ± std across N runs.
    High std reveals model instability on unseen data (see Paulreich 2025, p.8).

    Robust to heterogeneous runs: a model or evaluator that is missing from
    some runs is aggregated over only the runs where it is present, instead of
    raising KeyError off run 0.
    """
    model_names = {m for run in all_run_scores for m in run}
    final = {}
    for model_name in model_names:
        final[model_name] = {}
        evaluators = {
            ev for run in all_run_scores for ev in run.get(model_name, {})
        }
        for evaluator in evaluators:
            present = [
                run[model_name][evaluator]
                for run in all_run_scores
                if model_name in run and evaluator in run[model_name]
            ]
            if isinstance(present[0], dict):
                subkeys = {sub for raw in present for sub in raw}
                final[model_name][evaluator] = {
                    sub: _mean_std([raw[sub] for raw in present if sub in raw])
                    for sub in subkeys
                }
            else:
                final[model_name][evaluator] = _mean_std(present)
    return final


# ── Synthetic data archiving ─────────────────────────────────

def save_synthetic_data(synthetic: list[dict], config: dict, task_name: str,
                        session_id: str, run_idx: int) -> str:
    """Archive one run's synthetic data under data/generated/<task>/ instead of discarding it."""
    base_dir = (config.get("output") or {}).get(
        "generated_data_dir", "framework/data/generated"
    )
    out_dir = os.path.join(base_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{session_id}_run{run_idx + 1}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)
    return path


# ── Generation dispatch ───────────────────────────────────────

def _run_generation(generator, task, config, real_data, error_dist, judge_call):
    """Dispatch to forward or inverse generation based on generation.mode.
    Both return the same {"original", "corrupted", "error_type"} contract."""
    gen_cfg = config["generation"]
    mode = gen_cfg.get("mode", "forward")
    sample_size = gen_cfg["sample_size"]

    if mode == "inverse":
        inverse_cfg = gen_cfg.get("inverse") or {}
        source_field = inverse_cfg.get("source_field", "correct")
        if real_data and not any(item.get(source_field) for item in real_data):
            raise ValueError(
                f"Inverse mode: source_field '{source_field}' is missing or empty on "
                f"all {len(real_data)} real samples. Set generation.inverse.source_field "
                f"to a field the task produces (spam: 'incorrect', gec: 'correct')."
            )
        return generator.generate_inverse(
            real_samples=real_data,
            inverse_prompt=task.get_inverse_prompt(),
            error_descriptions=task.get_error_descriptions(),
            type_dist=error_dist["type_dist"],
            count_dist=error_dist["count_dist"],
            sample_size=sample_size,
            source_field=source_field,
            judge_prompt=task.get_inverse_judge_prompt() if judge_call else None,
            judge_call=judge_call,
        )

    return generator.generate(
        real_samples=real_data,
        error_types=task.get_error_types(),
        prompt_instruction=task.get_prompt_instruction(),
        sample_size=sample_size,
        judge_prompt=task.get_judge_prompt() if judge_call else None,
        judge_call=judge_call,
        request_delay=gen_cfg.get("request_delay", 0.0),
    )


# ── Main pipeline ─────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """
    Run the full GET pipeline (Generate → Evaluate → Trash) N times,
    then aggregate results as mean ± std across runs.
    "Trash" means the synthetic data is never reused for evaluation —
    each run is archived under data/generated/ for inspection.
    """
    task          = load_task(config["task"]["name"])
    real_data     = load_real_data(config, task)
    generator     = load_generator(config["generation"])
    judge_call    = _build_judge_call(config, generator)
    evaluator_fns = task.get_evaluator_fns()

    mode = config["generation"].get("mode", "forward")
    error_dist = (
        load_error_distribution(config, real_data, task)
        if mode == "inverse" else None
    )

    all_run_scores = []
    num_runs   = config["generation"]["num_runs"]
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for run_idx in range(num_runs):
        print(f"\n{'='*50}")
        print(f"RUN {run_idx + 1} / {num_runs}")
        print(f"{'='*50}")

        # ── GENERATE ──────────────────────────────────────
        synthetic = _run_generation(
            generator, task, config, real_data, error_dist, judge_call
        )
        # ── EVALUATE ──────────────────────────────────────
        eval_samples = task.get_eval_samples(synthetic)
        texts = [s["text"] for s in eval_samples]

        run_scores = {}
        for model_config in config["task_models"]:
            model       = task.get_model(model_config)
            predictions = model.predict(texts)
            results = [
                {**s, "prediction": pred}
                for s, pred in zip(eval_samples, predictions)
            ]
            run_scores[model_config["name"]] = {
                name: evaluator_fns[name](results) for name in task.get_evaluators()
            }
            for name, score in run_scores[model_config["name"]].items():
                print(f"  {model_config['name']}  {name}: {score}")

        all_run_scores.append(run_scores)

        # ── TRASH (archive, never reuse) ──────────────────
        saved_path = save_synthetic_data(
            synthetic, config, task.get_task_name(), session_id, run_idx
        )
        print(f"\nSynthetic data archived to {saved_path}")

    # ── AGGREGATE ─────────────────────────────────────────
    final = aggregate(all_run_scores)

    results_path = (config.get("output") or {}).get("results_path", "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return final
