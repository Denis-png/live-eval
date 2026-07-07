import json
import math
import os
import sys
from datetime import datetime

import numpy as np
from framework.data_loading import iter_local_rows, resolve_dataset_config
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
    Return a callable(prompt: str) -> str for the LLM-as-judge step,
    or None when judging is skipped.

    Judging is opt-in (matches the config.yaml comment):
      1. no judge block, or judge.enabled == false → None (judging skipped)
      2. judge block with provider+model → load a separate judge generator
      3. judge block enabled but missing provider/model → warn and fall back
         to the main generator (the user explicitly asked for judging)
    """
    judge_cfg = config.get("judge")
    if not judge_cfg:
        return None
    if judge_cfg.get("enabled", True) is False:
        return None
    if not judge_cfg.get("provider") or not judge_cfg.get("model"):
        print(
            "[WARN] judge block missing provider/model — falling back to main generator.",
            file=sys.stderr,
        )
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
    Load real samples from a HuggingFace dataset or a local file
    (m2 / csv / tsv — see data_loading.iter_local_rows).
    Supports streaming (dataset.huggingface.streaming: true) for large datasets.
    Field parsing and row filtering is delegated to task.parse_row().
    """
    ds_config = resolve_dataset_config(config["dataset"])
    sample_size = config["generation"]["sample_size"]

    if ds_config["source"] == "local":
        print(f"Loading local dataset: {ds_config['path']} ...")
        rows = iter_local_rows(ds_config["path"], ds_config["format"])
    else:
        from datasets import load_dataset  # lazy: keeps pipeline importable without HF deps

        print(f"Loading dataset: {ds_config['name']} ...")
        hf_token = (
            ds_config.get("hf_token")
            or (config.get("api_keys") or {}).get("huggingface")
            or os.getenv("HF_TOKEN")
        )
        rows = load_dataset(
            ds_config["name"],
            split=ds_config["split"],
            streaming=ds_config["streaming"],
            token=hf_token or None,
        )

    samples = []
    for row in rows:
        parsed = task.parse_row(row)
        if parsed is not None:
            samples.append(parsed)
        if len(samples) >= sample_size:
            break

    print(f"Loaded {len(samples)} real samples.")
    if len(samples) < sample_size:
        # sample_size counts USABLE samples (task.parse_row filters rows, e.g.
        # spam keeps HAM only) — the source ran out before filling the pool.
        print(
            f"[WARN] generation.sample_size asks for {sample_size} usable samples "
            f"but the source only yielded {len(samples)} — the run proceeds on "
            f"the smaller pool.",
            file=sys.stderr,
        )
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


# ── Output paths ──────────────────────────────────────────────

def resolve_output_paths(config: dict, task_name: str, session: str) -> dict:
    """All artifact paths for one run session, under output.base_dir/<task>/<session>/."""
    base = (config.get("output") or {}).get("base_dir", "framework/data/runs")
    session_dir = os.path.join(base, task_name, session)
    return {
        "session_dir": session_dir,
        "generated_dir": os.path.join(session_dir, "generated"),
        "results": os.path.join(session_dir, "results.json"),
        "real_sample": os.path.join(session_dir, "real_sample.json"),
        "profile": os.path.join(session_dir, "profile.json"),
        "plots_dir": os.path.join(session_dir, "plots"),
    }


# ── Synthetic data archiving ─────────────────────────────────

def save_synthetic_data(synthetic: list[dict], generated_dir: str, run_idx: int) -> str:
    """Archive one run's synthetic data under <session>/generated/run_<N>.json."""
    os.makedirs(generated_dir, exist_ok=True)
    path = os.path.join(generated_dir, f"run_{run_idx + 1}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)
    return path


# ── Results writing (with provenance) ────────────────────────

def _build_meta(config: dict, runs_completed: int,
                effective_samples_per_run: list[int], real_baseline: bool) -> dict:
    """Provenance block written next to the scores: what produced this file.

    `partial` is True while runs are still outstanding — results files are
    (re)written after every run so an interrupted session keeps the runs it
    already paid for."""
    gen = config["generation"]
    ds = resolve_dataset_config(config.get("dataset") or {})
    judge = config.get("judge") or {}
    judge_active = bool(judge) and judge.get("enabled", True) is not False
    num_runs = gen["num_runs"]
    if ds["source"] == "local":
        dataset_meta = {"source": "local", "path": ds["path"],
                        "format": ds["format"] or None,
                        "sample_size": config["generation"].get("sample_size")}
    else:
        dataset_meta = {"source": "huggingface", "name": ds["name"],
                        "split": ds["split"], "sample_size": config["generation"].get("sample_size")}
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": config["task"]["name"],
        "mode": gen.get("mode", "forward"),
        "provider": gen["provider"],
        "model": gen["model"],
        "num_runs": num_runs,
        "runs_completed": runs_completed,
        "partial": runs_completed < num_runs,
        "dataset": dataset_meta,
        "effective_samples_per_run": effective_samples_per_run,
        "judge": (
            {"provider": judge.get("provider"), "model": judge.get("model")}
            if judge_active else None
        ),
        "real_baseline": real_baseline,
        "class_balance": gen.get("class_balance", "empirical"),
    }


def _write_results(final: dict, results_path: str, meta: dict) -> str:
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": final}, f, indent=2)
    return results_path


# ── Generation dispatch ───────────────────────────────────────

def _run_generation(generator, task, config, real_data, error_dist, judge_call, class_prob):
    """Dispatch on the task's generation strategy. Corruption → forward/inverse
    (unchanged). Class-conditional → labeled SPAM/HAM records."""
    gen_cfg = config["generation"]
    sample_size = gen_cfg["sample_size"]
    strategy = task.get_generation_strategy()

    if strategy == "class_conditional":
        seed_field = gen_cfg.get("seed_field", "incorrect")
        synthetic = generator.generate_class_conditional(
            real_seeds=real_data,
            seed_field=seed_field,
            class_prob=class_prob,
            type_dist=error_dist["type_dist"],
            count_dist=error_dist["count_dist"],
            error_descriptions=task.get_error_descriptions(),
            inject_prompt=task.get_inverse_prompt(),
            ham_prompt=task.get_ham_generation_prompt(),
            positive_label="SPAM",
            negative_label="HAM",
            sample_size=sample_size,
            judge_prompt=task.get_inverse_judge_prompt() if judge_call else None,
            judge_call=judge_call,
            request_delay=gen_cfg.get("request_delay", 0.0),
        )
    else:
        mode = gen_cfg.get("mode", "forward")
        if mode == "inverse":
            inverse_cfg = gen_cfg.get("inverse") or {}
            source_field = inverse_cfg.get("source_field", "correct")
            if real_data and not any(item.get(source_field) for item in real_data):
                raise ValueError(
                    f"Inverse mode: source_field '{source_field}' is missing or empty on "
                    f"all {len(real_data)} real samples. Set generation.inverse.source_field "
                    f"to a field the task produces (spam: 'incorrect', gec: 'correct')."
                )
            synthetic = generator.generate_inverse(
                real_samples=real_data, inverse_prompt=task.get_inverse_prompt(),
                error_descriptions=task.get_error_descriptions(),
                type_dist=error_dist["type_dist"], count_dist=error_dist["count_dist"],
                sample_size=sample_size, source_field=source_field,
                judge_prompt=task.get_inverse_judge_prompt() if judge_call else None,
                judge_call=judge_call, request_delay=gen_cfg.get("request_delay", 0.0),
            )
        else:
            synthetic = generator.generate(
                real_samples=real_data, error_types=task.get_error_types(),
                prompt_instruction=task.get_prompt_instruction(), sample_size=sample_size,
                judge_prompt=task.get_judge_prompt() if judge_call else None,
                judge_call=judge_call, request_delay=gen_cfg.get("request_delay", 0.0),
            )

    if not synthetic:
        raise RuntimeError(
            f"Generation produced 0 usable samples out of {sample_size} requested "
            f"({strategy}). Scoring an empty set would report misleading 0.0 metrics. "
            f"Check the [SKIP]/failed lines above — typical causes: bad API key, wrong "
            f"model name, model refusals, or unparseable output."
        )
    return synthetic


# ── Post-generation helpers (class balance, real baseline, nesting, profiling) ──

def _resolve_class_prob(config: dict, real_reference) -> float:
    """P(positive class) for class-conditional generation. `empirical` → the real
    reference's positive fraction; a float → used directly."""
    cb = (config.get("generation") or {}).get("class_balance", "empirical")
    if isinstance(cb, (int, float)):
        return float(cb)
    if real_reference:
        pos = sum(1 for r in real_reference if r.get("label") == "SPAM")
        return pos / len(real_reference)
    return 0.5


def _evaluate_real_baseline(task, config, real_reference, evaluator_fns) -> dict:
    """Evaluate task_models once on the real benchmark (deterministic → no runs)."""
    if not real_reference:
        print("[real baseline] skipped — task has no real reference.")
        return {}
    texts = [s["text"] for s in real_reference]
    out = {}
    for model_config in config["task_models"]:
        model = task.get_model(model_config)
        predictions = model.predict(texts)
        results = [{**s, "prediction": p} for s, p in zip(real_reference, predictions)]
        out[model_config["name"]] = {
            name: evaluator_fns[name](results) for name in task.get_evaluators()
        }
    return out


def _nest_results(generated_agg: dict, real_scores: dict) -> dict:
    """Group each model's scores as {generated, real?}."""
    final = {}
    for model in set(generated_agg) | set(real_scores):
        final[model] = {}
        if model in generated_agg:
            final[model]["generated"] = generated_agg[model]
        if model in real_scores:
            final[model]["real"] = real_scores[model]
    return final


def _write_profile_artifacts(task, real_reference, all_generated, paths) -> None:
    """Persist the real sample + a {real, generated, fidelity} profile when the
    task supports profiling. No-op for tasks that don't (e.g. GEC)."""
    if real_reference is None:
        return
    with open(paths["real_sample"], "w", encoding="utf-8") as f:
        json.dump(real_reference, f, indent=2, ensure_ascii=False)
    real_profile = task.profile_dataset(real_reference)
    if real_profile is None:
        return
    generated_profile = task.profile_dataset(all_generated)
    fidelity = task.compare_profiles(real_profile, generated_profile)
    with open(paths["profile"], "w", encoding="utf-8") as f:
        json.dump({"real": real_profile, "generated": generated_profile,
                   "fidelity": fidelity}, f, indent=2, ensure_ascii=False)
    print(f"Fidelity profile saved to {paths['profile']}")


# ── Main pipeline ─────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """Run the GET pipeline N times, evaluate the generated benchmark (mean±std)
    and — by default — the same models on the real benchmark, profile real-vs-
    generated fidelity, and write all artifacts under one per-session directory."""
    task          = load_task(config["task"]["name"])
    real_data     = load_real_data(config, task)
    generator     = load_generator(config["generation"])
    judge_call    = _build_judge_call(config, generator)
    evaluator_fns = task.get_evaluator_fns()

    strategy = task.get_generation_strategy()
    mode = config["generation"].get("mode", "forward")
    error_dist = (
        load_error_distribution(config, real_data, task)
        if (strategy == "class_conditional" or mode == "inverse") else None
    )

    # Real reference feeds class balance, the real baseline, and profiling.
    real_reference = task.get_real_eval_samples(config, real_data)
    class_prob = _resolve_class_prob(config, real_reference)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.get("output", {}).get("session_id"):
        session_id = config["output"]["session_id"]
    paths = resolve_output_paths(config, task.get_task_name(), session_id)
    os.makedirs(paths["generated_dir"], exist_ok=True)
    os.makedirs(paths["plots_dir"], exist_ok=True)

    all_run_scores, effective_samples, all_generated = [], [], []
    num_runs = config["generation"]["num_runs"]
    real_baseline = (config.get("evaluation") or {}).get("real_baseline", True)

    for run_idx in range(num_runs):
        print(f"\n{'='*50}\nRUN {run_idx + 1} / {num_runs}\n{'='*50}")
        synthetic = _run_generation(generator, task, config, real_data, error_dist,
                                    judge_call, class_prob)
        all_generated.extend(synthetic)

        eval_samples = task.get_eval_samples(synthetic)
        texts = [s["text"] for s in eval_samples]
        run_scores = {}
        for model_config in config["task_models"]:
            model = task.get_model(model_config)
            predictions = model.predict(texts)
            results = [{**s, "prediction": p} for s, p in zip(eval_samples, predictions)]
            run_scores[model_config["name"]] = {
                name: evaluator_fns[name](results) for name in task.get_evaluators()
            }
            for name, score in run_scores[model_config["name"]].items():
                print(f"  {model_config['name']}  {name}: {score}")
        all_run_scores.append(run_scores)
        effective_samples.append(len(eval_samples))

        saved_path = save_synthetic_data(synthetic, paths["generated_dir"], run_idx)
        print(f"\nSynthetic data archived to {saved_path}")

        generated_agg = aggregate(all_run_scores)
        real_scores = _evaluate_real_baseline(task, config, real_reference,
                                              evaluator_fns) if real_baseline else {}
        final = _nest_results(generated_agg, real_scores)
        meta = _build_meta(config, runs_completed=run_idx + 1,
                           effective_samples_per_run=effective_samples,
                           real_baseline=bool(real_scores))
        _write_results(final, paths["results"], meta)
        if run_idx + 1 < num_runs:
            print(f"Partial results (run {run_idx + 1}/{num_runs}) saved to {paths['results']}")

    _write_profile_artifacts(task, real_reference, all_generated, paths)
    print(f"\nResults saved to {paths['results']}")
    return final
