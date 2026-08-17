import json
import os
import random
import sys
from datetime import datetime

import numpy as np
from framework.data_loading import iter_local_rows, resolve_dataset_config
from framework.generators.factory import load_generator
from framework.tasks.base_task import BaseTask


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

def load_task(task_name: str, task_config: dict | None = None) -> BaseTask:
    """Instantiate the task by name.

    `task_config` is the config's whole `task:` block. No task consumes it yet —
    `variant` is read straight from the config by resolve_output_paths/_build_meta —
    but the parameter is accepted so variant-aware task construction can be wired
    up without touching every call site."""
    if task_name == "gec":
        from framework.tasks.gec.task import GECTask
        return GECTask()
    elif task_name == "spam":
        from framework.tasks.spam.task import SpamTask
        return SpamTask()
    elif task_name == "taxonomy":
        from framework.tasks.taxonomy.task import TaxonomyTask
        return TaxonomyTask()
    raise ValueError(
        f"Unknown task: '{task_name}'. "
        f"Register it in pipeline.load_task() and add configs/{task_name}/{task_name}.json."
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


# ── Error distribution ───────────────────────────────────────

def load_error_distribution(config: dict, real_data: list[dict], task) -> dict:
    """Return {"type_dist": {key: prob}, "count_dist": {n: prob}} derived
    empirically from the real benchmark via task.profile_error_distribution.

    Raises RuntimeError when the data is insufficient — generation never runs
    on a distribution the benchmark doesn't exhibit."""
    empirical = task.profile_error_distribution(real_data, config=config)
    if not empirical:
        raise RuntimeError(
            f"Could not derive an empirical error distribution for task "
            f"'{task.get_task_name()}': fewer than 5 usable samples. "
            "Increase generation.sample_size (GEC), check that "
            "dataset.reference_size is not set too low (spam), or check "
            "that the dataset yields valid pairs."
        )
    return empirical


DEFAULT_PROFILE_DIR = "framework/data/profiles"


def _resolve_profile_path(config: dict, task) -> str:
    """The profile path seedless generation actually uses: generation.profile_path
    if the config sets one, else the default framework/data/profiles/<task>_profile.json.

    Single source of truth for that default, shared by `_load_generation_profile`
    (which loads the file) and `_build_meta` (which records the path as
    provenance). Profiles are gitignored, so `_build_meta`'s copy is the only
    surviving record of what generated a seedless benchmark — it must resolve
    the SAME default `_load_generation_profile` used, not just echo a possibly-
    unset config key."""
    gen = config.get("generation") or {}
    return gen.get("profile_path") or os.path.join(
        DEFAULT_PROFILE_DIR, f"{task.get_task_name()}_profile.json"
    )


def _load_generation_profile(config: dict, task) -> dict | None:
    """Load the benchmark profile that drives seedless generation.

    Returns None when generation.seedless is falsy. Runs before the generation
    loop so a missing or un-topic-profiled profile fails before any API spend."""
    gen = config.get("generation") or {}
    if task.get_generation_strategy() == "structured":
        if gen.get("seedless") is False:
            return None
        path = _resolve_profile_path(config, task)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if not gen.get("seedless"):
        return None
    from framework.profiling.spec_sampler import load_profile

    path = _resolve_profile_path(config, task)
    topics_key = (
        "topics_per_label"
        if task.get_generation_strategy() == "class_conditional"
        else "topics"
    )
    return load_profile(path, topics_key=topics_key)


def _should_load_error_distribution(strategy: str, mode: str | None, seedless: bool) -> bool:
    """Whether this strategy needs an empirical error distribution."""
    if strategy == "structured":
        return False
    return strategy == "class_conditional" or mode == "inverse" or seedless


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
    """All artifact paths for one run session, under output.base_dir/<task>/<session>/.
    If task.variant is set, the folder is named <task>_<variant> instead of <task>."""
    base = (config.get("output") or {}).get("base_dir", "framework/data/runs")
    variant = (config.get("task") or {}).get("variant")
    folder = f"{task_name}_{variant}" if variant else task_name
    session_dir = os.path.join(base, folder, session)
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

def _build_meta(config: dict, task, runs_completed: int,
                effective_samples_per_run: list[int], real_baseline: bool) -> dict:
    """Provenance block written next to the scores: what produced this file.

    `partial` is True while runs are still outstanding — results files are
    (re)written after every run so an interrupted session keeps the runs it
    already paid for.

    `mode` is meaningful for every strategy now: `corruption` (GEC) defaults
    to "forward" when omitted, `class_conditional` (spam) defaults to
    "inverse" — matching `_run_generation`'s own per-strategy default, so
    this echoes the mode that actually ran rather than a stray config value.

    `seedless` mirrors generation.seedless (False when the key is absent).
    `profile_path` is the resolved path of the profile that actually drove
    generation when seedless is true — generation.profile_path if the config
    set one, else the same default `_load_generation_profile` resolves
    internally (see `_resolve_profile_path`, shared by both) — and None when
    seedless is false. Profiles are gitignored, so this is the only record of
    what generated a seedless benchmark; it must NOT be left None in the
    common case where seedless is true and profile_path is left unset (both
    shipped configs ship it commented out)."""
    gen = config["generation"]
    ds = resolve_dataset_config(config.get("dataset") or {})
    judge = config.get("judge") or {}
    judge_active = bool(judge) and judge.get("enabled", True) is not False
    num_runs = gen["num_runs"]
    strategy = task.get_generation_strategy()
    if strategy == "structured":
        mode = None
    else:
        mode = gen.get("mode", "inverse" if strategy == "class_conditional" else "forward")
    seedless = True if strategy == "structured" else bool(gen.get("seedless"))
    if ds["source"] == "local":
        dataset_meta = {"source": "local", "path": ds["path"],
                        "format": ds["format"] or None,
                        "sample_size": config["generation"].get("sample_size")}
    else:
        dataset_meta = {"source": "huggingface", "name": ds["name"],
                        "split": ds["split"], "sample_size": config["generation"].get("sample_size")}
    task_cfg = config.get("task") or {}
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": task_cfg.get("name", config["task"]["name"]),
        "variant": task_cfg.get("variant") or None,
        "strategy": strategy,
        "mode": mode,
        "seedless": seedless,
        "profile_path": _resolve_profile_path(config, task) if seedless else None,
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

def _dataset_display_name(config: dict) -> str:
    """Human-readable name of the configured dataset, for error messages that
    need to point at what to fix: the local file path, or the HuggingFace
    dataset name."""
    ds = resolve_dataset_config(config.get("dataset") or {})
    return ds["path"] if ds["source"] == "local" else ds["name"]


def _run_generation(generator, task, config, real_data, error_dist, judge_call, class_prob,
                    profile=None):
    """Dispatch on the task's generation strategy. Corruption → forward/inverse,
    including the seedless forward/inverse cells for GEC (Task 5). class_conditional
    (spam) now dispatches its own four cells on (mode, seedless) — Task 8:
    inverse+seeded (today's unchanged production behavior) injects/paraphrases
    real HAM seeds via seed_policy="cross_class"; inverse+seedless does the same
    but over carriers synthesized from the profile in place of real seeds;
    forward+seeded imitates within a class via seed_policy="same_class" over
    `task.get_seed_pool(..., "forward")`; forward+seedless drops real seeds
    entirely via seed_policy="none" over per-label profile specs.

    `profile` is the pre-loaded seedless generation profile (None when
    generation.seedless is falsy). When `generation.seedless` is true, per-sample
    content specs are drawn from it and no real benchmark text reaches the
    generation prompt: forward mode calls `generate_seedless_pairs` directly;
    inverse mode synthesizes carriers via `generate_carriers` and feeds them into
    the unchanged `generate_inverse` in place of real seeds."""
    gen_cfg = config["generation"]
    sample_size = gen_cfg["sample_size"]
    strategy = task.get_generation_strategy()

    if strategy == "structured":
        if gen_cfg.get("mode") not in (None, "structured", "none", "n/a"):
            raise RuntimeError(
                f"{task.get_task_name()} uses structured generation; generation.mode "
                "is not applicable and must be omitted."
            )
        if gen_cfg.get("seedless") is False:
            raise RuntimeError(
                f"{task.get_task_name()} structured generation is profile-driven; "
                "seeded generation is not supported."
            )
        if profile is None:
            raise RuntimeError(
                f"{task.get_task_name()} structured generation requires a profile."
            )
        rng = random.Random()
        synthetic = []
        max_parse_attempts = max(1, gen_cfg.get("max_parse_attempts", 3))
        feedback_cfg = task.get_feedback_config(gen_cfg) if hasattr(task, "get_feedback_config") else {}
        feedback_enabled = bool(feedback_cfg.get("enabled", False))
        max_feedback_rounds = (
            max(0, int(feedback_cfg.get("max_rounds", 0)))
            if feedback_enabled else 0
        )

        for _ in range(sample_size):
            selected = None
            metadata = {
                "feedback_enabled": feedback_enabled,
                "max_feedback_rounds": max_feedback_rounds,
                "rounds": [],
                "early_stopped": False,
                "final_round_selected": None,
                "final_taxonomy_feedback_informed": False,
            }
            feedback = None
            for round_idx in range(max_feedback_rounds + 1):
                parsed = None
                attempts = 0
                while parsed is None and attempts < max_parse_attempts:
                    attempts += 1
                    prompt = task.build_structured_generation_prompt(
                        profile, rng=rng, feedback=feedback
                    )
                    raw = generator.call_api(prompt)
                    parsed = task.parse_structured_generation(raw)
                    if parsed is None:
                        print("[SKIP] structured generation returned invalid JSON/artifact.")

                if parsed is None:
                    metadata["rounds"].append({
                        "round": round_idx,
                        "feedback_informed": feedback is not None,
                        "parse_attempts": attempts,
                        "valid": False,
                    })
                    if selected is not None:
                        metadata["failed_feedback_round_preserved_previous"] = True
                        break
                    break

                selected = parsed
                if not feedback_enabled:
                    metadata["final_round_selected"] = round_idx
                    break
                round_info = task.build_structural_feedback(
                    profile, selected, generation_config=gen_cfg
                )
                feedback_result = round_info["feedback"]
                metadata["rounds"].append({
                    "round": round_idx,
                    "feedback_informed": feedback is not None,
                    "parse_attempts": attempts,
                    "valid": True,
                    "within_tolerance": feedback_result["within_tolerance"],
                    "feedback": feedback_result,
                    "comparison": round_info["comparison"],
                    "synthetic_profile": round_info["synthetic_profile"],
                })
                metadata["final_round_selected"] = round_idx
                metadata["final_taxonomy_feedback_informed"] = feedback is not None
                if feedback_result["within_tolerance"]:
                    metadata["early_stopped"] = True
                    break
                if round_idx >= max_feedback_rounds:
                    break
                feedback = feedback_result

            if selected is not None:
                selected["generation_feedback"] = metadata
                synthetic.append(selected)
        if len(synthetic) < sample_size:
            print(
                f"[WARN] structured generation produced {len(synthetic)} valid "
                f"taxonomies for {sample_size} requested.",
                file=sys.stderr,
            )
    elif strategy == "class_conditional":
        # No config sets "mode" explicitly today (spam.json's config comment
        # says so) — the default MUST resolve to "inverse" so that omitting
        # the key keeps reproducing today's production behavior unchanged.
        mode = gen_cfg.get("mode", "inverse")
        seedless = bool(gen_cfg.get("seedless"))
        # Required regardless of seed_policy — generate_class_conditional's
        # signature has no defaults for these, even though same_class/none
        # policies use forward_prompt/seedless_prompts instead of inject_prompt.
        common_kwargs = dict(
            class_prob=class_prob,
            type_dist=error_dist["type_dist"],
            count_dist=error_dist["count_dist"],
            error_descriptions=task.get_error_descriptions(),
            inject_prompt=task.get_inverse_prompt(),
            negative_prompt=task.get_ham_generation_prompt(),
            positive_label="SPAM",
            negative_label="HAM",
            sample_size=sample_size,
            judge_prompt=task.get_inverse_judge_prompt() if judge_call else None,
            judge_call=judge_call,
            request_delay=gen_cfg.get("request_delay", 0.0),
        )

        if mode == "inverse":
            if seedless:
                from framework.profiling.spec_sampler import render_spec, sample_content_spec
                carrier_prompt = task.get_carrier_prompt()
                if not carrier_prompt:
                    raise RuntimeError(
                        f"{task.get_task_name()} does not support mode=inverse with "
                        f"seedless=true (no carrier_prompt)."
                    )
                rng = random.Random()
                specs = [
                    render_spec(sample_content_spec(profile, rng, label="HAM"))
                    for _ in range(sample_size)
                ]
                carriers = generator.generate_carriers(
                    specs, carrier_prompt, "Message",
                    request_delay=gen_cfg.get("request_delay", 0.0),
                )
                real_seeds = [{"text": text} for text in carriers]
                seed_field = "text"
            else:
                # Post-parse_row contract: the seed text always lives in "incorrect".
                real_seeds = real_data
                seed_field = "incorrect"
            synthetic = generator.generate_class_conditional(
                real_seeds=real_seeds,
                seed_field=seed_field,
                seed_policy="cross_class",
                **common_kwargs,
            )
        elif seedless:
            from framework.profiling.spec_sampler import render_spec, sample_content_spec
            seedless_prompts = task.get_seedless_class_prompts()
            if not seedless_prompts:
                raise RuntimeError(
                    f"{task.get_task_name()} does not support mode=forward with "
                    f"seedless=true (no seedless_class_prompts)."
                )
            rng = random.Random()
            specs_by_label = {
                label: [
                    render_spec(sample_content_spec(profile, rng, label=label))
                    for _ in range(sample_size)
                ]
                for label in ("SPAM", "HAM")
            }
            synthetic = generator.generate_class_conditional(
                seed_policy="none",
                specs_by_label=specs_by_label,
                seedless_prompts=seedless_prompts,
                **common_kwargs,
            )
        else:
            forward_prompt = task.get_forward_prompt()
            if not forward_prompt:
                raise RuntimeError(
                    f"{task.get_task_name()} does not support mode=forward with "
                    f"seedless=false (no forward_prompt)."
                )
            real_seeds = task.get_seed_pool(config, real_data, "forward")
            # seed_policy="same_class" needs seeds of BOTH classes (it draws
            # from the subset matching the label rng picked). Check that BEFORE
            # any API call: the in-loop guard in generate_class_conditional
            # catches this too, but only after rng happens to draw the missing
            # class — anywhere from 1 to `sample_size` paid calls in, discarding
            # everything generated so far.
            label_field = "label"
            present_labels = {row.get(label_field) for row in real_seeds}
            for label in (common_kwargs["positive_label"], common_kwargs["negative_label"]):
                if label not in present_labels:
                    raise RuntimeError(
                        f"seed_policy='same_class' needs seeds labeled {label!r}, but "
                        f"the reference rows from dataset "
                        f"'{_dataset_display_name(config)}' carry no {label!r} class "
                        f"— check the dataset and {task.get_task_name()}'s "
                        f"get_seed_pool()."
                    )
            synthetic = generator.generate_class_conditional(
                real_seeds=real_seeds,
                seed_field="text",
                label_field=label_field,
                seed_policy="same_class",
                forward_prompt=forward_prompt,
                **common_kwargs,
            )
    else:
        mode = gen_cfg.get("mode", "forward")
        seedless = bool(gen_cfg.get("seedless"))
        if seedless:
            from framework.profiling.spec_sampler import render_spec, sample_content_spec
            rng = random.Random()
            side = task.get_profile_side(mode)
            specs = [
                render_spec(sample_content_spec(profile, rng, side=side))
                for _ in range(sample_size)
            ]

        if mode == "inverse":
            if seedless:
                carrier_prompt = task.get_carrier_prompt()
                if not carrier_prompt:
                    raise RuntimeError(
                        f"{task.get_task_name()} does not support mode=inverse with "
                        f"seedless=true (no carrier_prompt)."
                    )
                carriers = generator.generate_carriers(
                    specs, carrier_prompt, "Sentence",
                    request_delay=gen_cfg.get("request_delay", 0.0),
                )
                real_data = [{"correct": text} for text in carriers]
            # Post-parse_row contract: every corruption task normalizes rows to
            # {"incorrect", "correct"}; inverse mode corrupts the clean side.
            source_field = "correct"
            if real_data and not any(item.get(source_field) for item in real_data):
                raise ValueError(
                    f"Inverse mode corrupts the clean '{source_field}' field, but it "
                    f"is missing or empty on all {len(real_data)} real samples — "
                    "check the dataset and task.parse_row()."
                )
            synthetic = generator.generate_inverse(
                real_samples=real_data, inverse_prompt=task.get_inverse_prompt(),
                error_descriptions=task.get_error_descriptions(),
                type_dist=error_dist["type_dist"], count_dist=error_dist["count_dist"],
                sample_size=sample_size, source_field=source_field,
                judge_prompt=task.get_inverse_judge_prompt() if judge_call else None,
                judge_call=judge_call, request_delay=gen_cfg.get("request_delay", 0.0),
            )
        elif seedless:
            prompt = task.get_seedless_forward_prompt()
            if not prompt:
                raise RuntimeError(
                    f"{task.get_task_name()} does not support mode=forward with "
                    f"seedless=true (no seedless_forward_prompt)."
                )
            synthetic = generator.generate_seedless_pairs(
                specs, prompt, task.get_error_descriptions(),
                error_dist["type_dist"], error_dist["count_dist"],
                judge_prompt=task.get_judge_prompt() if judge_call else None,
                judge_call=judge_call, rng=rng,
                request_delay=gen_cfg.get("request_delay", 0.0),
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


def _nest_results(generated_agg: dict, real_scores: dict,
                  all_run_scores: list[dict] | None = None) -> dict:
    """Group each model's scores as {generated, real?, runs?}.

    `runs` lists the model's score dict for each completed run. It is additive —
    the printer and compare_models read only generated/real — and it is what the
    run-variance figure plots."""
    final = {}
    for model in set(generated_agg) | set(real_scores):
        final[model] = {}
        if model in generated_agg:
            final[model]["generated"] = generated_agg[model]
        if model in real_scores:
            final[model]["real"] = real_scores[model]
        runs = [run[model] for run in (all_run_scores or []) if model in run]
        if runs:
            final[model]["runs"] = runs
    return final


def _write_profile_artifacts(task, real_reference, all_generated, paths) -> None:
    """Persist the real sample + a {real, generated, fidelity} profile when the
    task supports profiling. No-op for tasks whose profile_dataset returns None."""
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


def _render_plots(config: dict, paths: dict) -> None:
    """Render the session's figures. Runs AFTER results/profile are on disk and is
    fail-soft: plotting must never cost a run that already succeeded."""
    if not (config.get("output") or {}).get("plots", True):
        return
    try:
        from framework.plotting.session import render_session
        render_session(paths["session_dir"], paths["plots_dir"])
    except Exception as e:
        print(f"[WARN] plotting failed (results are unaffected): {e}", file=sys.stderr)


# ── Main pipeline ─────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """Run the GET pipeline N times, evaluate the generated benchmark (mean±std)
    and — by default — the same models on the real benchmark, profile real-vs-
    generated fidelity, and write all artifacts under one per-session directory."""
    task          = load_task(config["task"]["name"], config.get("task"))
    real_data     = load_real_data(config, task)
    generator     = load_generator(config["generation"])
    judge_call    = _build_judge_call(config, generator)
    evaluator_fns = task.get_evaluator_fns()

    strategy = task.get_generation_strategy()
    mode = None if strategy == "structured" else config["generation"].get("mode", "forward")
    seedless = (
        True if strategy == "structured"
        else bool(config["generation"].get("seedless"))
    )
    error_dist = (
        load_error_distribution(config, real_data, task)
        if _should_load_error_distribution(strategy, mode, seedless) else None
    )
    profile = _load_generation_profile(config, task)

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

    # The real baseline is deterministic (fixed reference sample, fixed task
    # models): compute it once and reuse it in every per-run checkpoint write.
    real_scores = (
        _evaluate_real_baseline(task, config, real_reference, evaluator_fns)
        if real_baseline else {}
    )

    for run_idx in range(num_runs):
        print(f"\n{'='*50}\nRUN {run_idx + 1} / {num_runs}\n{'='*50}")
        synthetic = _run_generation(generator, task, config, real_data, error_dist,
                                    judge_call, class_prob, profile=profile)
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
        final = _nest_results(generated_agg, real_scores, all_run_scores)
        meta = _build_meta(config, task, runs_completed=run_idx + 1,
                           effective_samples_per_run=effective_samples,
                           real_baseline=bool(real_scores))
        _write_results(final, paths["results"], meta)
        if run_idx + 1 < num_runs:
            print(f"Partial results (run {run_idx + 1}/{num_runs}) saved to {paths['results']}")

    _write_profile_artifacts(task, real_reference, all_generated, paths)
    _render_plots(config, paths)
    print(f"\nResults saved to {paths['results']}")
    return final
