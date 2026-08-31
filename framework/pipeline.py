import glob
import json
import os
import random
import re
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

def load_task(task_name: str) -> BaseTask:
    """Instantiate the task by name."""
    if task_name == "gec":
        from framework.tasks.gec.task import GECTask
        return GECTask()
    elif task_name == "spam":
        from framework.tasks.spam.task import SpamTask
        return SpamTask()
    elif task_name == "sentiment":
        from framework.tasks.sentiment.task import SentimentTask
        return SentimentTask()
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
            ds_config.get("subset"),
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
    return _apply_calibration(config, task, empirical)


# Provenance for _build_meta: results.json is the only surviving record of which
# calibration produced a benchmark, because artifacts are gitignored.
_LAST_CALIBRATION: dict | None = None


def _apply_calibration(config: dict, task, empirical: dict) -> dict:
    """Prefer a matching calibration artifact over the raw empirical target.

    Falls back to `empirical` when none exists, when the user opted out, or when
    the stored setpoint no longer matches the benchmark — a stale artifact must
    never be used silently.
    """
    global _LAST_CALIBRATION
    _LAST_CALIBRATION = None

    from framework.calibration.artifact import (
        load_calibration,
        resolve_calibration_path,
        targets_match,
    )

    strategy = task.get_generation_strategy()
    path = resolve_calibration_path(config, task, strategy)
    if not path:
        print(f"[NOTE] no calibration artifact for "
              f"{generation_cell_slug(config, strategy)}; generating from the raw "
              f"empirical distribution. Build one with: python -m "
              f"framework.calibrate --config <config.yaml>")
        return empirical

    # The whole read-validate-select sequence is guarded, not just the file
    # read: a JSON-valid but structurally corrupt artifact (a non-dict
    # "target", a non-object top-level payload, a non-numeric distribution
    # value) must warn and fall back like any other bad artifact, never crash
    # the run. AttributeError/TypeError/KeyError cover malformed shapes
    # (e.g. calling .get on a list/str, float() on a non-numeric value);
    # OSError/ValueError cover unreadable files and invalid JSON.
    try:
        payload = load_calibration(path)
        matched = targets_match(payload.get("target") or {}, empirical)
        calibrated = payload.get("calibrated") or {}
        type_dist = calibrated.get("type_dist")
        count_dist = calibrated.get("count_dist")
        usable = bool(type_dist) and bool(count_dist)
        result = {"type_dist": dict(type_dist), "count_dist": dict(count_dist)} if usable else None
        selected_round = payload.get("selected_round")
    except (OSError, ValueError, AttributeError, TypeError, KeyError) as e:
        print(f"[WARN] calibration {path!r} could not be read or is malformed "
              f"({e}); using the empirical distribution.", file=sys.stderr)
        return empirical

    if not matched:
        print(f"[WARN] calibration {path!r} was built against a different "
              "benchmark setpoint (dataset or sample size changed); using the "
              "empirical distribution instead.", file=sys.stderr)
        return empirical

    if not usable:
        print(f"[WARN] calibration {path!r} has no usable distributions; "
              "using the empirical distribution.", file=sys.stderr)
        return empirical

    _LAST_CALIBRATION = {"path": path, "selected_round": selected_round,
                         "class_prob": calibrated.get("class_prob")}
    print(f"Calibration: {path} (round {selected_round})")
    return result


DEFAULT_PROFILE_DIR = "framework/data/profiles"


def benchmark_slug(config: dict) -> str:
    """Short identifier for the benchmark a profile describes: a local file's
    stem, or the last path component of a HuggingFace dataset name."""
    ds = resolve_dataset_config(config.get("dataset") or {})
    if ds.get("source") == "local" and ds.get("path"):
        raw = os.path.splitext(os.path.basename(ds["path"]))[0]
    else:
        raw = (ds.get("name") or "dataset").split("/")[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(raw)).strip("_").lower()


def benchmark_profile_filename(config: dict, task_name: str, num_samples: int) -> str:
    """<benchmark>_<sample_size>_<task>_profile.json — the sample size is the
    number of rows actually profiled, so the name states what the file covers."""
    return f"{benchmark_slug(config)}_{num_samples}_{task_name}_profile.json"


def benchmark_profile_dir(task_name: str) -> str:
    """Profiles are grouped per task: framework/data/profiles/<task>/."""
    return os.path.join(DEFAULT_PROFILE_DIR, task_name)


def _resolve_benchmark_profile_path(config: dict, task) -> str:
    """The profile path seedless generation actually uses.

    `generation.profile_path` wins when set. Otherwise the task's profile
    directory is searched: the filename encodes the benchmark and the number of
    rows profiled, which the pipeline cannot predict (a spam profile covers the
    whole split, not generation.sample_size), so it matches rather than computes.
    Exactly one profile resolves silently; several is ambiguous and raises,
    naming them, rather than guessing which benchmark the run meant.

    Single source of truth, shared by `_load_benchmark_profile` (which loads the
    file) and `_build_meta` (which records the path as provenance). Profiles are
    gitignored, so `_build_meta`'s copy is the only surviving record of what
    generated a seedless benchmark."""
    gen = config.get("generation") or {}
    if gen.get("profile_path"):
        return gen["profile_path"]
    task_name = task.get_task_name()
    pattern = os.path.join(benchmark_profile_dir(task_name), f"*_{task_name}_profile.json")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        # Hand the pattern back so the not-found error shows the expected shape.
        return pattern
    raise RuntimeError(
        f"{len(matches)} profiles exist for task '{task_name}' in "
        f"{benchmark_profile_dir(task_name)}: " + ", ".join(os.path.basename(m) for m in matches)
        + ". Set generation.profile_path to choose which benchmark to generate from."
    )


def _load_benchmark_profile(config: dict, task) -> dict | None:
    """Load the benchmark profile that drives seedless generation.

    Returns None when generation.seedless is falsy. Runs before the generation
    loop so a missing or un-topic-profiled profile fails before any API spend."""
    gen = config.get("generation") or {}
    if task.get_generation_strategy() == "structured":
        if gen.get("seedless") is False:
            return None
        path = _resolve_benchmark_profile_path(config, task)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if not gen.get("seedless"):
        return None
    from framework.profiling.spec_sampler import load_profile

    path = _resolve_benchmark_profile_path(config, task)
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

def resolve_mode(config: dict, strategy: str) -> str:
    """The mode that actually runs — shared by the session name, _build_meta and
    the generation dispatch so all three always agree.

    `mode` asks where the annotation comes from: `inverse` draws it independently
    and IMPOSES it on the source; `forward` INHERITS it from the source (the seed,
    or the artifact just generated). Defaults follow what each strategy does when
    the config says nothing: `corruption` infers the seed's error (forward), while
    `class_conditional` draws a label and `structured` draws a target structure
    (both inverse).
    """
    default = "forward" if strategy == "corruption" else "inverse"
    return (config.get("generation") or {}).get("mode", default)


def generation_cell_slug(config: dict, strategy: str) -> str:
    """Filesystem-safe label for the generation cell, e.g. "inverse_seeded".

    Naming a session after its setup means a directory listing shows what was
    run without opening results.json, and two cells of the same task can never
    collide in one output directory."""
    # Structured IS on the mode axis — it imposes a sampled target structure
    # (inverse) or lets structure emerge (forward) — and is seedless until
    # seeded structured generation is implemented.
    seeding = (
        "seedless" if strategy == "structured"
        or (config.get("generation") or {}).get("seedless")
        else "seeded"
    )
    return f"{resolve_mode(config, strategy)}_{seeding}"


def resolve_output_paths(config: dict, task_name: str, session: str) -> dict:
    """All artifact paths for one run session, under output.base_dir/<task>/<session>/.

    The session name already carries the setup that produced it (see
    generation_cell_slug), so every run of a task lands in one flat directory."""
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
    set one, else the same default `_load_benchmark_profile` resolves
    internally (see `_resolve_benchmark_profile_path`, shared by both) — and None when
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
    mode = resolve_mode(config, strategy)
    seedless = True if strategy == "structured" else bool(gen.get("seedless"))
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
        "strategy": strategy,
        "mode": mode,
        "seedless": seedless,
        "profile_path": _resolve_benchmark_profile_path(config, task) if seedless else None,
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
        "calibration": _LAST_CALIBRATION,
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
    _profile_driven = False

    if strategy == "structured":
        if gen_cfg.get("seedless") is False:
            # Not impossible — perturbing a real ontology subtree and keeping the
            # perturbation as ground truth is coherent. It is unimplemented, and
            # saying so is what keeps the design open rather than foreclosed.
            raise RuntimeError(
                f"seeded structured generation is not implemented for "
                f"{task.get_task_name()}; it needs a real-artifact corpus and a "
                "perturbation operator. Use seedless: true."
            )
        if profile is None:
            raise RuntimeError(
                f"{task.get_task_name()} structured generation requires a profile."
            )
        feedback_cfg = task.get_feedback_config(gen_cfg)
        feedback_enabled = bool(feedback_cfg.get("enabled", False))
        # Fail before the first API call, not mid-round: enabling the loop
        # without a comparator is a config/implementation error, and the
        # framework's rule is that an unsupported capability says so up front.
        if feedback_enabled and (
            type(task).build_structural_feedback
            is BaseTask.build_structural_feedback
        ):
            raise RuntimeError(
                f"{task.get_task_name()} enables the structured feedback loop "
                "(get_feedback_config) but does not implement "
                "build_structural_feedback()."
            )
        # Delegate the loop itself, like every other strategy. The generator
        # stays ontology-agnostic: it receives callables, never the task.
        rng = random.Random()
        synthetic = generator.generate_structured(
            build_prompt=lambda feedback: task.build_structured_generation_prompt(
                profile, rng=rng, feedback=feedback
            ),
            parse=task.parse_structured_generation_with_diagnostics,
            build_feedback=(
                (lambda artifact: task.build_structural_feedback(
                    profile, artifact, generation_config=gen_cfg))
                if feedback_enabled else None
            ),
            sample_size=sample_size,
            max_parse_attempts=gen_cfg.get("max_parse_attempts", 3),
            max_feedback_rounds=int(feedback_cfg.get("max_rounds", 0)),
            request_delay=gen_cfg.get("request_delay", 0.0),
        )
    elif strategy == "class_conditional":
        # No config sets "mode" explicitly today (spam.json's config comment
        # says so) — the default MUST resolve to "inverse" so that omitting
        # the key keeps reproducing today's production behavior unchanged.
        mode = gen_cfg.get("mode", "inverse")
        seedless = bool(gen_cfg.get("seedless"))
        # The strategy is label -> text, so the class names come from the task.
        # Hardcoding them here would mean every classification task is generated
        # under the first one's vocabulary.
        labels = task.get_class_labels()
        if not labels:
            raise RuntimeError(
                f"{task.get_task_name()} declares the class_conditional strategy "
                "but get_class_labels() returned None — it must return "
                "(positive_label, negative_label)."
            )
        positive_label, negative_label = labels
        negative_prompt = task.get_negative_generation_prompt()
        if not negative_prompt:
            raise RuntimeError(
                f"{task.get_task_name()} does not support class_conditional "
                "generation (no negative_generation_prompt)."
            )
        # Required regardless of seed_policy — generate_class_conditional's
        # signature has no defaults for these, even though same_class/none
        # policies use forward_prompts/seedless_prompts instead of inject_prompt.
        common_kwargs = dict(
            class_prob=class_prob,
            type_dist=error_dist["type_dist"],
            count_dist=error_dist["count_dist"],
            error_descriptions=task.get_error_descriptions(),
            inject_prompt=task.get_inverse_prompt(),
            negative_prompt=negative_prompt,
            positive_label=positive_label,
            negative_label=negative_label,
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
                    render_spec(sample_content_spec(profile, rng, label=negative_label))
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
                for label in (positive_label, negative_label)
            }
            synthetic = generator.generate_class_conditional(
                seed_policy="none",
                specs_by_label=specs_by_label,
                seedless_prompts=seedless_prompts,
                **common_kwargs,
            )
        else:
            forward_prompts = task.get_forward_prompts()
            if not forward_prompts:
                raise RuntimeError(
                    f"{task.get_task_name()} does not support mode=forward with "
                    f"seedless=false (no forward_prompts)."
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
                forward_prompts=forward_prompts,
                **common_kwargs,
            )
    else:
        mode = gen_cfg.get("mode", "forward")
        seedless = bool(gen_cfg.get("seedless"))
        # Hoisted out of the seedless block: forward+seeded needs it too, for the
        # calibrated seed draw. Every run draws fresh (no fixed seed) — pinning
        # it would hand every run in a session the identical seed set and
        # collapse the run-to-run variance the framework exists to measure.
        rng = random.Random()
        if seedless:
            from framework.profiling.spec_sampler import render_spec, sample_content_spec
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
            # forward+seeded: the generator picks its own error type, so the only
            # control input is WHICH seeds it sees. Without calibrated weights
            # get_seed_pool returns real_data untouched.
            seed_weights = (gen_cfg.get("seed_weights")
                            if isinstance(gen_cfg.get("seed_weights"), dict) else None)
            synthetic = generator.generate_forward(
                real_samples=task.get_seed_pool(config, real_data, "forward",
                                                seed_weights=seed_weights, rng=rng),
                error_types=task.get_error_types(),
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

def _resolve_class_prob(config: dict, real_reference, task=None) -> float:
    """P(positive class) for class-conditional generation.

    An explicit float in `generation.class_balance` is a user instruction and
    always wins, calibration included. A calibrated `class_prob` corrects the
    EMPIRICAL balance for differential attrition, so it only applies when
    `class_balance` is `empirical` (the default) — never overriding a balance
    the user asked for by name. With no calibration, `empirical` falls back to
    the real reference's positive fraction."""
    cb = (config.get("generation") or {}).get("class_balance", "empirical")
    if isinstance(cb, (int, float)):
        return float(cb)
    if _LAST_CALIBRATION and isinstance(_LAST_CALIBRATION.get("class_prob"), float):
        return _LAST_CALIBRATION["class_prob"]
    labels = task.get_class_labels() if task is not None else None
    if real_reference and labels:
        positive = labels[0]
        pos = sum(1 for r in real_reference if r.get("label") == positive)
        return pos / len(real_reference)
    # No task, or a task with no class axis: there is no positive label to
    # count, so an empirical fraction is not defined. Guessing one task's
    # label here is how the vocabulary leaked in the first place.
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


def _write_fidelity_artifacts(task, real_reference, all_generated, paths) -> None:
    """Persist the real sample + a {real, generated, fidelity} profile when the
    task supports profiling. No-op for tasks whose build_fidelity_profile returns None."""
    if real_reference is None:
        return
    with open(paths["real_sample"], "w", encoding="utf-8") as f:
        json.dump(real_reference, f, indent=2, ensure_ascii=False)
    real_profile = task.build_fidelity_profile(real_reference)
    if real_profile is None:
        return
    generated_profile = task.build_fidelity_profile(all_generated)
    fidelity = task.compare_fidelity_profiles(real_profile, generated_profile)
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


# ── Generation context ────────────────────────────────────────

def _load_seed_weights(config: dict, task, strategy: str, mode: str | None,
                       seedless: bool) -> dict | None:
    """Calibrated seed weights for cells whose only control input is seed choice.

    GEC forward+seeded never reaches load_error_distribution (the generator picks
    its own error type, so _should_load_error_distribution is False), so its
    calibration artifact has to be resolved here instead.
    """
    if strategy != "corruption" or mode != "forward" or seedless:
        return None
    from framework.calibration.artifact import load_calibration, resolve_calibration_path

    path = resolve_calibration_path(config, task, strategy)
    if not path:
        print(f"[NOTE] no calibration artifact for "
              f"{generation_cell_slug(config, strategy)}; drawing seeds in the "
              f"unweighted first-N order. Build one with: python -m "
              f"framework.calibrate --config <config.yaml>")
        return None
    # Same guard as _apply_calibration: a JSON-valid but structurally corrupt
    # artifact must fall back to today's behavior, never crash the run.
    try:
        payload = load_calibration(path)
        weights = (payload.get("calibrated") or {}).get("seed_weights")
        if not isinstance(weights, dict) or not weights:
            return None
        # EVERY value is coerced, deliberately not `any(float(w) > 0 ...)`: that
        # short-circuits, so {"R:DET": 1.0, "R:PREP": "bad"} would pass the guard
        # and raise inside draw_weighted_seeds mid-run — the exact crash this
        # check exists to prevent.
        values = [float(w) for w in weights.values()]
        if not any(v > 0 for v in values):
            return None
    except (OSError, ValueError, AttributeError, TypeError, KeyError) as e:
        print(f"[WARN] calibration {path!r} could not be read or is malformed "
              f"({e}); drawing seeds in the unweighted first-N order.",
              file=sys.stderr)
        return None

    # Provenance, same as _apply_calibration records for every other cell:
    # artifacts are gitignored, so results.json is the only surviving record of
    # what produced a benchmark. Written only on success and only after
    # build_generation_context's unconditional reset, so a run without an
    # artifact still ends at None. class_prob is not applicable here — seed
    # weights are a corruption-cell control input and carry no class balance.
    global _LAST_CALIBRATION
    _LAST_CALIBRATION = {"path": path,
                         "selected_round": payload.get("selected_round"),
                         "class_prob": None}
    print(f"Calibration: {path} (round {payload.get('selected_round')}) — "
          f"seed weights over {len(weights)} edit types")
    return weights


def build_generation_context(config: dict) -> dict:
    """Everything needed to generate for `config`, resolved once.

    Shared by `run_pipeline` and `framework.calibrate` so both provably build
    the same task, seeds, generator, distributions and class balance. Makes no
    API call: `load_generator` only constructs a client.
    """
    # _LAST_CALIBRATION is set by _apply_calibration, which only runs when
    # _should_load_error_distribution(...) is True below. Strategies/modes that
    # skip that lookup entirely (structured, or corruption forward+seeded) must
    # not let _build_meta report a PREVIOUS config's calibration in this same
    # process (e.g. scripts/compare_models.py loops run_pipeline over configs).
    # Reset unconditionally here, on top of the reset inside _apply_calibration.
    global _LAST_CALIBRATION
    _LAST_CALIBRATION = None

    task          = load_task(config["task"]["name"])
    real_data     = load_real_data(config, task)
    generator     = load_generator(config["generation"])
    judge_call    = _build_judge_call(config, generator)
    evaluator_fns = task.get_evaluator_fns()

    strategy = task.get_generation_strategy()
    # One source of truth: resolve_mode also feeds the session name and _build_meta.
    mode = resolve_mode(config, strategy)
    seedless = (
        True if strategy == "structured"
        else bool(config["generation"].get("seedless"))
    )
    error_dist = (
        load_error_distribution(config, real_data, task)
        if _should_load_error_distribution(strategy, mode, seedless) else None
    )
    profile = _load_benchmark_profile(config, task)

    # Published onto the config so _run_generation's forward+seeded branch (which
    # reads generation.seed_weights) sees them: that cell never calls
    # _apply_calibration, so this is the only place its artifact can be resolved.
    seed_weights = _load_seed_weights(config, task, strategy, mode, seedless)
    if seed_weights:
        config.setdefault("generation", {})["seed_weights"] = seed_weights

    # Real reference feeds class balance, the real baseline, and profiling.
    real_reference = task.get_real_eval_samples(config, real_data)

    return {
        "task": task,
        "real_data": real_data,
        "generator": generator,
        "judge_call": judge_call,
        "evaluator_fns": evaluator_fns,
        "strategy": strategy,
        "mode": mode,
        "seedless": seedless,
        "error_dist": error_dist,
        "seed_weights": seed_weights,
        "profile": profile,
        "real_reference": real_reference,
        "class_prob": _resolve_class_prob(config, real_reference, task),
    }


# ── Main pipeline ─────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """Run the GET pipeline N times, evaluate the generated benchmark (mean±std)
    and — by default — the same models on the real benchmark, profile real-vs-
    generated fidelity, and write all artifacts under one per-session directory."""
    ctx            = build_generation_context(config)
    task           = ctx["task"]
    real_data      = ctx["real_data"]
    generator      = ctx["generator"]
    judge_call     = ctx["judge_call"]
    evaluator_fns  = ctx["evaluator_fns"]
    strategy       = ctx["strategy"]
    error_dist     = ctx["error_dist"]
    profile        = ctx["profile"]
    real_reference = ctx["real_reference"]
    class_prob     = ctx["class_prob"]

    # <timestamp>_<mode>_<seeded|seedless>: timestamp first so a directory
    # listing still sorts chronologically, setup second so it is readable.
    session_id = (f"{datetime.now():%Y%m%d_%H%M%S}"
                  f"_{generation_cell_slug(config, strategy)}")
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

    _write_fidelity_artifacts(task, real_reference, all_generated, paths)
    _render_plots(config, paths)
    print(f"\nResults saved to {paths['results']}")
    return final
