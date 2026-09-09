"""Standalone generation-only entry point.

Run one synthetic-data generation batch and stop before evaluation, fidelity
profiling, or plotting:

    python -m framework.generate_dataset --config framework/configs/gec/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from framework.main import _expand_env_vars, _load_dotenv, _resolve_api_keys
from framework import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic samples without running evaluation or profiling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to framework config YAML")
    parser.add_argument(
        "--output",
        default="framework/data/generated/generated.json",
        help="Path to write generated JSON samples",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Override generation.sample_size for this generation-only run",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return _expand_env_vars(yaml.safe_load(f))


def _prepare_config(config: dict[str, Any], sample_size: int | None = None) -> dict[str, Any]:
    prepared = deepcopy(config)
    if sample_size is not None:
        prepared.setdefault("generation", {})["sample_size"] = sample_size
    if "task" not in prepared or "name" not in prepared["task"]:
        raise ValueError("Invalid config: missing key 'task.name'")
    generation = prepared.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Invalid config: missing section 'generation'")
    for key in ("provider", "model", "sample_size"):
        if key not in generation:
            raise ValueError(f"Invalid config: missing key 'generation.{key}'")
    return _resolve_api_keys(prepared, strict=True)


def _build_generation_only_context(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve only the objects required to generate samples.

    This intentionally mirrors pipeline.build_generation_context except for the
    evaluation-only pieces: no evaluator functions and no task models are
    constructed here.
    """
    task = pipeline.load_task(config["task"]["name"])
    real_data = pipeline.load_real_data(config, task)
    generator = pipeline.load_generator(config["generation"])
    judge_call = pipeline._build_judge_call(config, generator)

    strategy = task.get_generation_strategy()
    mode = pipeline.resolve_mode(config, strategy)
    seedless = (
        True if strategy == "structured"
        else bool(config["generation"].get("seedless"))
    )
    error_dist = (
        pipeline.load_error_distribution(config, real_data, task)
        if pipeline._should_load_error_distribution(strategy, mode, seedless) else None
    )
    profile = pipeline._load_benchmark_profile(config, task)

    seed_weights = pipeline._load_seed_weights(config, task, strategy, mode, seedless)
    if seed_weights:
        config.setdefault("generation", {})["seed_weights"] = seed_weights

    real_reference = task.get_real_eval_samples(config, real_data)

    return {
        "task": task,
        "real_data": real_data,
        "generator": generator,
        "judge_call": judge_call,
        "strategy": strategy,
        "mode": mode,
        "seedless": seedless,
        "error_dist": error_dist,
        "seed_weights": seed_weights,
        "profile": profile,
        "real_reference": real_reference,
        "class_prob": pipeline._resolve_class_prob(config, real_reference, task),
    }


def generate_dataset_once(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Generate one batch of samples and write it as JSON.

    This intentionally reuses the existing pipeline generation context and
    dispatch, but stops before task-model evaluation, fidelity profiling, and
    plotting. ``generation.num_runs`` is ignored here: one invocation writes one
    generated dataset.
    """
    prepared = _prepare_config(config, sample_size)
    ctx = _build_generation_only_context(prepared)
    synthetic = pipeline._run_generation(
        ctx["generator"],
        ctx["task"],
        prepared,
        ctx["real_data"],
        ctx["error_dist"],
        ctx["judge_call"],
        ctx["class_prob"],
        profile=ctx["profile"],
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)
    return synthetic


def main() -> None:
    args = parse_args()
    _load_dotenv()
    try:
        synthetic = generate_dataset_once(
            load_config(args.config),
            args.output,
            sample_size=args.sample_size,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print("Generation-only run complete")
    print(f"Samples : {len(synthetic)}")
    print(f"Output  : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
