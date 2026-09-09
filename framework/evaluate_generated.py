"""Standalone evaluation-only entry point.

Evaluate an existing generated JSON file and stop before generation, benchmark
profiling, or plotting:

    python -m framework.evaluate_generated --config framework/configs/gec/config.yaml \
      --generated framework/data/generated/generated.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from framework.main import _expand_env_vars, _load_dotenv
from framework.pipeline import load_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing generated samples without generating or profiling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to framework config YAML")
    parser.add_argument("--generated", required=True, help="Path to generated JSON samples")
    parser.add_argument(
        "--output",
        default="framework/data/evaluations/results.json",
        help="Path to write evaluation results JSON",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return _expand_env_vars(yaml.safe_load(f))


def _prepare_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = deepcopy(config)
    if "task" not in prepared or "name" not in prepared["task"]:
        raise ValueError("Invalid config: missing key 'task.name'")
    if not prepared.get("task_models"):
        raise ValueError("Invalid config: 'task_models' must be a non-empty list")
    return _resolve_task_model_api_keys(prepared)


def _resolve_task_model_api_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve only credentials needed by evaluation/task models.

    Generation-only provider credentials are deliberately ignored here: the
    generated samples already exist, so a missing generation key must not block
    evaluation.
    """
    api_keys = config.get("api_keys") or {}
    for model in config.get("task_models") or []:
        if model.get("api_key"):
            continue
        provider = model.get("provider")
        if provider:
            key = api_keys.get(provider) or os.getenv(f"{provider.upper()}_API_KEY", "")
            if not key:
                raise ValueError(
                    f"No API key found for evaluation model provider '{provider}'. "
                    f"Set {provider.upper()}_API_KEY in .env or override "
                    f"api_keys.{provider}."
                )
            model["api_key"] = key
        elif model.get("type") == "claude":
            key = api_keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY", "")
            if key:
                model["api_key"] = key
    return config


def _load_generated(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Generated file '{path}' must contain a JSON list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Generated file '{path}' must contain only JSON objects")
    return rows


def evaluate_generated_file(
    config: dict[str, Any],
    generated_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one existing generated JSON file and write results.

    The function uses the existing task/model/evaluator interfaces exactly as
    the full pipeline does, but never calls any generation, profiling, or
    plotting helpers.
    """
    prepared = _prepare_config(config)
    generated = _load_generated(generated_path)
    task = load_task(prepared["task"]["name"])
    evaluator_fns = task.get_evaluator_fns()
    eval_samples = task.get_eval_samples(generated)
    texts = [sample["text"] for sample in eval_samples]

    scores: dict[str, dict[str, Any]] = {}
    for model_config in prepared["task_models"]:
        model = task.get_model(model_config)
        predictions = model.predict(texts)
        results = [
            {**sample, "prediction": prediction}
            for sample, prediction in zip(eval_samples, predictions)
        ]
        scores[model_config["name"]] = {
            name: evaluator_fns[name](results) for name in task.get_evaluators()
        }

    payload = {
        "meta": {
            "task": task.get_task_name(),
            "generated_path": str(generated_path),
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "num_samples": len(eval_samples),
            "task_models": [model["name"] for model in prepared["task_models"]],
        },
        "results": scores,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    args = parse_args()
    _load_dotenv()
    try:
        payload = evaluate_generated_file(
            load_config(args.config),
            args.generated,
            args.output,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print("Evaluation-only run complete")
    print(f"Samples : {payload['meta']['num_samples']}")
    print(f"Models  : {payload['meta']['task_models']}")
    print(f"Output  : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
