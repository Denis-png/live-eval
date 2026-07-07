import argparse
import os
import re
import sys

import yaml

try:
    from framework.pipeline import run_pipeline
except ModuleNotFoundError as exc:
    if exc.name != "framework":
        raise
    sys.exit(
        "framework is not importable — run as a module from the repo root "
        "(the parent of framework/):\n    python -m framework.main [flags]"
    )

DEFAULT_CONFIG = "framework/configs/config.yaml"
_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GET Framework — Generate, Evaluate, Trash",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",       default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--task",         help="Task name (e.g. gec)")
    parser.add_argument("--provider",     help="Generator provider (groq|openai|anthropic|...)")
    parser.add_argument("--model",        help="Generator model name")
    parser.add_argument("--runs",         type=int, help="Number of GET runs")
    parser.add_argument("--sample-size",  type=int, dest="sample_size", help="Synthetic samples per run")
    parser.add_argument("--mode",         choices=["forward", "inverse"], help="Generation mode")
    parser.add_argument("--output",       help="Output base directory (output.base_dir)")
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable the LLM-as-judge filter (--judge / --no-judge)")
    parser.add_argument("--real-baseline", action=argparse.BooleanOptionalAction, default=None,
                        help="Evaluate models on the real benchmark too (--real-baseline / --no-real-baseline)")
    return parser.parse_args()


def apply_overrides(config: dict, args) -> dict:
    """Apply CLI flags on top of YAML config. Only overrides keys that were explicitly passed."""
    # Use `is not None` so falsy-but-valid overrides (e.g. --runs 0) still apply.
    if args.task is not None:
        config["task"]["name"] = args.task
    if args.provider is not None:
        config["generation"]["provider"] = args.provider
    if args.model is not None:
        config["generation"]["model"] = args.model
    if args.runs is not None:
        config["generation"]["num_runs"] = args.runs
    if args.sample_size is not None:
        config["generation"]["sample_size"] = args.sample_size
    if getattr(args, "mode", None) is not None:
        config["generation"]["mode"] = args.mode
    if getattr(args, "output", None) is not None:
        config.setdefault("output", {})["base_dir"] = args.output
    if getattr(args, "judge", None) is not None:
        config.setdefault("judge", {})["enabled"] = args.judge
    if getattr(args, "real_baseline", None) is not None:
        config.setdefault("evaluation", {})["real_baseline"] = args.real_baseline
    return config


def validate_config(config: dict) -> None:
    """Check required keys and cross-field invariants up front, with errors that
    name the offending config path — instead of a raw KeyError traceback deep
    inside the pipeline. Dataset requirements depend on dataset.source."""
    from framework.data_loading import (
        SUPPORTED_LOCAL_FORMATS,
        infer_format,
        resolve_dataset_config,
    )

    required = {
        "task": ["name"],
        "generation": ["provider", "model", "num_runs", "sample_size"],
    }
    problems = []
    for section, keys in required.items():
        block = config.get(section)
        if not isinstance(block, dict):
            problems.append(f"missing section '{section}'")
            continue
        problems.extend(
            f"missing key '{section}.{key}'" for key in keys if key not in block
        )
    if not config.get("task_models"):
        problems.append("'task_models' must be a non-empty list of models to evaluate")

    if isinstance(config.get("dataset"), dict):
        ds = resolve_dataset_config(config["dataset"])
        if ds["source"] == "local":
            if not ds["path"]:
                problems.append("missing key 'dataset.local.path' (required for source: local)")
            elif (ds["format"] or infer_format(ds["path"])) not in SUPPORTED_LOCAL_FORMATS:
                problems.append(
                    f"cannot determine local dataset format for '{ds['path']}' — set "
                    f"'dataset.local.format' to one of: {', '.join(SUPPORTED_LOCAL_FORMATS)}"
                )
        elif ds["source"] == "huggingface":
            if not ds["name"]:
                problems.append("missing key 'dataset.huggingface.name' (required for source: huggingface)")
            if not ds["split"]:
                problems.append("missing key 'dataset.huggingface.split' (required for source: huggingface)")
        else:
            problems.append(
                f"'dataset.source' must be 'huggingface' or 'local' (got '{ds['source']}')"
            )

    if not problems:
        gen, ds = config["generation"], config["dataset"]
        if gen["num_runs"] < 1:
            problems.append(f"'generation.num_runs' must be >= 1 (got {gen['num_runs']})")
        mode = gen.get("mode", "forward")
        if mode not in ("forward", "inverse"):
            problems.append(f"'generation.mode' must be 'forward' or 'inverse' (got '{mode}')")

    if problems:
        raise ValueError(
            "Invalid config:\n  - " + "\n  - ".join(problems)
        )


def _load_dotenv() -> None:
    """Load .env into os.environ. Warns loudly if python-dotenv is missing —
    silent failures here cause confusing 'empty api_key' errors downstream."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "[WARN] python-dotenv not installed — .env will be ignored. "
            "Install with: pip install python-dotenv",
            file=sys.stderr,
        )
        return
    load_dotenv()


def _expand_env_vars(value):
    """Recursively expand ${VAR} references in strings using os.environ."""
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    return value


def _resolve_api_keys(config: dict, strict: bool = False) -> dict:
    """Inject provider-specific keys from api_keys[*] into call sites that need them.
    Explicit api_key values already in the config are preserved.

    strict=True raises on a missing generator/judge key instead of warning —
    a run without a key burns minutes before failing, so entry points should
    fail fast. Non-strict is for intermediate resolution (e.g. compare_models'
    base config, whose provider may be superseded per entry)."""
    api_keys = config.get("api_keys") or {}

    def _inject(block: dict, label: str) -> None:
        if not block or block.get("api_key"):
            return
        provider = block.get("provider")
        block["api_key"] = api_keys.get(provider, "")
        if not block["api_key"] and provider:
            message = (
                f"No API key found for {label} provider '{provider}'. "
                f"Set {provider.upper()}_API_KEY in .env or override api_keys.{provider}."
            )
            if strict:
                raise ValueError(message)
            print(f"[WARN] {message}", file=sys.stderr)

    _inject(config.get("generation"), "generator")

    judge = config.get("judge")
    if judge and judge.get("enabled", True):
        _inject(judge, "judge")

    for model in config.get("task_models") or []:
        if model.get("type") == "claude" and not model.get("api_key"):
            model["api_key"] = api_keys.get("anthropic", "")

    return config


def format_results_lines(results: dict) -> list[str]:
    """Render the nested {model: {generated, real?}} results as printable lines."""
    lines = []
    for model, blocks in results.items():
        lines.append(f"\n{model}:")
        gen = blocks.get("generated", {})
        real = blocks.get("real", {})
        for ev, val in gen.items():
            if isinstance(val, dict) and "mean" in val:
                base = f"  generated.{ev}: {val['mean']} ± {val['std']}"
                if ev in real:
                    base += f"   | real.{ev}: {real[ev]}"
                lines.append(base)
            else:  # nested metric (e.g. errant.precision)
                for sub, v in val.items():
                    lines.append(f"  generated.{ev}.{sub}: {v['mean']} ± {v['std']}")
        for ev, val in real.items():
            if ev not in gen:
                lines.append(f"  real.{ev}: {val}")
    return lines


def main():
    args = parse_args()
    _load_dotenv()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = _expand_env_vars(config)
    config = apply_overrides(config, args)
    try:
        validate_config(config)
        config = _resolve_api_keys(config, strict=True)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    # Propagate compute.device into the env for lazy-loaded evaluator/model code.
    device_pref = (config.get("compute") or {}).get("device", "auto")
    os.environ["FRAMEWORK_DEVICE"] = str(device_pref)

    print(f"Task     : {config['task']['name']}")
    print(f"Mode     : {config['generation'].get('mode', 'forward')}")
    print(f"Provider : {config['generation']['provider']}")
    print(f"Model    : {config['generation']['model']}")
    print(f"Runs     : {config['generation']['num_runs']}")
    print(f"Models   : {[m['name'] for m in config['task_models']]}")

    try:
        results = run_pipeline(config)
    except (RuntimeError, ValueError) as e:
        # User-facing pipeline failures (0 usable samples, bad source_field,
        # unknown provider/task) — exit cleanly instead of dumping a traceback.
        sys.exit(f"\n[ERROR] {e}")

    print("\n" + "=" * 55)
    print("FINAL RESULTS (generated mean ± std across runs | real baseline)")
    print("=" * 55)
    for line in format_results_lines(results):
        print(line)


if __name__ == "__main__":
    main()
