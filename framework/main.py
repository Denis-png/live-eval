import argparse
import os
import re

import yaml

from framework.pipeline import run_pipeline

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
    return config


def _load_dotenv() -> None:
    """Load .env into os.environ. Warns loudly if python-dotenv is missing —
    silent failures here cause confusing 'empty api_key' errors downstream."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "[WARN] python-dotenv not installed — .env will be ignored. "
            "Install with: pip install python-dotenv"
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


def _resolve_api_keys(config: dict) -> dict:
    """Inject provider-specific keys from api_keys[*] into call sites that need them.
    Explicit api_key values already in the config are preserved."""
    api_keys = config.get("api_keys") or {}

    def _inject(block: dict, label: str) -> None:
        if not block or block.get("api_key"):
            return
        provider = block.get("provider")
        block["api_key"] = api_keys.get(provider, "")
        if not block["api_key"] and provider:
            print(
                f"[WARN] No API key found for {label} provider '{provider}'. "
                f"Set {provider.upper()}_API_KEY in .env or override api_keys.{provider}."
            )

    _inject(config.get("generation"), "generator")

    judge = config.get("judge")
    if judge and judge.get("enabled", True):
        _inject(judge, "judge")

    for model in config.get("task_models") or []:
        if model.get("type") == "claude" and not model.get("api_key"):
            model["api_key"] = api_keys.get("anthropic", "")

    return config


def main():
    args = parse_args()
    _load_dotenv()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = _expand_env_vars(config)
    config = apply_overrides(config, args)
    config = _resolve_api_keys(config)

    # Propagate compute.device into the env for lazy-loaded evaluator/model code.
    device_pref = (config.get("compute") or {}).get("device", "auto")
    os.environ["FRAMEWORK_DEVICE"] = str(device_pref)

    print(f"Task     : {config['task']['name']}")
    print(f"Provider : {config['generation']['provider']}")
    print(f"Model    : {config['generation']['model']}")
    print(f"Runs     : {config['generation']['num_runs']}")
    print(f"Models   : {[m['name'] for m in config['task_models']]}")

    results = run_pipeline(config)

    print("\n" + "=" * 55)
    print("FINAL RESULTS (mean ± std across runs)")
    print("=" * 55)
    for model, scores in results.items():
        print(f"\n{model}:")
        for evaluator, values in scores.items():
            if isinstance(values, dict) and "mean" in values:
                print(f"  {evaluator}: {values['mean']} ± {values['std']}")
            else:
                for sub, v in values.items():
                    print(f"  {evaluator}.{sub}: {v['mean']} ± {v['std']}")


if __name__ == "__main__":
    main()
