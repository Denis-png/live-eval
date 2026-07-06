"""Run several generation models over the SAME benchmark sample and compare.

Reads a normal config YAML plus a top-level `generation_models:` list; runs the
unmodified pipeline once per entry, writing a per-model results file, then a
combined comparison file and a printed table. The same-sample guarantee comes
from the framework's deterministic first-N sampling (dataset.* held constant).
"""
import argparse
import copy
import json
import os
import re
import sys

import yaml

from framework.main import (
    _expand_env_vars,
    _load_dotenv,
    _resolve_api_keys,
    apply_overrides,
    validate_config,
)
from framework.pipeline import run_pipeline


def parse_compare_args(argv=None):
    """Own parser, deliberately WITHOUT --provider/--model: per-model provider
    and model come from the generation_models list, and a CLI provider would
    silently leak into entries that omit one. Only sample-shaping flags are
    accepted (they apply identically to every entry, keeping the comparison fair)."""
    parser = argparse.ArgumentParser(
        description="Compare several generation models over the SAME benchmark sample. "
                    "Per-model provider/model come from the config's generation_models list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default="framework/configs/config.yaml",
                        help="Path to config YAML (must contain generation_models)")
    parser.add_argument("--task",        help="Task name (e.g. gec)")
    parser.add_argument("--runs",        type=int, help="Number of GET runs per model")
    parser.add_argument("--sample-size", type=int, dest="sample_size",
                        help="Synthetic samples per run")
    return parser.parse_args(argv)


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug: runs of non-alphanumerics collapse to '_'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def _per_model_config(base: dict, entry: dict) -> dict:
    """Deep-copy base, overlay one generation_models entry, re-resolve the API key,
    and route output to a per-model results file under output.results_dir."""
    cfg = copy.deepcopy(base)
    cfg["generation"] = {**cfg["generation"], **entry}
    cfg["generation"].pop("api_key", None)  # force re-resolution for the new provider
    _resolve_api_keys(cfg, strict=True)

    task = cfg["task"]["name"]
    mode = cfg["generation"].get("mode", "forward")
    provider = cfg["generation"]["provider"]
    model = cfg["generation"]["model"]

    out_dir = (cfg.get("output") or {}).get("results_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{task}_{mode}_{_slug(provider)}_{_slug(model)}.json"
    cfg.setdefault("output", {})["results_path"] = os.path.join(out_dir, fname)
    return cfg


def _flatten(scores: dict) -> dict:
    """Flatten evaluator scores to {name: {mean, std}}, expanding nested metrics
    (e.g. errant.precision) so they fit one table column each."""
    flat = {}
    for ev, v in scores.items():
        if isinstance(v, dict) and "mean" in v:
            flat[ev] = v
        elif isinstance(v, dict):
            for sub, sv in v.items():
                if isinstance(sv, dict) and "mean" in sv:
                    flat[f"{ev}.{sub}"] = sv
    return flat


def _print_table(all_results: dict) -> None:
    task_models = sorted({tm for r in all_results.values() for tm in r})
    for tm in task_models:
        print(f"\n=== task model: {tm} ===")
        rows = {gl: _flatten(r[tm]) for gl, r in all_results.items() if tm in r}
        cols = sorted({c for f in rows.values() for c in f})
        print("gen_model".ljust(30) + "".join(c.ljust(20) for c in cols))
        for gl, f in rows.items():
            cells = "".join(
                (f"{f[c]['mean']:.3f}±{f[c]['std']:.3f}" if c in f else "-").ljust(20)
                for c in cols
            )
            print(gl.ljust(30) + cells)


def run_comparison(base_config: dict) -> dict:
    entries = base_config.get("generation_models") or []
    if not entries:
        raise ValueError(
            "compare_models requires a non-empty 'generation_models' list in the config."
        )

    # Build every per-model config up front: a missing API key for entry 3
    # should fail here, not after entries 1 and 2 have already burned API spend.
    configs = []
    for entry in entries:
        cfg = _per_model_config(base_config, entry)
        configs.append((f"{cfg['generation']['provider']}/{cfg['generation']['model']}", cfg))

    all_results = {}
    for label, cfg in configs:
        print(f"\n{'#'*60}\nGEN MODEL: {label}  ->  {cfg['output']['results_path']}\n{'#'*60}")
        all_results[label] = run_pipeline(cfg)

    task = base_config["task"]["name"]
    mode = (base_config.get("generation") or {}).get("mode", "forward")
    out_dir = (base_config.get("output") or {}).get("results_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, f"comparison_{task}_{mode}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined comparison written to {combined_path}")

    _print_table(all_results)
    return all_results


def main():
    args = parse_compare_args()
    _load_dotenv()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = _expand_env_vars(config)
    # apply_overrides expects main.py's full flag set; fill the flags this
    # driver deliberately doesn't expose with None (= no override).
    overrides = argparse.Namespace(
        provider=None, model=None, mode=None, output=None, judge=None, **vars(args)
    )
    config = apply_overrides(config, overrides)
    try:
        validate_config(config)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    # Non-strict here: the base provider may be superseded per generation_models
    # entry — each entry is strictly re-resolved in _per_model_config.
    config = _resolve_api_keys(config)

    device_pref = (config.get("compute") or {}).get("device", "auto")
    os.environ["FRAMEWORK_DEVICE"] = str(device_pref)

    run_comparison(config)


if __name__ == "__main__":
    main()
