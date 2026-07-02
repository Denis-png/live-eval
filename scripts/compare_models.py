"""Run several generation models over the SAME benchmark sample and compare.

Reads a normal config YAML plus a top-level `generation_models:` list; runs the
unmodified pipeline once per entry, writing a per-model results file, then a
combined comparison file and a printed table. The same-sample guarantee comes
from the framework's deterministic first-N sampling (dataset.* held constant).
"""
import copy
import json
import os
import re

import yaml

from framework.main import (
    _expand_env_vars,
    _load_dotenv,
    _resolve_api_keys,
    apply_overrides,
    parse_args,
)
from framework.pipeline import run_pipeline


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug: runs of non-alphanumerics collapse to '_'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def _per_model_config(base: dict, entry: dict) -> dict:
    """Deep-copy base, overlay one generation_models entry, re-resolve the API key,
    and route output to a per-model results file under output.results_dir."""
    cfg = copy.deepcopy(base)
    cfg["generation"] = {**cfg["generation"], **entry}
    cfg["generation"].pop("api_key", None)  # force re-resolution for the new provider
    _resolve_api_keys(cfg)

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

    all_results = {}
    for entry in entries:
        cfg = _per_model_config(base_config, entry)
        label = f"{cfg['generation']['provider']}/{cfg['generation']['model']}"
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
    args = parse_args()
    _load_dotenv()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = _expand_env_vars(config)
    # Apply only sample-shaping overrides; do NOT pass --provider/--model to the driver
    # (per-model provider/model come from generation_models).
    config = apply_overrides(config, args)
    config = _resolve_api_keys(config)

    device_pref = (config.get("compute") or {}).get("device", "auto")
    os.environ["FRAMEWORK_DEVICE"] = str(device_pref)

    run_comparison(config)


if __name__ == "__main__":
    main()
