"""Rescore a finished session from its persisted artifacts — no LLM calls.

Re-evaluates the config's task_models on each generated/run_<N>.json, which
  * restores per-run scores ("runs") for sessions created before they were
    persisted, and
  * adds generated + real-baseline scores for task models that did not exist
    when the session originally ran,
then rebuilds profile.json via task.profile_dataset (backfilling GEC sessions
from before GEC profiling existed). Generation provenance in meta is kept;
a meta["rescored"] stamp records what was recomputed and when.

Usage:
    python -m scripts.rescore_session --config framework/configs/<task>/config.yaml
        SESSION_DIR [SESSION_DIR ...] [--skip-eval] [--skip-profile] [--plots]
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import yaml

from framework.main import _expand_env_vars, _load_dotenv
from framework.pipeline import (
    _evaluate_real_baseline,
    _nest_results,
    aggregate,
    load_task,
)


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_files(session_dir):
    files = glob.glob(os.path.join(session_dir, "generated", "run_*.json"))
    return sorted(files, key=lambda p: int(re.search(r"run_(\d+)", p).group(1)))


def _drift_report(old_results, new_results):
    """Compare old vs recomputed generated means for models present in both;
    a mismatch means the persisted generated data no longer reproduces the
    stored scores and deserves eyes."""
    def _means(block, prefix=""):
        out = {}
        for k, v in (block or {}).items():
            if isinstance(v, dict) and "mean" in v:
                out[f"{prefix}{k}"] = v["mean"]
            elif isinstance(v, dict):
                out.update(_means(v, prefix=f"{prefix}{k}."))
        return out

    lines = []
    for model in set(old_results) & set(new_results):
        old = _means(old_results[model].get("generated"))
        new = _means(new_results[model].get("generated"))
        for key in sorted(set(old) & set(new)):
            if abs(old[key] - new[key]) > 0.005:
                lines.append(f"  [drift] {model} {key}: stored {old[key]:.4f} "
                             f"-> recomputed {new[key]:.4f}")
    return lines


def rescore_session(session_dir, config, *, skip_eval=False, skip_profile=False,
                    plots=False):
    results_path = os.path.join(session_dir, "results.json")
    old = _load_json(results_path)
    meta = old["meta"]
    if config["task"]["name"] != meta["task"]:
        raise ValueError(
            f"config task '{config['task']['name']}' does not match session task "
            f"'{meta['task']}' ({session_dir}); pass a matching --config."
        )

    task = load_task(meta["task"])
    evaluator_fns = task.get_evaluator_fns()
    runs_data = [_load_json(p) for p in _run_files(session_dir)]
    real_path = os.path.join(session_dir, "real_sample.json")
    real_reference = _load_json(real_path) if os.path.exists(real_path) else None

    if not skip_eval:
        if not runs_data:
            raise ValueError(f"no generated/run_*.json in {session_dir}")
        per_run_samples = [task.get_eval_samples(s) for s in runs_data]
        per_run_texts = [[s["text"] for s in samples] for samples in per_run_samples]

        # Models outer, runs inner: each model is loaded once and released
        # before the next (coedit-large on CPU makes per-run reloading and
        # keeping all models resident both wasteful).
        all_run_scores = [{} for _ in runs_data]
        for model_config in config["task_models"]:
            model = task.get_model(model_config)
            for i, samples in enumerate(per_run_samples):
                predictions = model.predict(per_run_texts[i])
                results = [{**s, "prediction": p} for s, p in zip(samples, predictions)]
                all_run_scores[i][model_config["name"]] = {
                    name: evaluator_fns[name](results) for name in task.get_evaluators()
                }
            del model

        real_scores = (
            _evaluate_real_baseline(task, config, real_reference, evaluator_fns)
            if real_reference else {}
        )
        final = _nest_results(aggregate(all_run_scores), real_scores, all_run_scores)

        for line in _drift_report(old["results"], final):
            print(line)

        strategy = task.get_generation_strategy()
        meta = {
            **meta,
            # Recomputed from the task's actual strategy, not carried over —
            # older sessions predate the mode/strategy provenance fix and may
            # have a stale mode string with no strategy key at all.
            "strategy": strategy,
            # Carried over from the old meta — rescoring has no access to the
            # original generation config, only the persisted meta — and
            # defaulted per-strategy exactly like _build_meta/_run_generation:
            # "inverse" for class_conditional (spam), "forward" otherwise.
            # Every strategy has a real mode now, so it is always preserved,
            # never nulled out.
            "mode": meta.get("mode", "inverse" if strategy == "class_conditional" else "forward"),
            # Recomputed from the actual run-file count, not carried over from
            # the old meta — keeps a merged/reconciled session's provenance
            # honest (e.g. two 3-run sessions merged into one 6-run session).
            "num_runs": len(runs_data),
            "runs_completed": len(runs_data),
            "partial": False,
            "effective_samples_per_run": [len(s) for s in per_run_samples],
            "real_baseline": bool(real_scores),
            "rescored": {
                "at": datetime.now().isoformat(timespec="seconds"),
                "task_models": [m["name"] for m in config["task_models"]],
                "note": "scores recomputed offline from persisted generated runs",
            },
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": final}, f, indent=2)
        print(f"Rescored results written to {results_path}")

    if not skip_profile and real_reference:
        real_profile = task.profile_dataset(real_reference)
        if real_profile is not None:
            all_generated = [item for run in runs_data for item in run]
            generated_profile = task.profile_dataset(all_generated)
            fidelity = task.compare_profiles(real_profile, generated_profile)
            profile_path = os.path.join(session_dir, "profile.json")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump({"real": real_profile, "generated": generated_profile,
                           "fidelity": fidelity}, f, indent=2, ensure_ascii=False)
            print(f"Fidelity profile written to {profile_path}")

    if plots:
        from framework.plotting.session import render_session
        render_session(session_dir)

    return session_dir


def main():
    parser = argparse.ArgumentParser(
        description="Rescore finished sessions from persisted artifacts (no LLM calls).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session_dirs", nargs="+", help="Session directories to rescore")
    parser.add_argument("--config", required=True,
                        help="Config YAML supplying task_models (task must match the session, "
                             "e.g. framework/configs/spam/config.yaml)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Only rebuild profile.json; leave results.json untouched")
    parser.add_argument("--skip-profile", action="store_true",
                        help="Only rescore results.json; leave profile.json untouched")
    parser.add_argument("--plots", action="store_true",
                        help="Re-render the session's figures afterwards")
    args = parser.parse_args()

    _load_dotenv()
    with open(args.config, encoding="utf-8") as f:
        config = _expand_env_vars(yaml.safe_load(f))
    device_pref = (config.get("compute") or {}).get("device", "cpu")
    os.environ["FRAMEWORK_DEVICE"] = str(device_pref)

    for session_dir in args.session_dirs:
        try:
            rescore_session(session_dir, config, skip_eval=args.skip_eval,
                            skip_profile=args.skip_profile, plots=args.plots)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
