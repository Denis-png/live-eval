"""Merge N sessions for the SAME (task, generation model, mode, seedless) cell
into one session.

One-off consolidation tool. Forward and inverse are genuinely different
generation cells now — not just a label on an otherwise-identical process —
and seedless is a third, independent axis, so a session may only be merged
with another that matches on all four: task, generation model, mode, and
seedless. Only sessions that are real, independent replicate runs of the
IDENTICAL process get concatenated.

`mode` is compared via the same per-strategy default `scripts.analyze_results
._strategy_of` uses (mode-or-else-inverse-for-class_conditional-else-forward),
so an archived session with `"mode": null` (the old `_build_meta` forced that
for spam) still lines up correctly against a new-style session of the same
cell that records its resolved mode explicitly, instead of being rejected as
a false mismatch or — worse — merged with a session from a different cell.

This concatenates the matching sessions' `generated/run_*.json` (renumbered)
into one output session, then reuses `scripts.rescore_session.rescore_session`
to recompute results.json/profile.json from the combined run count. No new
LLM calls; no new scoring logic.

Usage:
    python -m scripts.merge_sessions --config framework/configs/spam/config.yaml \\
        --out framework/data/runs/spam/openrouter_minimax_m3_merged \\
        SESSION_DIR SESSION_DIR [SESSION_DIR ...]
"""
import argparse
import json
import os
import shutil
import sys

import yaml

from framework.main import _expand_env_vars, _load_dotenv
from scripts.analyze_results import _strategy_of
from scripts.rescore_session import _load_json, _run_files, rescore_session


def merge_sessions(session_dirs, out_dir, config, *, plots=False):
    if len(session_dirs) < 2:
        raise ValueError("merge_sessions needs at least 2 source session dirs")

    metas = [_load_json(os.path.join(d, "results.json"))["meta"] for d in session_dirs]
    task, model, cell = metas[0]["task"], metas[0]["model"], _strategy_of(metas[0])
    for d, m in zip(session_dirs, metas):
        if (m["task"], m["model"], _strategy_of(m)) != (task, model, cell):
            raise ValueError(
                f"session {d} is ({m['task']}, {m['model']}, {_strategy_of(m)}), expected "
                f"({task}, {model}, {cell}) — merge_sessions only combines replicate runs "
                f"of the SAME task+model+mode+seedless cell."
            )

    real_samples = [_load_json(os.path.join(d, "real_sample.json"))
                    for d in session_dirs if os.path.exists(os.path.join(d, "real_sample.json"))]
    if real_samples and any(rs != real_samples[0] for rs in real_samples[1:]):
        print(f"[WARN] source sessions have differing real_sample.json — using "
              f"{session_dirs[0]}'s (same dataset/reference sample should make these "
              f"identical; a mismatch means they weren't really replicate runs).",
              file=sys.stderr)

    os.makedirs(os.path.join(out_dir, "generated"), exist_ok=True)
    run_idx = 0
    for d in session_dirs:
        for run_path in _run_files(d):
            run_idx += 1
            shutil.copy(run_path, os.path.join(out_dir, "generated", f"run_{run_idx}.json"))
    total_runs = run_idx

    if real_samples:
        with open(os.path.join(out_dir, "real_sample.json"), "w", encoding="utf-8") as f:
            json.dump(real_samples[0], f, indent=2, ensure_ascii=False)

    # Seed results.json from the first source so rescore_session has a meta
    # template (provider/model/dataset/judge/...) to preserve; merged_from
    # survives rescore_session's meta rebuild since it only overwrites a
    # known subset of keys.
    seed = _load_json(os.path.join(session_dirs[0], "results.json"))
    seed["meta"]["merged_from"] = [
        {"session": d, "created": m["created"], "runs": m["runs_completed"]}
        for d, m in zip(session_dirs, metas)
    ]
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(seed, f, indent=2)

    print(f"Merged {len(session_dirs)} sessions ({total_runs} runs total) into {out_dir}")
    rescore_session(out_dir, config, plots=plots)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Merge replicate sessions for the same (task, model) into one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session_dirs", nargs="+", help="Source session directories (>= 2)")
    parser.add_argument("--out", required=True, help="Output session directory")
    parser.add_argument("--config", required=True,
                        help="Config YAML supplying task_models for the post-merge rescore")
    parser.add_argument("--plots", action="store_true",
                        help="Render the merged session's figures afterwards")
    args = parser.parse_args()

    _load_dotenv()
    with open(args.config, encoding="utf-8") as f:
        config = _expand_env_vars(yaml.safe_load(f))
    os.environ["FRAMEWORK_DEVICE"] = str((config.get("compute") or {}).get("device", "cpu"))

    try:
        merge_sessions(args.session_dirs, args.out, config, plots=args.plots)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
