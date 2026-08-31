"""Calibrate a task's generation distributions against its real benchmark.

Run once per (benchmark, cell); the artifact is then reused by every session:

    python -m framework.calibrate --config framework/configs/spam/config.yaml

Calibration is a SEPARATE phase from measurement. Steering runs inside a scored
session would make them non-i.i.d. and quietly turn results.json's mean+-std --
the core GET instability signal -- into "generator noise plus controller
settling". So this emits a tuned spec and the GET session runs unchanged at it.
"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime

from framework.calibration.artifact import (
    default_calibration_path,
    write_calibration,
)
from framework.calibration.class_balance import correct_class_prob
from framework.calibration.controller import (
    converged,
    jsd_report,
    select_best,
    stalled,
    update_request,
)

DEFAULT_ROUNDS = 3
DEFAULT_ALPHA = 0.5
DEFAULT_TOLERANCE = 0.1
# Below this a 5-category rate carries ~0.08 standard error against a 0.1
# tolerance: the update would be steering on noise rather than on bias.
MIN_SAMPLE_SIZE = 100
MIN_INFORMATIVE = 50


def calibration_settings(config: dict, args=None) -> dict:
    """Merge the optional `calibration:` block with defaults and CLI overrides.

    generation.sample_size is deliberately NOT inherited as-is: it is tuned for
    eval cost, not estimation precision.
    """
    block = config.get("calibration") or {}
    gen = config.get("generation") or {}
    settings = {
        "rounds": block.get("rounds", DEFAULT_ROUNDS),
        "alpha": block.get("alpha", DEFAULT_ALPHA),
        "tolerance": block.get("tolerance", DEFAULT_TOLERANCE),
        "sample_size": block.get(
            "sample_size", max(gen.get("sample_size", 0), MIN_SAMPLE_SIZE)
        ),
    }
    for key in settings:
        value = getattr(args, key, None) if args is not None else None
        if value is not None:
            settings[key] = value
    return settings


def informative_count(task, rows: list[dict], profile: dict | None = None) -> int:
    """Samples the measurement is actually estimated from, which is not the
    round's sample size: SPAM rows for classification (HAM carries no signals);
    for corruption, the surviving pairs ERRANT actually annotated —
    `profile_gec_edit_types`'s own `n_annotated`, read off the SAME profiling
    pass `_measure` already made rather than re-annotating every pair a second
    time just to count them."""
    if task.get_generation_strategy() == "class_conditional":
        return sum(1 for r in rows if r.get("label") == "SPAM")
    return (profile or {}).get("n_annotated", 0)


def _measure(task, rows: list[dict]) -> dict:
    """Re-detect the full profile on generated rows using the task's own
    profiler, so real and generated are measured with the same instrument.

    Returns the WHOLE profile rather than pre-selecting the calibration keys:
    callers also need `n_annotated`/`supported_fraction` off the same rows, and
    profiling (a full ERRANT pass for GEC) is expensive enough that doing it
    twice per round is worth avoiding."""
    return task.profile_dataset(rows)


def run_calibration(
    config: dict,
    *,
    rounds: int = DEFAULT_ROUNDS,
    alpha: float = DEFAULT_ALPHA,
    tolerance: float = DEFAULT_TOLERANCE,
    sample_size: int | None = None,
    output_path: str | None = None,
) -> dict:
    """Iterate generate -> profile -> correct, writing the artifact each round."""
    from framework import pipeline

    # The setpoint MUST be the real benchmark, never a previously-written
    # calibration artifact: build_generation_context is also what a normal run
    # uses to CONSUME an artifact (via load_error_distribution / seed weights),
    # so calling it on the caller's config would let an existing artifact's
    # already-calibrated error_dist/class_prob/seed_weights become this round's
    # "empirical" target, drifting further from the benchmark on every
    # re-calibration and publishing stale seed weights onto the caller's own
    # config dict in the process. Force the lookup off on a private copy;
    # `default_calibration_path`/`generation_cell_slug` below still read the
    # original `config`, so artifact filenames are unaffected.
    base_config = copy.deepcopy(config)
    base_config.setdefault("generation", {})["calibration_path"] = None
    ctx = pipeline.build_generation_context(base_config)
    task, strategy = ctx["task"], ctx["strategy"]

    keys = task.get_calibration_keys()
    if not keys:
        raise RuntimeError(
            f"Task '{task.get_task_name()}' does not support calibration "
            f"(get_calibration_keys() returned None). Cell: "
            f"{pipeline.generation_cell_slug(config, strategy)}."
        )
    # Corruption forward+seeded injects no distribution at all — the generator
    # identifies the seed's error itself. Its one control input is WHICH seeds it
    # sees, so that cell calibrates seed weights instead (Task 8).
    seed_mode = (strategy == "corruption" and ctx["mode"] == "forward"
                 and not ctx["seedless"])
    if ctx["error_dist"] is None and not seed_mode:
        raise RuntimeError(
            f"Cell {pipeline.generation_cell_slug(config, strategy)} of task "
            f"'{task.get_task_name()}' samples no error distribution, so there "
            "is nothing to calibrate."
        )
    if seed_mode and "type_dist" not in keys:
        raise RuntimeError(
            f"Cell {pipeline.generation_cell_slug(config, strategy)} of task "
            f"'{task.get_task_name()}' calibrates seed choice, which needs a "
            "'type_dist' calibration key."
        )

    # Both dimensions are still MEASURED in seed mode; only type_dist is steered.
    measure_keys = dict(keys)
    if seed_mode:
        # Forward mode leaves the edit COUNT entirely to the generator, so there
        # is no control input for count_dist: keeping it in the target would make
        # converged() unsatisfiable and burn the whole round budget for nothing.
        # It stays in the round trace as a diagnostic instead.
        keys = {"type_dist": keys["type_dist"]}
        # ctx["error_dist"] is None here, so the setpoint comes from the seed
        # pool's own ERRANT profile — the same instrument _measure uses on the
        # generated rows.
        seed_profile = task.profile_dataset(ctx["real_reference"])
        target = {name: dict(seed_profile.get(key) or {}) for name, key in keys.items()}
    else:
        target = {name: dict(ctx["error_dist"][name]) for name in keys}
    # Routed through calibration_settings (not read off config["generation"]
    # directly) so a programmatic caller that omits `sample_size` still gets
    # the documented max(generation.sample_size, MIN_SAMPLE_SIZE) floor — the
    # same settings the CLI path (main()) already applies.
    settings_size = sample_size or calibration_settings(config)["sample_size"]

    # Only the positive class carries signals, so calibrating at the empirical
    # balance would waste most of the budget. is_positive only GATES the
    # _sample_categories call, so positives-only is measurement-equivalent.
    forced_class_prob = 1.0 if strategy == "class_conditional" else None
    class_prob = forced_class_prob if forced_class_prob is not None else ctx["class_prob"]

    run_config = copy.deepcopy(base_config)
    run_config["generation"]["sample_size"] = settings_size

    path = output_path or default_calibration_path(
        config, task.get_task_name(), strategy, settings_size
    )
    payload = {
        "meta": {
            "task": task.get_task_name(),
            "cell": pipeline.generation_cell_slug(config, strategy),
            "benchmark": pipeline.benchmark_slug(config),
            "sample_size": settings_size,
            "generator": {"provider": config["generation"].get("provider"),
                          "model": config["generation"].get("model")},
            "alpha": alpha, "tolerance": tolerance, "rounds": rounds,
            "forced_class_prob": forced_class_prob,
            "timestamp": f"{datetime.now():%Y-%m-%dT%H:%M:%S}",
        },
        "target": target,
        "calibrated": dict(target),
        "selected_round": 0,
        "rounds": [],
    }

    request = {name: dict(dist) for name, dist in target.items()}
    for round_idx in range(rounds + 1):
        print(f"\n{'='*50}\nCALIBRATION ROUND {round_idx} / {rounds}\n{'='*50}")
        if seed_mode:
            # The control input is the seed draw, not an injected distribution.
            run_config["generation"]["seed_weights"] = request["type_dist"]
            round_dist = ctx["error_dist"]
        else:
            round_dist = {"type_dist": request["type_dist"],
                          "count_dist": request["count_dist"]}
        synthetic = pipeline._run_generation(
            ctx["generator"], task, run_config, ctx["real_data"], round_dist,
            ctx["judge_call"], class_prob, profile=ctx["profile"],
        )
        profile = _measure(task, synthetic)
        measured_all = {name: (profile.get(key) or {}) for name, key in measure_keys.items()}
        measured = {name: measured_all[name] for name in keys}
        n_informative = informative_count(task, synthetic, profile)
        if n_informative < MIN_INFORMATIVE:
            print(f"[WARN] round {round_idx} measured on {n_informative} informative "
                  f"samples (< {MIN_INFORMATIVE}); the update may chase noise.",
                  file=sys.stderr)

        report = jsd_report(target, measured)
        entry = {
            "round": round_idx,
            "request": {n: dict(d) for n, d in request.items()},
            "measured": {n: dict(d) for n, d in measured.items()},
            "jsd": report,
            "informative_samples": n_informative,
            "supported_fraction": profile.get("supported_fraction"),
        }
        # Uncontrollable dimensions (seed mode's edit count) are observed but
        # never steered on; recording them keeps the trace diagnosable.
        diagnostic = {n: dict(d) for n, d in measured_all.items() if n not in keys}
        if diagnostic:
            entry["diagnostic"] = diagnostic
        payload["rounds"].append(entry)
        best = select_best(payload["rounds"])
        payload["selected_round"] = best
        payload["calibrated"] = {
            n: dict(d) for n, d in payload["rounds"][best]["request"].items()
        }
        if seed_mode:
            # The name a run looks for: pipeline._load_seed_weights reads
            # calibrated.seed_weights, not calibrated.type_dist.
            payload["calibrated"]["seed_weights"] = dict(
                payload["calibrated"]["type_dist"])
        write_calibration(path, payload)
        for name, value in report.items():
            print(f"  {name} JSD: {value:.4f}")

        if converged(report, tolerance):
            print(f"Converged at round {round_idx}.")
            break
        if stalled(payload["rounds"]):
            print(f"No improvement for 2 rounds; stopping at round {round_idx}.")
            break
        if round_idx >= rounds:
            print("Round budget exhausted without converging; "
                  f"keeping best round {best}.")
            break

        request = {
            name: update_request(request[name], target[name], measured.get(name) or {},
                                 alpha=alpha)
            for name in target
        }

    # Stage B — class balance. Stage A ran at class_prob 1.0 to make every
    # sample informative, which makes the balance unmeasurable there; one round
    # at the real balance with Stage A's distributions supplies the attrition.
    if strategy == "class_conditional":
        print(f"\n{'='*50}\nCALIBRATION STAGE B — class balance\n{'='*50}")
        stage_b = pipeline._run_generation(
            ctx["generator"], task, run_config,
            ctx["real_data"],
            {"type_dist": payload["calibrated"]["type_dist"],
             "count_dist": payload["calibrated"]["count_dist"]},
            ctx["judge_call"], ctx["class_prob"], profile=ctx["profile"],
        )
        attrition = getattr(ctx["generator"], "last_class_attrition", None) or {}
        corrected = correct_class_prob(ctx["class_prob"], attrition,
                                       n=len(stage_b))
        payload["meta"]["class_attrition"] = attrition
        if corrected is None:
            print("Class balance inside the noise floor; class_prob unchanged "
                  f"at {ctx['class_prob']:.4f}.")
        else:
            print(f"Class balance corrected: {ctx['class_prob']:.4f} -> "
                  f"{corrected:.4f}")
            payload["calibrated"]["class_prob"] = corrected
        write_calibration(path, payload)

    print(f"\nCalibration written to {path} (selected round "
          f"{payload['selected_round']}).")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate generation distributions against the real benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Task config YAML (task.name selects the task)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Damping in (0,1]; lower is more conservative")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Per-dimension JSD at which the loop stops")
    parser.add_argument("--sample-size", dest="sample_size", type=int, default=None)
    parser.add_argument("--mode", choices=("forward", "inverse"), default=None)
    parser.add_argument("--seedless", dest="seedless", action="store_true",
                        default=None)
    parser.add_argument("--no-seedless", dest="seedless", action="store_false")
    parser.add_argument("--output", default=None,
                        help="Artifact path (default: beside the task's profiles)")
    return parser.parse_args()


def main() -> None:
    # Same entry-point contract as framework.main: load .env BEFORE the config is
    # read (load_config expands ${VAR} against os.environ), then inject
    # api_keys[provider] into the generation/judge blocks. Without this the
    # generator is constructed with no api_key and dies on a bare KeyError.
    from framework.main import _load_dotenv, _resolve_api_keys
    from framework.profile_dataset import load_config

    args = parse_args()
    _load_dotenv()
    config = load_config(args.config)
    if args.mode is not None:
        config["generation"]["mode"] = args.mode
    if args.seedless is not None:
        config["generation"]["seedless"] = args.seedless

    # strict: a calibration without a key would otherwise burn the dataset load
    # and the first round's setup before failing.
    try:
        config = _resolve_api_keys(config, strict=True)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    settings = calibration_settings(config, args)
    print(f"Task     : {config['task']['name']}")
    print(f"Provider : {config['generation']['provider']} / {config['generation']['model']}")
    print(f"Rounds   : {settings['rounds']}  alpha: {settings['alpha']}  "
          f"tolerance: {settings['tolerance']}")
    print(f"Samples  : {settings['sample_size']} per round")
    run_calibration(
        config,
        rounds=settings["rounds"], alpha=settings["alpha"],
        tolerance=settings["tolerance"], sample_size=settings["sample_size"],
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
