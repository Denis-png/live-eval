"""Damped multiplicative control law for generation-distribution calibration.

The distribution handed to `_sample_categories` is a *request*; the distribution
re-detected on generated text is the *output*; the real benchmark profile is the
*setpoint*. The generator honours categories at different rates, so
`measured ~= request * compliance` per category, and a ratio update inverts
exactly that. Pure functions over dicts: no I/O, no API calls, no randomness.
"""

from __future__ import annotations

from framework.profiling.fidelity import jensen_shannon_divergence

_MIN_DENOM = 1e-12


def project_measured(target: dict, measured: dict) -> dict:
    """Restrict `measured` to `target`'s key space and renormalize.

    The setpoint and the measurement do not share a key space:
    `profile_error_distribution` covers only the task's supported vocabulary,
    while `profile_gec_edit_types` reports every ERRANT type it sees. Without
    projection the unsupported mass dilutes every ratio and a perfect generator
    still shows nonzero divergence. Returns all-zeros when nothing in `measured`
    falls inside the target's keys — callers treat that as "no measurement".
    """
    restricted = {k: max(0.0, float(measured.get(k, 0.0))) for k in target}
    total = sum(restricted.values())
    if total <= 0:
        return {k: 0.0 for k in target}
    return {k: v / total for k, v in restricted.items()}


def update_request(
    request: dict,
    target: dict,
    measured: dict,
    *,
    alpha: float = 0.5,
    epsilon: float = 1e-3,
) -> dict:
    """One damped step of iterative proportional fitting.

        request'[i] ∝ request[i] * (target[i] / measured[i]) ** alpha

    alpha damps the step (measurement is noisy; a full step oscillates).
    epsilon is Laplace smoothing on the measurement AND a floor on the request,
    so a category delivered zero times neither divides by zero nor becomes
    permanently unreachable. Categories with `target[i] == 0` stay 0: calibration
    must never introduce a category the benchmark does not exhibit.
    """
    projected = project_measured(target, measured)
    if not any(projected.values()):
        # Nothing landed in the target's key space — no information to act on.
        return dict(target)

    updated: dict = {}
    for key, t in target.items():
        t = float(t)
        if t <= 0:
            updated[key] = 0.0
            continue
        r = max(float(request.get(key, 0.0)), epsilon, _MIN_DENOM)
        m = max(projected.get(key, 0.0) + epsilon, _MIN_DENOM)
        updated[key] = r * ((t / m) ** alpha)

    total = sum(updated.values())
    if total <= 0:
        return dict(target)
    return {k: v / total for k, v in updated.items()}


def jsd_report(targets: dict[str, dict], measured: dict[str, dict]) -> dict[str, float]:
    """Per-control-input Jensen-Shannon divergence, computed in projected space.

    Convergence therefore means "the mix of *requestable* categories matches" —
    edit mass outside the vocabulary is reported separately as a diagnostic and
    never enters the loop, because it cannot be controlled.
    """
    return {
        name: jensen_shannon_divergence(
            target, project_measured(target, measured.get(name) or {})
        )
        for name, target in targets.items()
    }


def worst(report: dict[str, float]) -> float:
    """The worst dimension's divergence. Convergence requires every dimension
    inside tolerance, so the maximum is the scalar that ranks rounds."""
    return max(report.values()) if report else float("inf")


def converged(report: dict[str, float], tolerance: float) -> bool:
    """True when every measured dimension is inside tolerance. An empty report
    is NOT converged: nothing measured is not the same as everything matching."""
    return bool(report) and worst(report) <= tolerance


def select_best(rounds: list[dict]) -> int:
    """Index of the round with the lowest worst-dimension divergence, ties going
    to the earlier round.

    Measurement is noisy, so the final round is not reliably the best. Because
    round 0 is by construction the uncalibrated distribution, selecting the best
    guarantees calibration can never ship something worse than current behavior.
    """
    if not rounds:
        raise ValueError("select_best() needs at least one round")
    scores = [worst(r.get("jsd") or {}) for r in rounds]
    return scores.index(min(scores))


def stalled(rounds: list[dict], patience: int = 2) -> bool:
    """True when the best score has not improved in the last `patience` rounds."""
    if len(rounds) <= patience:
        return False
    scores = [worst(r.get("jsd") or {}) for r in rounds]
    return min(scores[-patience:]) >= min(scores[:-patience])
