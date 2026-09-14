"""The real baseline a consumer should compare a session's generated scores against.

A session whose task pairs (see BaseTask.paired_real_indices) records, beside
the unpaired `real` point, a per-run `real_paired` block: the real items each
run actually delivered. That is the like-for-like comparator, so every reader
prefers it where it exists.
"""


def real_point(blocks: dict) -> dict:
    """One model's real scores as a point: the paired mean when the session
    paired its real side, else the unpaired `real` block. Nested blocks keep
    their shape -- {"diagnostics": {"tp": {"mean": 8, "std": 1}}} becomes
    {"diagnostics": {"tp": 8}}."""
    paired = blocks.get("real_paired")
    if not paired:
        return dict(blocks.get("real") or {})
    return {name: _means(value) for name, value in paired.items()}


def _means(value):
    if isinstance(value, dict) and set(value) == {"mean", "std"}:
        return value["mean"]
    if isinstance(value, dict):
        return {key: _means(sub) for key, sub in value.items()}
    return value
