"""Standalone CLI for profiling the original benchmark dataset.

Run with:
    python -m framework.profile_dataset
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from framework.profiling.dataset_profiler import save_profile_json
from framework.profiling.gec_profiler import profile_gec_rows

DEFAULT_CONFIG = "framework/configs/config.yaml"
DEFAULT_OUTPUT = "framework/data/profiles/gec_profile.json"
_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def parse_args() -> argparse.Namespace:
    """Parse standalone profiling CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Profile the original benchmark dataset without running GET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output JSON profile")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config for dataset settings."""
    import yaml

    with Path(path).open(encoding="utf-8") as f:
        return _expand_env_vars(yaml.safe_load(f))


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} placeholders for standalone config loading."""
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda match: os.environ.get(match.group(1), ""), value)
    return value


def main() -> None:
    """Load the configured benchmark dataset, profile it, and save JSON output."""
    args = parse_args()

    from framework.pipeline import load_real_data

    config = load_config(args.config)

    rows = load_real_data(config)
    profile = profile_gec_rows(rows)
    output_path = save_profile_json(profile, args.output)

    print("\nDataset profile summary")
    print("=" * 24)
    print(f"Samples                  : {profile['num_samples']}")
    print(f"Avg incorrect word count : {profile['incorrect_word_count']['mean']}")
    print(f"Avg correct word count   : {profile['correct_word_count']['mean']}")
    print(f"Avg similarity           : {profile['similarity']['stats']['mean']}")
    print(f"Output                   : {output_path}")


if __name__ == "__main__":
    main()
