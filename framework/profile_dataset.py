"""Standalone CLI for profiling the original benchmark dataset.

Run with:
    python -m framework.profile_dataset
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from framework.profiling.dataset_profiler import save_profile_json
from framework.profiling.gec_profiler import profile_gec_rows

DEFAULT_GEC_OUTPUT = "framework/data/profiles/gec_profile.json"
DEFAULT_SPAM_OUTPUT = "framework/data/profiles/spam_profile.json"
DEFAULT_GEC_DATASET = "agentlans/grammar-correction"
DEFAULT_GEC_SPLIT = "train"
_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def parse_args() -> argparse.Namespace:
    """Parse standalone profiling CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Profile the original benchmark dataset without running GET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=("gec", "spam"), default="gec", help="Task to profile")
    parser.add_argument("--config", required=True,
                        help="Path to config YAML (e.g. framework/configs/gec/config.yaml)")
    parser.add_argument("--output", help="Path to output JSON profile")
    parser.add_argument(
        "--topics", action="store_true",
        help="Also run LLM topic profiling (uses the 'profiling' config block, "
             "falling back to 'generation')",
    )
    parser.add_argument(
        "--topic-sample-size", type=int, default=200,
        help="Texts sampled per topic pass (per label for spam)",
    )
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


def _build_topic_call(config: dict):
    """Build call_api for --topics from the 'profiling' config block.

    Mirrors pipeline._build_judge_call's fallback: profiling block missing
    provider/model -> warn and use the 'generation' block; neither usable ->
    SystemExit. The api_key resolves from api_keys[provider] unless the block
    carries an explicit one (standalone CLI never goes through
    main._resolve_api_keys)."""
    import framework.pipeline as pipeline

    profiling_cfg = config.get("profiling") or {}
    if profiling_cfg.get("provider") and profiling_cfg.get("model"):
        cfg = dict(profiling_cfg)
    else:
        generation_cfg = config.get("generation") or {}
        if not (generation_cfg.get("provider") and generation_cfg.get("model")):
            raise SystemExit(
                "--topics requires a 'profiling' (or fallback 'generation') "
                "config block with provider and model set."
            )
        print(
            "[WARN] profiling block missing provider/model — falling back to "
            "the generation LLM.",
            file=sys.stderr,
        )
        cfg = dict(generation_cfg)

    if not cfg.get("api_key"):
        cfg["api_key"] = (config.get("api_keys") or {}).get(cfg["provider"], "")
    if not cfg["api_key"]:
        raise SystemExit(
            f"No API key for topic profiling provider '{cfg['provider']}'. "
            f"Set {cfg['provider'].upper()}_API_KEY in .env or api_keys.{cfg['provider']}."
        )
    print(f"Topics   : {cfg['provider']} / {cfg['model']}")
    return pipeline.load_generator(cfg).call_api


def _fmt_length(words: dict[str, Any]) -> str:
    """One-line digest of a length_distribution's "words" block."""
    bins = words.get("bins") or {}
    quantiles = words.get("quantiles") or {}
    busiest = max(bins, key=bins.get) if bins else "-"
    return (f"mostly {busiest} words "
            f"(p50 {quantiles.get('p50', 0)}, p90 {quantiles.get('p90', 0)})")


def _fmt_style(style: dict[str, Any], limit: int = 4) -> str:
    """Top style features by rate. punctuation_density is a mean, not a rate,
    so it is reported separately rather than ranked alongside the others."""
    rates = sorted(
        ((k[:-5].replace("_", " "), v) for k, v in style.items()
         if k.endswith("_rate") and v),
        key=lambda kv: -kv[1],
    )
    head = ", ".join(f"{name} {rate:.0%}" for name, rate in rates[:limit]) or "none"
    return f"{head}; punct density {style.get('punctuation_density', 0):.2f}"


def _fmt_vocab(vocab: dict[str, Any]) -> str:
    return (f"TTR {vocab.get('type_token_ratio', 0):.3f}, "
            f"hapax {vocab.get('hapax_fraction', 0):.0%}, "
            f"avg word {vocab.get('avg_word_length', 0):.1f} chars")


def _fmt_syntax(syntax: dict[str, Any]) -> str:
    top_pos = sorted((syntax.get("pos_dist") or {}).items(), key=lambda kv: -kv[1])[:3]
    return (f"depth {(syntax.get('parse_depth') or {}).get('mean', 0):.1f}, "
            f"clauses {(syntax.get('clauses_per_text') or {}).get('mean', 0):.1f}, "
            f"top POS " + "/".join(f"{pos} {rate:.0%}" for pos, rate in top_pos))


def _fmt_topics(block: dict[str, Any], limit: int = 3) -> str:
    top = sorted((block.get("topics") or {}).items(),
                 key=lambda item: -item[1]["fraction"])[:limit]
    return ", ".join(f"{name} ({b['fraction']:.0%})" for name, b in top) or "none"


def _profile_gec(config: dict[str, Any], output: str, topic_call=None, topic_sample_size: int = 200) -> str:
    """Profile normalized GEC rows using the existing dataset loader."""
    from framework.pipeline import load_real_data, load_task

    if (config.get("task") or {}).get("name") != "gec":
        dataset_config = config.get("dataset") or {}
        config = {
            **config,
            "task": {"name": "gec"},
            "dataset": {
                **dataset_config,
                "source": "huggingface",
                "name": DEFAULT_GEC_DATASET,
                "split": DEFAULT_GEC_SPLIT,
            },
        }

    task = load_task("gec")
    rows = load_real_data(config, task)
    profile = profile_gec_rows(rows)

    from framework.profiling.syntax_stats import syntax_profile

    correct_texts = [
        row["correct"] for row in rows if isinstance(row.get("correct"), str)
    ]
    syntax = syntax_profile(correct_texts)
    if syntax is not None:
        profile["syntax"] = syntax

    if topic_call is not None:
        from framework.profiling.topics import profile_topics

        profile["topics"] = profile_topics(
            correct_texts, topic_call, sample_size=topic_sample_size
        )

    output_path = save_profile_json(profile, output)

    print("\nGEC dataset profile summary")
    print("=" * 27)
    print(f"Samples                  : {profile['num_samples']}")
    print(f"Avg incorrect word count : {profile['incorrect_word_count']['mean']}")
    print(f"Avg correct word count   : {profile['correct_word_count']['mean']}")
    print(f"Avg similarity           : {profile['similarity']['stats']['mean']}")
    lengths = (profile.get("length_distributions") or {}).get("correct") or {}
    if lengths.get("words"):
        print(f"Correct length           : {_fmt_length(lengths['words'])}")
    style = (profile.get("style") or {}).get("correct")
    if style:
        print(f"Correct style            : {_fmt_style(style)}")
    vocabulary = (profile.get("vocabulary") or {}).get("correct")
    if vocabulary:
        print(f"Correct vocabulary       : {_fmt_vocab(vocabulary)}")
    if profile.get("syntax"):
        print(f"Correct syntax           : {_fmt_syntax(profile['syntax'])}")
    if "topics" in profile:
        print(f"Top topics               : {_fmt_topics(profile['topics'])}")
    print(f"Output                   : {output_path}")
    return output_path


def _profile_spam(config: dict[str, Any], output: str, topic_call=None, topic_sample_size: int = 200) -> str:
    """Profile the raw spam dataset without using SpamTask.parse_row()."""
    from framework.profiling.spam_profiler import (
        DEFAULT_SPAM_DATASET,
        DEFAULT_SPAM_SPLIT,
        profile_spam_dataset,
    )

    dataset_config = config.get("dataset") or {}
    task_config = config.get("task") or {}
    use_config_dataset = task_config.get("name") == "spam"
    dataset_name = dataset_config.get("name") if use_config_dataset else DEFAULT_SPAM_DATASET
    split = dataset_config.get("split") if use_config_dataset else DEFAULT_SPAM_SPLIT
    streaming = dataset_config.get("streaming", False) if use_config_dataset else False
    hf_token = (
        dataset_config.get("hf_token")
        or (config.get("api_keys") or {}).get("huggingface")
        or None
    )

    profile = profile_spam_dataset(
        dataset_name=dataset_name or DEFAULT_SPAM_DATASET,
        split=split or DEFAULT_SPAM_SPLIT,
        streaming=streaming,
        hf_token=hf_token,
        topic_call_api=topic_call,
        topic_sample_size=topic_sample_size,
    )
    output_path = save_profile_json(profile, output)
    labels = profile["label_distribution"]

    signals = profile.get("spam_signals", {})
    print("\nSpam dataset profile summary")
    print("=" * 40)
    print(f"Samples    : {profile['num_samples']}")
    print(f"HAM count  : {labels.get('HAM', 0)}")
    print(f"SPAM count : {labels.get('SPAM', 0)}")
    print()
    print("Spam signals (fraction of texts per label):")
    for label, stats in signals.items():
        print(f"  {label}:")
        for signal, rate in stats.items():
            print(f"    {signal:<16}: {rate:.2%}")
    lengths = profile.get("length_distributions_per_label") or {}
    styles = profile.get("style_per_label") or {}
    vocabularies = profile.get("vocabulary_per_label") or {}
    topics = profile.get("topics_per_label") or {}
    if lengths or styles or vocabularies or topics:
        print()
        print("Characteristics per label:")
        for label in sorted(set(lengths) | set(styles) | set(vocabularies) | set(topics)):
            print(f"  {label}:")
            if (lengths.get(label) or {}).get("words"):
                print(f"    length    : {_fmt_length(lengths[label]['words'])}")
            if styles.get(label):
                print(f"    style     : {_fmt_style(styles[label])}")
            if vocabularies.get(label):
                print(f"    vocabulary: {_fmt_vocab(vocabularies[label])}")
            if topics.get(label):
                print(f"    topics    : {_fmt_topics(topics[label])}")
    print(f"\nOutput     : {output_path}")
    return output_path


def main() -> None:
    """Load the selected benchmark dataset, profile it, and save JSON output."""
    args = parse_args()
    config = load_config(args.config)
    topic_call = _build_topic_call(config) if args.topics else None

    if args.task == "spam":
        _profile_spam(config, args.output or DEFAULT_SPAM_OUTPUT,
                      topic_call, args.topic_sample_size)
    else:
        _profile_gec(config, args.output or DEFAULT_GEC_OUTPUT,
                     topic_call, args.topic_sample_size)


if __name__ == "__main__":
    main()
