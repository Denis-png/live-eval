"""Sample per-sample content specs from a benchmark profile (stdlib only).

Seedless generation replaces a real seed sentence with a *content spec* drawn
from the profile: a topic, a target length, and a set of style features. Style
rates are treated as Bernoulli parameters and sampled independently, so the
generated style distribution matches the profile by construction rather than by
instruction.
"""

from __future__ import annotations

import json
import os

_PROFILE_HELP = (
    "Run: python -m framework.profile_dataset --task <task> "
    "--config <config.yaml> --topics"
)

# Style rates renderable as prompt instructions. punctuation_density is
# excluded on purpose: it is a mean density, not a per-text probability.
_STYLE_PHRASES = {
    "question_rate": "phrase it as a question",
    "exclaim_rate": "include an exclamation mark",
    "first_person_rate": "write in the first person",
    "second_person_rate": "address the reader directly",
    "contraction_rate": "use contractions",
    "digit_rate": "include a number",
    "uppercase_word_rate": "include an ALL-CAPS word",
    "texting_slang_rate": "use texting slang",
}


def _topic_mapping(block: dict) -> dict:
    """Topic mapping from either profile shape: the nested {"topics": {...}}
    envelope written by profile_topics, or a bare {name: {...}} mapping."""
    if isinstance(block, dict) and "topics" in block:
        return block["topics"] or {}
    return block or {}


def load_profile(path: str, topics_key: str = "topics") -> dict:
    """Load and validate a benchmark profile for seedless generation."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"Seedless generation needs a benchmark profile, but {path!r} does "
            f"not exist. {_PROFILE_HELP}"
        )
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    if profile.get("profile_version", 1) < 2:
        raise RuntimeError(
            f"Profile {path!r} has profile_version "
            f"{profile.get('profile_version', 1)}; seedless generation needs "
            f"version 2 or newer. {_PROFILE_HELP}"
        )
    if not profile.get(topics_key):
        raise RuntimeError(
            f"Profile {path!r} has no {topics_key!r} block — it was written "
            f"without LLM topic profiling, which seedless generation needs. "
            f"{_PROFILE_HELP}"
        )
    # Validate that the inner topic mapping is non-empty, distinguishing
    # "no topics block" from "profiled but labeled nothing".
    if topics_key == "topics_per_label":
        # At least one label must have topics.
        topics_per_label = profile.get(topics_key, {})
        if not any(_topic_mapping(topics_per_label.get(label, {}))
                   for label in topics_per_label):
            raise RuntimeError(
                f"Profile {path!r} has {topics_key!r}, but no label has any "
                f"topics — the profile was written without successful topic "
                f"labeling. {_PROFILE_HELP}"
            )
    else:
        # Single topics block must be non-empty.
        topics_block = profile.get(topics_key, {})
        if not _topic_mapping(topics_block):
            raise RuntimeError(
                f"Profile {path!r} has {topics_key!r}, but it is empty — "
                f"the profile was written without successful topic labeling. "
                f"{_PROFILE_HELP}"
            )
    return profile


def _parse_bin_label(label: str) -> tuple[int, int | None]:
    """"11-15" -> (11, 15); "51+" -> (51, None)."""
    if label.endswith("+"):
        return int(label[:-1]), None
    low, _, high = label.partition("-")
    return int(low), int(high)


def _weighted_choice(weights: dict, rng, what: str):
    """Draw a key from {key: weight}; raises when no positive weight exists."""
    keys = list(weights)
    values = [max(0.0, float(weights[k])) for k in keys]
    if not keys or sum(values) <= 0:
        raise RuntimeError(
            f"Profile has no usable {what} distribution (all weights are zero). "
            f"{_PROFILE_HELP}"
        )
    return rng.choices(keys, weights=values, k=1)[0]


def _slice_blocks(profile: dict, label: str | None, side: str) -> tuple[dict, dict, dict]:
    """Return (topics, word_length_block, style_rates) for the requested slice."""
    if label is not None:
        topics_block = (profile.get("topics_per_label") or {}).get(label) or {}
        topics = _topic_mapping(topics_block)
        lengths = ((profile.get("length_distributions_per_label") or {})
                   .get(label) or {}).get("words") or {}
        style = (profile.get("style_per_label") or {}).get(label) or {}
        return topics, lengths, style
    topics_block = (profile.get("topics") or {})
    topics = _topic_mapping(topics_block)
    lengths = ((profile.get("length_distributions") or {}).get(side) or {}).get("words") or {}
    style = (profile.get("style") or {}).get(side) or {}
    return topics, lengths, style


def sample_content_spec(
    profile: dict,
    rng,
    *,
    label: str | None = None,
    side: str = "correct",
) -> dict:
    """Draw one content spec: topic, target word count, and style features."""
    topics, lengths, style = _slice_blocks(profile, label, side)

    topic = _weighted_choice(
        {name: block.get("fraction", 0.0) for name, block in topics.items()},
        rng, "topic",
    )
    bin_label = _weighted_choice(lengths.get("bins") or {}, rng, "length")
    low, high = _parse_bin_label(bin_label)
    if high is None:
        high = max(low, int((lengths.get("quantiles") or {}).get("p90", low)))
    target_words = rng.randint(low, high)

    features = [
        key for key, rate in style.items()
        if key in _STYLE_PHRASES and rng.random() < float(rate)
    ]
    return {
        "topic": topic,
        "topic_description": (topics.get(topic) or {}).get("description", ""),
        "target_words": target_words,
        "style_features": features,
    }


def render_spec(spec: dict) -> str:
    """Render a content spec as the prompt-ready phrase for {spec}."""
    parts = [f"topic: {spec['topic']}"]
    if spec.get("topic_description"):
        parts.append(f"({spec['topic_description']})")
    parts.append(f"roughly {spec['target_words']} words")
    parts.extend(_STYLE_PHRASES[key] for key in spec.get("style_features", []))
    return "; ".join(parts)
