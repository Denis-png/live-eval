"""Provider-neutral generator construction."""

from __future__ import annotations

_ANTHROPIC_BASE_URLS = {
    "minimax": "https://api.minimax.io/anthropic",
    # "anthropic" -> None (default endpoint)
}


def load_generator(config: dict):
    """Instantiate the configured text generator provider."""
    provider = config["provider"]
    if provider in ("openai", "groq", "openrouter", "mistral"):
        from framework.generators.openai_generator import OpenAIGenerator

        return OpenAIGenerator(config)
    if provider in ("anthropic", "minimax"):
        from framework.generators.anthropic_generator import AnthropicGenerator

        if not config.get("base_url") and provider in _ANTHROPIC_BASE_URLS:
            config = {**config, "base_url": _ANTHROPIC_BASE_URLS[provider]}
        return AnthropicGenerator(config)
    if provider == "google":
        from framework.generators.google_generator import GoogleGenerator

        return GoogleGenerator(config)
    raise ValueError(
        f"Unknown provider: '{provider}'. "
        f"Supported: openai, groq, openrouter, mistral, anthropic, minimax, google."
    )
