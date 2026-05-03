from anthropic import Anthropic
from .base_generator import BaseGenerator


class AnthropicGenerator(BaseGenerator):
    """Generator for Anthropic models and Anthropic-compatible providers
    (e.g. MiniMax) — set `base_url` in config to redirect to a compatible host."""

    def __init__(self, config: dict):
        self.model       = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens  = config["max_tokens"]
        self.client      = Anthropic(
            api_key=config["api_key"],
            base_url=config.get("base_url"),       # None → official Anthropic endpoint
            timeout=config.get("timeout", 300),       # seconds; SDK default is 600
            max_retries=config.get("max_retries", 1), # SDK default is 2
        )

    def _call_api(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Skip ThinkingBlocks (emitted by reasoning models like MiniMax-M2.7)
        # and concatenate all text blocks in case the response is split.
        return "".join(b.text for b in response.content if b.type == "text")
