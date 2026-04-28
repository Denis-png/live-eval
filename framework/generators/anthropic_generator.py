from anthropic import Anthropic
from .base_generator import BaseGenerator


class AnthropicGenerator(BaseGenerator):
    """Generator for Anthropic models (claude-*)."""

    def __init__(self, config: dict):
        self.model       = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens  = config["max_tokens"]
        self.client      = Anthropic(api_key=config["api_key"])

    def _call_api(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
