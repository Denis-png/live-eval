from openai import OpenAI
from .base_generator import BaseGenerator

# OpenAI-compatible base URLs for providers that mirror the OpenAI API
_BASE_URLS = {
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral":    "https://api.mistral.ai/v1",
    # Note: minimax is routed through AnthropicGenerator (Anthropic-compatible API)
    # "openai" is the default — no base_url needed
}


class OpenAIGenerator(BaseGenerator):
    """
    Generator for all OpenAI-compatible providers:
    openai, groq, openrouter, mistral.
    Set provider in config.yaml; the correct base_url is resolved automatically.
    """

    def __init__(self, config: dict):
        self.model       = config["model"]
        self.temperature = config["temperature"]
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=_BASE_URLS.get(config["provider"]),  # None → default OpenAI endpoint
            timeout=config.get("timeout", 300),       # seconds; SDK default is 600
            max_retries=config.get("max_retries", 1), # SDK default is 2
        )

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
