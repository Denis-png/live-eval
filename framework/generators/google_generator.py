import google.generativeai as genai
from google.api_core import exceptions as gax_exceptions
from google.api_core.retry import Retry

from .base_generator import BaseGenerator


class GoogleGenerator(BaseGenerator):
    """Generator for Google AI models (gemini-*).

    Honors the same `temperature`, `max_tokens`, `timeout`, and `max_retries`
    knobs as the OpenAI/Anthropic generators so config behaves consistently
    across providers.
    """

    def __init__(self, config: dict):
        self.temperature = config["temperature"]
        self.timeout     = config.get("timeout", 300)
        # google-generativeai retries via google.api_core.Retry, which counts
        # attempts differently from the OpenAI/Anthropic SDKs; map max_retries
        # (extra attempts) onto a retry policy only when > 0.
        max_retries = config.get("max_retries", 1)
        self._retry = (
            Retry(
                predicate=gax_exceptions.if_transient_error,
                maximum=max_retries + 1,
            )
            if max_retries
            else None
        )
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(
            model_name=config["model"],
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=config.get("max_tokens"),
            ),
        )

    def call_api(self, prompt: str) -> str:
        request_options = {"timeout": self.timeout}
        if self._retry is not None:
            request_options["retry"] = self._retry
        response = self.model.generate_content(prompt, request_options=request_options)
        # response.text raises if the model returned no usable candidate
        # (e.g. a safety block or recitation stop) — surface an empty string so
        # the generator's parse-failure handling skips the sample instead of
        # crashing the whole run.
        if not response.candidates or not response.candidates[0].content.parts:
            return ""
        return response.text
