import google.generativeai as genai
from .base_generator import BaseGenerator


class GoogleGenerator(BaseGenerator):
    """Generator for Google AI models (gemini-*)."""

    def __init__(self, config: dict):
        self.temperature = config["temperature"]
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(
            model_name=config["model"],
            generation_config=genai.GenerationConfig(temperature=self.temperature),
        )

    def _call_api(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text
