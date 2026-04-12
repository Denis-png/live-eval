# ============================================================
# LLM Generator
# ============================================================
# This generator uses an LLM (via Groq or OpenAI API) to
# create synthetic corrupted sentences from real data.
#
# The user can configure:
#   - provider: "groq" or "openai"
#   - model: any model available on that provider
#   - temperature: controls randomness (paper recommends > 0)
#
# This is the default generator in the framework.
# ============================================================

import random
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base_generator import BaseGenerator


class LLMGenerator(BaseGenerator):

    def __init__(self, config: dict):
        """
        Initialize the LLM generator with config settings.

        Args:
            config: the generation section of config.yaml
        """
        self.config = config

        # Initialize the LLM based on provider
        if config["provider"] == "groq":
            self.llm = ChatGroq(
                model=config["model"],
                temperature=config["temperature"],
                api_key=config["api_key"]
            )
        else:
            raise ValueError(
                f"Unsupported provider: {config['provider']}. "
                f"Supported: groq, openai"
            )

    def generate(
        self,
        real_samples: list[dict],
        error_types: list[str],
        prompt_instruction: str,
        sample_size: int,
    ) -> list[dict]:
        """
        Generate synthetic corrupted sentences using the LLM.

        For each real sentence:
        1. Pick a random error type
        2. Ask the LLM to introduce that error
        3. Store original (ground truth) + corrupted (model input)
        """
        # Build the prompt chain
        prompt = PromptTemplate.from_template(prompt_instruction)
        chain = prompt | self.llm | StrOutputParser()

        synthetic = []
        samples = real_samples[:sample_size]

        for item in samples:
            error_type = random.choice(error_types)
            try:
                corrupted = chain.invoke({
                    "sentence":   item["correct"],
                    "error_type": error_type
                }).strip()

                # Skip if LLM returned the same sentence unchanged
                if corrupted == item["correct"].strip():
                    print(f"[SKIP] LLM returned unchanged sentence.")
                    continue

                synthetic.append({
                    "original":   item["correct"],
                    "corrupted":  corrupted,
                    "error_type": error_type
                })

            except Exception as e:
                print(f"[WARN] Generation failed: {e}")

        print(f"Generated {len(synthetic)} synthetic samples.")
        return synthetic