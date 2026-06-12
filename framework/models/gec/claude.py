"""Claude API GEC model (Denis's `haiku` corrector).

Uses the same numbered-batch protocol as `experiments/Denis/summary/evaluation.py`
to amortise the API call cost.
"""
import os
import re
import time

from anthropic import Anthropic, RateLimitError

from ..base_model import BaseModel

_DEFAULT_SYSTEM = (
    "You are a grammatical error corrector. "
    "For each numbered sentence, correct all grammatical errors and return only the "
    "corrected version with the same number. Make minimal changes — fix grammar only, "
    "preserve the original meaning and topic. "
    "Return only the numbered corrected sentences, one per line."
)

_NUMBERED_RE = re.compile(r"^[0-9]+[.)]")


class ClaudeModel(BaseModel):
    """Anthropic Claude as a zero-shot GEC corrector."""

    def load_model(self, model_config: dict):
        self.model_id    = model_config["name"]
        self.batch_size  = model_config.get("batch_size", 5)
        self.max_tokens  = model_config.get("max_tokens", 1024)
        self.temperature = model_config.get("temperature", 0)
        self.system      = model_config.get("system_prompt", _DEFAULT_SYSTEM)
        self.sleep_sec   = model_config.get("sleep_sec", 0)

        api_key = model_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ClaudeModel requires an API key. Set ANTHROPIC_API_KEY or "
                "task_models[*].api_key in config."
            )
        self.client = Anthropic(api_key=api_key)
        print(f"Loaded Claude evaluator: {self.model_id}")

    def predict(self, sentences: list[str]) -> list[str]:
        results = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            results.extend(self._predict_batch(batch))
            if self.sleep_sec:
                time.sleep(self.sleep_sec)
        return results

    def _predict_batch(self, batch: list[str]) -> list[str]:
        user_input = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch))
        for attempt in range(5):
            try:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system,
                    messages=[{"role": "user", "content": user_input}],
                )
                parsed = self._parse_numbered(response.content[0].text, len(batch))
                if parsed is not None:
                    return parsed
            except RateLimitError:
                time.sleep(60 * (2 ** attempt))
            except Exception as e:
                print(f"[WARN] Claude correction failed: {e}")
                time.sleep(10 * (2 ** attempt))
        return ["[CORRECTION FAILED]"] * len(batch)

    @staticmethod
    def _parse_numbered(text: str, expected: int) -> list[str] | None:
        numbered = [
            re.sub(r"^[0-9]+[.)\s]+", "", l).strip()
            for l in text.strip().splitlines()
            if _NUMBERED_RE.match(l.strip())
        ]
        if len(numbered) == expected:
            return numbered
        plain = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(plain) == expected:
            return plain
        return None
