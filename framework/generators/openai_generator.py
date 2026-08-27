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

_REFUSAL_PREVIEW_LIMIT = 500


def _get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_len(value):
    try:
        return len(value)
    except TypeError:
        return None


def _safe_type_names(values):
    if not isinstance(values, list):
        return None
    names = []
    for value in values:
        if isinstance(value, dict):
            names.append(str(value.get("type", "dict")))
        else:
            names.append(type(value).__name__)
    return names


class OpenAIGenerator(BaseGenerator):
    """
    Generator for all OpenAI-compatible providers:
    openai, groq, openrouter, mistral.
    Set provider in config.yaml; the correct base_url is resolved automatically.
    """

    def __init__(self, config: dict):
        self.model       = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens  = config.get("max_tokens")
        self.last_response_diagnostic = None
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=_BASE_URLS.get(config["provider"]),  # None → default OpenAI endpoint
            timeout=config.get("timeout", 300),       # seconds; SDK default is 600
            max_retries=config.get("max_retries", 1), # SDK default is 2
        )

    def call_api(self, prompt: str) -> str | None:
        self.last_response_diagnostic = None
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        content = message.content
        if not isinstance(content, str):
            self.last_response_diagnostic = self._response_diagnostic(response, choice, message)
        return content

    def _response_diagnostic(self, response, choice, message) -> dict:
        """Safe shape-only metadata for empty/non-text provider responses."""
        diagnostic = {}
        finish_reason = _get_field(choice, "finish_reason")
        if finish_reason is not None:
            diagnostic["finish_reason"] = finish_reason
        response_model = _get_field(response, "model")
        if response_model is not None:
            diagnostic["response_model"] = response_model
        response_id = _get_field(response, "id")
        if response_id is not None:
            diagnostic["response_id"] = response_id

        usage = _get_field(response, "usage")
        if usage is not None:
            usage_diagnostic = {}
            for src, dst in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                value = _get_field(usage, src)
                if value is not None:
                    usage_diagnostic[dst] = value
            if usage_diagnostic:
                diagnostic["usage"] = usage_diagnostic

        reasoning = _get_field(message, "reasoning")
        if reasoning is not None:
            diagnostic["reasoning_present"] = True
            length = _safe_len(reasoning)
            if length is not None:
                diagnostic["reasoning_length"] = length

        reasoning_details = _get_field(message, "reasoning_details")
        if reasoning_details is not None:
            diagnostic["reasoning_details_present"] = True
            count = _safe_len(reasoning_details)
            if count is not None:
                diagnostic["reasoning_details_count"] = count
            type_names = _safe_type_names(reasoning_details)
            if type_names is not None:
                diagnostic["reasoning_details_types"] = type_names

        refusal = _get_field(message, "refusal")
        if refusal is not None:
            diagnostic["refusal_present"] = True
            length = _safe_len(refusal)
            if length is not None:
                diagnostic["refusal_length"] = length
            if isinstance(refusal, str):
                diagnostic["refusal_preview"] = refusal[:_REFUSAL_PREVIEW_LIMIT]

        tool_calls = _get_field(message, "tool_calls")
        if tool_calls is not None:
            count = _safe_len(tool_calls)
            if count is not None:
                diagnostic["tool_calls_count"] = count

        function_call = _get_field(message, "function_call")
        if function_call is not None:
            diagnostic["function_call_present"] = True

        native_finish_reason = _get_field(choice, "native_finish_reason")
        if native_finish_reason is not None:
            diagnostic["native_finish_reason"] = native_finish_reason

        return diagnostic
