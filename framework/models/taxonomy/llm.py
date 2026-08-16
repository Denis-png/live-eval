"""LLM task model for Taxonomy Induction / Subclass Axiom Induction."""

from __future__ import annotations

import json
from typing import Any

from framework.evaluators.taxonomy.metrics import parse_prediction_relations
from framework.models.base_model import BaseModel

DEFAULT_PROMPT_TEMPLATE = """\
You are given a domain and a list of ontology classes.

Infer only direct subclass relationships between the supplied classes.

Return strict JSON:

{{
  "subclass_axioms": [
    ["ChildClass", "ParentClass"]
  ]
}}

Rules:
- use only class identifiers from the provided list
- first item is the child
- second item is the direct parent
- do not invent classes
- do not output transitive relations unless they are believed to be direct
- return JSON only

Domain:
{domain}

Classes:
{classes_json}
"""


class TaxonomyLLMModel(BaseModel):
    """Provider-agnostic LLM wrapper for predicting direct subclass axioms.

    The model receives the serialized taxonomy eval input created by
    TaxonomyTask. That input contains only {"domain", "classes"}; gold
    subclass axioms and metadata are never part of the prompt.
    """

    def load_model(self, model_config: dict):
        from framework.pipeline import load_generator

        self.model_name = model_config["name"]
        self.prompt_template = model_config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        generator_config = {
            **model_config,
            "model": model_config.get("model", model_config["name"]),
            "temperature": model_config.get("temperature", 0),
        }
        self.generator = load_generator(generator_config)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict subclass relations for each serialized taxonomy input."""
        return [self._predict_one(text) for text in texts]

    def build_prompt(self, text: str) -> str:
        """Build a prompt from model-visible taxonomy input only."""
        payload = self._parse_model_input(text)
        classes = payload["classes"]
        return self.prompt_template.format(
            domain=payload["domain"],
            classes_json=json.dumps(classes, ensure_ascii=False, indent=2),
        )

    def _predict_one(self, text: str) -> dict[str, Any]:
        try:
            payload = self._parse_model_input(text)
            raw_output = self.generator.call_api(self.build_prompt(text))
            parsed = parse_prediction_relations(raw_output, payload["classes"])
            return self._prediction_payload(raw_output, parsed)
        except Exception as exc:
            return {
                "subclass_axioms": [],
                "raw_output": "",
                "diagnostics": {
                    "malformed": True,
                    "error": str(exc),
                    "invalid_relation_count": 0,
                    "unknown_class_relation_count": 0,
                    "malformed_relation_count": 0,
                },
            }

    @staticmethod
    def _parse_model_input(text: str) -> dict[str, Any]:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Taxonomy model input must be a JSON object.")
        domain = payload.get("domain")
        classes = payload.get("classes")
        if not isinstance(domain, str) or not isinstance(classes, list):
            raise ValueError("Taxonomy model input must contain domain and classes.")
        normalized_classes = [item for item in classes if isinstance(item, str) and item.strip()]
        return {"domain": domain, "classes": normalized_classes}

    @staticmethod
    def _prediction_payload(raw_output: str, parsed: dict[str, Any]) -> dict[str, Any]:
        relations = sorted(parsed["relations"])
        diagnostics = {
            "malformed": parsed["malformed"],
            "invalid_relation_count": parsed["invalid_relation_count"],
            "invalid_relation_rate": parsed["invalid_relation_rate"],
            "unknown_class_relation_count": parsed["unknown_class_relation_count"],
            "malformed_relation_count": parsed["malformed_relation_count"],
        }
        return {
            "subclass_axioms": [[child, parent] for child, parent in relations],
            "raw_output": raw_output,
            "diagnostics": diagnostics,
        }
