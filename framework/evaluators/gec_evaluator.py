# ============================================================
# GEC Evaluator
# ============================================================
# This evaluator loads and runs GEC task models.
# It supports T5 and Grammar Error Correcter v1 (Gramformer).
#
# How it works:
# 1. Load the model from HuggingFace
# 2. Take corrupted sentences as input
# 3. Return corrected sentences as output
#
# To add a new GEC model, just add a new elif block
# in the load_model method.
# ============================================================

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from .base_evaluator import BaseEvaluator


class GECEvaluator(BaseEvaluator):

    def __init__(self, model_config: dict):
        """
        Initialize and load the GEC model.

        Args:
            model_config: {"name": "...", "type": "t5" or "gec_v1"}
        """
        self.model_name = model_config["name"]
        self.model_type = model_config["type"]
        self.tokenizer  = None
        self.model      = None
        self.load_model(model_config)

    def load_model(self, model_config: dict):
        """Load the correct model based on type."""
        print(f"Loading model: {model_config['name']} ...")

        if model_config["type"] == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(model_config["name"])
            self.model     = T5ForConditionalGeneration.from_pretrained(
                model_config["name"]
            )
            self.prefix = "grammar: "

        elif model_config["type"] == "gec_v1":
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
            self.model     = AutoModelForSeq2SeqLM.from_pretrained(
                model_config["name"]
            )
            self.prefix = "gec: "

        else:
            raise ValueError(
                f"Unsupported model type: {model_config['type']}. "
                f"Supported: t5, gec_v1"
            )

        self.model.eval()
        print(f"✅ {model_config['name']} loaded.")

    def predict(self, sentences: list[str]) -> list[str]:
        """
        Run the GEC model on a list of corrupted sentences.

        Args:
            sentences: list of grammatically incorrect sentences

        Returns:
            list of corrected sentences
        """
        results = []
        for sentence in sentences:
            inputs = self.tokenizer.encode(
                self.prefix + sentence,
                return_tensors="pt",
                max_length=128,
                truncation=True
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    num_beams=5,
                    max_length=128
                )
            corrected = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            results.append(corrected)
        return results