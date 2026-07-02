import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ...device import resolve_device
from ..base_model import BaseModel


class Seq2SeqModel(BaseModel):
    """Prefix-prompted encoder-decoder GEC corrector.

    Backs the `t5`, `gec_v1`, and `coedit` model types — they differ only in
    their prompt prefix and weights, both supplied from configs/tasks/gec.json.
    Runs on the device resolved from compute.device (FRAMEWORK_DEVICE).
    """

    def load_model(self, model_config: dict):
        print(f"Loading model: {model_config['name']} ...")
        self.prefix     = model_config.get("prefix", "")
        self.num_beams  = model_config.get("num_beams", 5)
        self.max_length = model_config.get("max_length", 128)
        self.device     = resolve_device()
        self.tokenizer  = AutoTokenizer.from_pretrained(model_config["name"])
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(model_config["name"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {model_config['name']} on {self.device}.")

    def predict(self, sentences: list[str]) -> list[str]:
        results = []
        for sentence in sentences:
            inputs = self.tokenizer.encode(
                self.prefix + sentence,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, num_beams=self.num_beams, max_length=self.max_length
                )
            results.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return results
