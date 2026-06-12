import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from ..base_model import BaseModel


class T5Model(BaseModel):

    def load_model(self, model_config: dict):
        print(f"Loading model: {model_config['name']} ...")
        self.prefix = model_config.get("prefix", "grammar: ")
        self.num_beams = model_config.get("num_beams", 5)
        self.max_length = model_config.get("max_length", 128)
        self.tokenizer = T5Tokenizer.from_pretrained(model_config["name"])
        self.model = T5ForConditionalGeneration.from_pretrained(model_config["name"])
        self.model.eval()
        print(f"Loaded {model_config['name']}.")

    def predict(self, sentences: list[str]) -> list[str]:
        results = []
        for sentence in sentences:
            inputs = self.tokenizer.encode(
                self.prefix + sentence,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, num_beams=self.num_beams, max_length=self.max_length
                )
            results.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return results
