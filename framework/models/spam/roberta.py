import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..base_model import BaseModel


class RobertaSpamModel(BaseModel):

    def load_model(self, model_config: dict):
        print(f"Loading model: {model_config['name']} ...")
        self.max_length = model_config.get("max_length", 128)
        self.label_map = model_config.get("label_map", {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_config["name"])
        self.model.eval()
        print(f"Loaded {model_config['name']}.")

    def predict(self, sentences: list[str]) -> list[str]:
        results = []
        for sentence in sentences:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            label_id = torch.argmax(logits, dim=-1).item()
            label = self.model.config.id2label[label_id]
            results.append(self.label_map.get(label, label))
        return results
