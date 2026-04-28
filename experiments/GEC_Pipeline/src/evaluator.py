# src/evaluator.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

class GECEvaluator:
    def __init__(self, model_configs):
        """
        Initializes models based on the configuration file.
        """
        self.models = {}
        for config in model_configs:
            name = config['name']
            m_type = config['type']
            
            print(f"Loading task model: {name}...")
            if m_type == "t5":
                tokenizer = T5Tokenizer.from_pretrained(name)
                model = T5ForConditionalGeneration.from_pretrained(name)
            else:  # Default to standard Seq2Seq for gec_v1
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSeq2SeqLM.from_pretrained(name)
            
            model.eval()
            self.models[name] = {"model": model, "tokenizer": tokenizer, "type": m_type}

    def predict(self, model_name, sentences):
        """Runs batch inference for a specific model."""
        m_data = self.models[model_name]
        results = []
        
        for s in sentences:
            prefix = "grammar: " if m_data['type'] == "t5" else "gec: "
            inputs = m_data['tokenizer'].encode(
                prefix + s, return_tensors="pt", max_length=128, truncation=True
            )
            
            with torch.no_grad():
                outputs = m_data['model'].generate(inputs, max_length=128)
                
            prediction = m_data['tokenizer'].decode(outputs[0], skip_special_tokens=True)
            results.append(prediction)
            
        return results