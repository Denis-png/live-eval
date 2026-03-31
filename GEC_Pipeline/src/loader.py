from datasets import load_dataset
import random

def get_seed_sentences(dataset_name="agentlans/grammar-correction", split="train", count=50):
    try:
        # streaming=True is great for large datasets!
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        seeds = []
        for i, example in enumerate(dataset):
            if i >= count:
                break
            
            # Logic to handle different dataset schemas
            if 'text' in example:
                seeds.append(example['text'])
            elif 'correct' in example: # Fallback for other datasets
                seeds.append(example['correct'])
            elif 'target' in example:  # Common in T5 datasets
                seeds.append(example['target'])
            else:
                # If we don't know the key, just take the first value 
                # (Risky, but better than failing)
                seeds.append(next(iter(example.values())))
                
        return seeds
    except Exception as e:
        print(f"Error loading seeds: {e}")
        return ["The quick brown fox jumps over the lazy dog."]