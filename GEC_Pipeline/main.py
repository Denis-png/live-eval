import os
import json
import numpy as np
from dotenv import load_dotenv
from src.config_loader import load_config
from src.loader import get_seed_sentences
from src.generator import GECGenerator
from src.verifier import verify_generation
from src.evaluator import GECEvaluator
from src.metrics import calculate_gleu, calculate_errant

# 1. Initialization
load_dotenv()
config = load_config()

def run_pipeline():
    # Load seeds once
    seeds = get_seed_sentences(
        dataset_name=config['dataset']['name'],
        count=config['dataset']['seed_size']
    )
    
    generator = GECGenerator(
        model_name=config['generation']['model'],
        temperature=config['generation']['temperature']
    )
    
    evaluator = GECEvaluator(config['task_models'])
    
    # Storage for stability analysis
    all_runs_results = {model['name']: {"gleu": [], "f0.5": []} for model in config['task_models']}

    # 2. Stability Loop (N Runs)
    for run_idx in range(config['pipeline']['num_runs']):
        print(f"\n--- RUN {run_idx + 1} / {config['pipeline']['num_runs']} ---")
        
        # GENERATE & VERIFY
        synthetic_batch = []
        for seed in seeds[:config['pipeline']['sample_size']]:
            import random
            err_type = random.choice(config['generation']['error_types'])
            corrupted = generator.corrupt_sentence(seed, err_type)
            
            if verify_generation(seed, corrupted):
                synthetic_batch.append({"original": seed, "corrupted": corrupted})

        # EVALUATE
        for model_config in config['task_models']:
            name = model_config['name']
            corrupted_inputs = [item['corrupted'] for item in synthetic_batch]
            ground_truth = [item['original'] for item in synthetic_batch]
            
            predictions = evaluator.predict(name, corrupted_inputs)
            
            # Score
            gleu = calculate_gleu(ground_truth, predictions)
            errant_stats = calculate_errant(corrupted_inputs, ground_truth, predictions)
            
            all_runs_results[name]["gleu"].append(gleu)
            all_runs_results[name]["f0.5"].append(errant_stats['f0.5'])
            
            print(f"Model: {name} | GLEU: {gleu} | F0.5: {errant_stats['f0.5']}")

        # TRASH (Clear batch to ensure fresh start next run)
        synthetic_batch.clear()

    # 3. Final Aggregation (Mean ± Std)
    print("\n" + "="*30 + "\nFINAL STABILITY REPORT\n" + "="*30)
    for name, metrics in all_runs_results.items():
        print(f"\n{name}:")
        print(f"  GLEU: {np.mean(metrics['gleu']):.4f} ± {np.std(metrics['gleu']):.4f}")
        print(f"  F0.5: {np.mean(metrics['f0.5']):.4f} ± {np.std(metrics['f0.5']):.4f}")

if __name__ == "__main__":
    run_pipeline()