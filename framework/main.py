# ============================================================
# Main — Entry Point
# ============================================================
# This is the file you run to start the pipeline.
#
# Usage:
#   python main.py
#
# It reads config.yaml and runs the full GET pipeline.
# All parameters (model, dataset, metrics) are in config.yaml
# — no need to touch this file.
# ============================================================

import yaml
from framework.pipeline import run_pipeline


def main():
    # Load configuration from config.yaml
    print("Loading configuration...")
    with open("framework/config.yaml") as f:
        config = yaml.safe_load(f)

    print(f"Task     : {config['task']['name']}")
    print(f"Generator: {config['generation']['model']}")
    print(f"Runs     : {config['generation']['num_runs']}")
    print(f"Models   : {[m['name'] for m in config['task_models']]}")

    # Run the full GET pipeline
    results = run_pipeline(config)

    # Print final results
    print("\n" + "=" * 55)
    print("FINAL RESULTS (mean ± std across runs)")
    print("=" * 55)
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, values in metrics.items():
            if isinstance(values, dict) and "mean" in values:
                print(f"  {metric}: {values['mean']} ± {values['std']}")
            else:
                for sub, v in values.items():
                    print(f"  {metric}.{sub}: {v['mean']} ± {v['std']}")


if __name__ == "__main__":
    main()