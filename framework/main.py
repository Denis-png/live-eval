import argparse
import yaml
from framework.pipeline import run_pipeline

DEFAULT_CONFIG = "framework/configs/config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="GET Framework — Generate, Evaluate, Trash",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",       default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--task",         help="Task name (e.g. gec)")
    parser.add_argument("--provider",     help="Generator provider (groq|openai)")
    parser.add_argument("--model",        help="Generator model name")
    parser.add_argument("--runs",         type=int, help="Number of GET runs")
    parser.add_argument("--sample-size",  type=int, dest="sample_size", help="Synthetic samples per run")
    return parser.parse_args()


def apply_overrides(config: dict, args) -> dict:
    """Apply CLI flags on top of YAML config. Only overrides keys that were explicitly passed."""
    if args.task:
        config["task"]["name"] = args.task
    if args.provider:
        config["generation"]["provider"] = args.provider
    if args.model:
        config["generation"]["model"] = args.model
    if args.runs:
        config["generation"]["num_runs"] = args.runs
    if args.sample_size:
        config["generation"]["sample_size"] = args.sample_size
    return config


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = apply_overrides(config, args)

    print(f"Task     : {config['task']['name']}")
    print(f"Provider : {config['generation']['provider']}")
    print(f"Model    : {config['generation']['model']}")
    print(f"Runs     : {config['generation']['num_runs']}")
    print(f"Models   : {[m['name'] for m in config['task_models']]}")

    results = run_pipeline(config)

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
