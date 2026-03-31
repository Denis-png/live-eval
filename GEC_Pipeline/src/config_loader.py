import yaml
# load config
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Usage in main.py
config = load_config()
RUNS = config['pipeline']['num_runs']