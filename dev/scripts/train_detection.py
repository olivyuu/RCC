import os
import yaml
import argparse
import shutil
import datetime
from dev.detection.trainer import train_detection

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to detection YAML config")
    parser.add_argument("--testing", action="store_true", help="Run only 5 epochs for testing pipeline")
    return parser.parse_args()

def make_run_dir(base_dir):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{now}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    args = parse_args()
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise RuntimeError("Config could not be loaded. Check your YAML formatting.")

    # Handle testing flag
    if args.testing:
        print("[INFO] Testing mode: setting epochs=5 and batch_size=8.")
        config['train']['epochs'] = 5
        config['data']['batch_size'] = 8

    base_run_dir = config.get("output_dir", "dev/runs/")
    run_dir = make_run_dir(base_run_dir)
    print(f"[INFO] Outputs/logs will be saved to: {run_dir}")

    # Save config in run folder for reproducibility
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    # Train!
    train_detection(config, run_dir)

if __name__ == "__main__":
    main()
