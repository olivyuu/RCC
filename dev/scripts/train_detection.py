# dev/scripts/train_detection.py

import os
import shutil
from datetime import datetime
import yaml
import torch
from dev.detection.trainer import train_detection

def get_run_dir(base='dev/runs/'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def copy_config(config_path, run_dir):
    config_dst = os.path.join(run_dir, os.path.basename(config_path))
    shutil.copy(config_path, config_dst)
    return config_dst

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dev/config/detection.yaml')
    parser.add_argument('--testing', action='store_true', help='If set, only runs for 5 epochs')
    args = parser.parse_args()

    run_dir = get_run_dir()
    config_dst = copy_config(args.config, run_dir)

    # Load config (so we can pass in the settings, possibly with modifications)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) if f.read().strip() else {}

    # Force settings for test runs
    if args.testing:
        config['epochs'] = 5

    print(f"Output dir for this run: {run_dir}")
    train_detection(config=config, run_dir=run_dir)
