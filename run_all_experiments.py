#!/usr/bin/env python
"""
Run all thesis experiments sequentially.
Usage: python run_all_experiments.py [--epochs 20] [--encoder scratch_resnet]
"""

import subprocess
import argparse
import time
from datetime import datetime

EXPERIMENTS = {
    "temporal_straightening": {
        "name": "temporal_straightening",
        "args": "training.straighten=cos1e-1",
    },
    "second_difference": {
        "name": "second_difference",
        "args": "training.straighten=2nd1e-1",
    },
    "distance_consistency": {
        "name": "distance_consistency",
        "args": "training.straighten=dist1e-1",
    },
    "combined_manual": {
        "name": "combined_manual",
        "args": "training.regularizers.straightening.enabled=true training.regularizers.straightening.lambda=0.1 training.regularizers.second_difference.enabled=true training.regularizers.second_difference.lambda=0.1",
    },
    "moo_upgrad": {
        "name": "moo_upgrad",
        "args": "training.moo.enabled=true training.moo.algorithm=upgrad 'training.moo.objectives=[prediction,straightening,second_difference]'",
    },
    "moo_cagrad": {
        "name": "moo_cagrad",
        "args": "training.moo.enabled=true training.moo.algorithm=cagrad 'training.moo.objectives=[prediction,straightening,second_difference]'",
    },
    "moo_amtl_min": {
        "name": "moo_amtl_min",
        "args": "training.moo.enabled=true training.moo.algorithm=amtl training.moo.amtl_scale_mode=min 'training.moo.objectives=[prediction,straightening,second_difference]'",
    },
    "moo_amtl_median": {
        "name": "moo_amtl_median",
        "args": "training.moo.enabled=true training.moo.algorithm=amtl training.moo.amtl_scale_mode=median 'training.moo.objectives=[prediction,straightening,second_difference]'",
    },
    "moo_amtl_rmse": {
        "name": "moo_amtl_rmse",
        "args": "training.moo.enabled=true training.moo.algorithm=amtl training.moo.amtl_scale_mode=rmse 'training.moo.objectives=[prediction,straightening,second_difference]'",
    },
}


def run_experiment(name, config, base_args):
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config['name']}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    cmd = f"python train.py {base_args} {config['args']} 'hydra.run.dir=checkpoints/{name}'"
    print(f"Command: {cmd}\n")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n{status}: {config['name']} (took {elapsed/60:.1f} minutes)")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all thesis experiments")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--encoder", type=str, default="scratch_resnet", help="Encoder type")
    parser.add_argument("--env", type=str, default="point_maze", help="Environment")
    parser.add_argument("--only", type=str, nargs="+", help="Run only specific experiments")
    parser.add_argument("--skip", type=str, nargs="+", help="Skip specific experiments")
    args = parser.parse_args()

    base_args = f"env={args.env} encoder={args.encoder} training.epochs={args.epochs} training.batch_size={args.batch_size}"

    experiments_to_run = EXPERIMENTS.copy()
    if args.only:
        experiments_to_run = {k: v for k, v in EXPERIMENTS.items() if k in args.only}
    if args.skip:
        experiments_to_run = {k: v for k, v in experiments_to_run.items() if k not in args.skip}

    print(f"\n{'='*60}")
    print("THESIS EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder}")
    print(f"Environment: {args.env}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Experiments to run: {list(experiments_to_run.keys())}")
    print(f"{'='*60}\n")

    results = {}
    total_start = time.time()

    for name, config in experiments_to_run.items():
        success = run_experiment(name, config, base_args)
        results[name] = success
        if success:
            time.sleep(10)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")
    print(f"\nTotal time: {total_time/3600:.1f} hours")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
