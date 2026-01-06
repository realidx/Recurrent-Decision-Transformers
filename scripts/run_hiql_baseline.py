"""Run HIQL baseline on OGBench for comparison with U-GCDT.

HIQL (Hierarchical Implicit Q-Learning) is the official strong baseline
in OGBench. We run it to establish the target performance for our methods.

Usage:
    python scripts/run_hiql_baseline.py --dataset antmaze-large-stitch-v0 --seed 42
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np


def run_hiql_baseline(dataset: str, seed: int, output_dir: str, num_steps: int = 1000000):
    """
    Run HIQL baseline using OGBench's official implementation.

    Args:
        dataset: OGBench dataset name (e.g., 'antmaze-large-stitch-v0')
        seed: Random seed
        output_dir: Directory to save results
        num_steps: Number of training steps
    """
    try:
        import ogbench
        from ogbench.manipspace_benchmark.train_hiql import train_hiql
    except ImportError:
        print("Error: OGBench not installed or HIQL not available.")
        print("Install with: pip install ogbench")
        print("\nAlternatively, run HIQL using the official OGBench scripts:")
        print(f"  cd <ogbench_repo> && python train_hiql.py --env {dataset} --seed {seed}")
        return None

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"hiql_{dataset}_{seed}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"Running HIQL Baseline")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")
    print(f"Output: {exp_dir}")
    print(f"=" * 60)

    # Load environment and data
    env, train_data, val_data = ogbench.make_env_and_datasets(dataset)

    # HIQL configuration (OGBench defaults)
    hiql_config = {
        "hidden_dims": (256, 256),
        "discount": 0.99,
        "expectile": 0.9,
        "temperature": 1.0,
        "way_steps": 25,
        "actor_lr": 3e-4,
        "value_lr": 3e-4,
        "critic_lr": 3e-4,
        "batch_size": 256,
    }

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({
            "method": "hiql",
            "dataset": dataset,
            "seed": seed,
            "num_steps": num_steps,
            **hiql_config
        }, f, indent=2)

    # Train HIQL
    try:
        results = train_hiql(
            env=env,
            train_data=train_data,
            val_data=val_data,
            seed=seed,
            num_steps=num_steps,
            eval_every=10000,
            **hiql_config
        )
    except Exception as e:
        print(f"HIQL training failed with official API: {e}")
        print("\nFalling back to command-line execution...")
        return run_hiql_cli(dataset, seed, output_dir, num_steps)

    # Save results
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nHIQL training complete!")
    print(f"Final success rate: {results.get('final_success_rate', 'N/A')}")
    print(f"Results saved to: {exp_dir}")

    return results


def run_hiql_cli(dataset: str, seed: int, output_dir: str, num_steps: int = 1000000):
    """
    Run HIQL using OGBench command-line interface.

    This is a fallback if the Python API is not available.
    """
    import subprocess

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"hiql_{dataset}_{seed}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Try to find ogbench installation
    try:
        import ogbench
        ogbench_path = os.path.dirname(ogbench.__file__)
    except ImportError:
        print("Error: OGBench not installed. Please install with: pip install ogbench")
        return None

    # Construct command
    cmd = [
        sys.executable, "-m", "ogbench.train",
        "--algo", "hiql",
        "--env", dataset,
        "--seed", str(seed),
        "--max_steps", str(num_steps),
        "--log_dir", exp_dir,
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"HIQL failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Could not find OGBench training script.")
        print("\nManual instructions:")
        print(f"1. Clone OGBench: git clone https://github.com/seohongpark/ogbench")
        print(f"2. Install: pip install -e ogbench")
        print(f"3. Run: python -m ogbench.train --algo hiql --env {dataset} --seed {seed}")
        return None

    return {"exp_dir": exp_dir}


def evaluate_hiql_checkpoint(checkpoint_path: str, dataset: str, num_episodes: int = 100):
    """
    Evaluate a trained HIQL checkpoint.

    Args:
        checkpoint_path: Path to HIQL checkpoint
        dataset: OGBench dataset name
        num_episodes: Number of evaluation episodes
    """
    import ogbench

    env, _, _ = ogbench.make_env_and_datasets(dataset)

    # Load HIQL agent
    try:
        from ogbench.algorithms.hiql import HIQL
        agent = HIQL.load(checkpoint_path)
    except Exception as e:
        print(f"Could not load HIQL checkpoint: {e}")
        return None

    successes = []
    for task_id in range(1, 6):  # 5 evaluation tasks
        task_successes = 0
        for _ in range(num_episodes // 5):
            obs, info = env.reset(options={"task_id": task_id})
            goal = info["goal"]
            done = False
            steps = 0

            while not done and steps < 1000:
                action = agent.sample_action(obs, goal)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                if info.get("success", False):
                    task_successes += 1
                    break

        successes.append(task_successes / (num_episodes // 5))

    return {
        "success_rate": np.mean(successes),
        "success_per_task": successes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HIQL baseline on OGBench")
    parser.add_argument("--dataset", type=str, default="antmaze-large-stitch-v0",
                        help="OGBench dataset name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs/baselines",
                        help="Output directory for results")
    parser.add_argument("--num_steps", type=int, default=1000000,
                        help="Number of training steps")
    parser.add_argument("--eval_checkpoint", type=str, default=None,
                        help="Evaluate existing checkpoint instead of training")
    args = parser.parse_args()

    if args.eval_checkpoint:
        results = evaluate_hiql_checkpoint(
            args.eval_checkpoint,
            args.dataset,
            num_episodes=100
        )
        if results:
            print(f"HIQL Evaluation Results:")
            print(f"  Success Rate: {results['success_rate']:.3f}")
            print(f"  Per-task: {results['success_per_task']}")
    else:
        run_hiql_baseline(
            dataset=args.dataset,
            seed=args.seed,
            output_dir=args.output_dir,
            num_steps=args.num_steps
        )
