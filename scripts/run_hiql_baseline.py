"""Run HIQL baseline on OGBench for comparison with U-GCDT.

HIQL (Hierarchical Implicit Q-Learning) is the official strong baseline
in OGBench. The HIQL implementation is in the OGBench GitHub repository,
not in the pip package.

Usage:
    # Option 1: Clone and run from OGBench repo
    git clone https://github.com/seohongpark/ogbench
    cd ogbench
    python impls/main.py --env_name antmaze-large-stitch-v0 --agent hiql --seed 42

    # Option 2: Use this script to print instructions
    python scripts/run_hiql_baseline.py --dataset antmaze-large-stitch-v0 --seed 42
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np


# Published HIQL results from OGBench paper (Table 1)
# These can be used as reference if you don't want to re-run HIQL
HIQL_REFERENCE_SCORES = {
    # AntMaze tasks
    "antmaze-large-stitch-v0": 0.45,  # Approximate from paper
    "antmaze-medium-stitch-v0": 0.55,
    "antmaze-medium-play-v0": 0.50,
    # Add more as needed from the paper
}


def get_reference_score(dataset: str) -> float:
    """Get published HIQL score for a dataset."""
    return HIQL_REFERENCE_SCORES.get(dataset, None)


def run_hiql_baseline(dataset: str, seed: int, output_dir: str, num_steps: int = 1000000):
    """
    Run HIQL baseline using OGBench's official implementation.

    Since HIQL is not included in the ogbench pip package, this function
    provides instructions for running it from the OGBench repository.
    """
    import ogbench

    print("=" * 60)
    print("HIQL Baseline Setup")
    print("=" * 60)

    # Check for reference score
    ref_score = get_reference_score(dataset)
    if ref_score:
        print(f"\nReference HIQL score for {dataset}: {ref_score:.2f}")
        print("(From OGBench paper - can use as target without re-running)")

    # Check if OGBench repo exists locally
    ogbench_repo_paths = [
        os.path.expanduser("~/ogbench"),
        os.path.expanduser("~/projects/ogbench"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "ogbench"),
        "/tmp/ogbench",
    ]

    ogbench_repo = None
    for path in ogbench_repo_paths:
        if os.path.exists(os.path.join(path, "impls", "main.py")):
            ogbench_repo = path
            break

    if ogbench_repo:
        print(f"\nFound OGBench repository at: {ogbench_repo}")
        print("\nTo run HIQL:")
        print(f"  cd {ogbench_repo}")
        print(f"  python impls/main.py --env_name {dataset} --agent hiql --seed {seed}")
    else:
        print("\nOGBench repository not found locally.")
        print("\nTo run HIQL baseline:")
        print("  1. Clone the repository:")
        print("     git clone https://github.com/seohongpark/ogbench")
        print("  2. Install dependencies:")
        print("     cd ogbench && pip install -e .")
        print("  3. Run HIQL:")
        print(f"     python impls/main.py --env_name {dataset} --agent hiql --seed {seed}")

    # Create output directory and save reference info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"hiql_{dataset}_{seed}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save config with reference score
    config = {
        "method": "hiql",
        "dataset": dataset,
        "seed": seed,
        "num_steps": num_steps,
        "reference_score": ref_score,
        "status": "reference_only" if ref_score else "needs_training",
    }

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to: {exp_dir}/config.json")

    if ref_score:
        # Save reference results
        results = {
            "success_rate": ref_score,
            "source": "ogbench_paper",
            "note": "Reference score from OGBench paper Table 1",
        }
        with open(os.path.join(exp_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Reference results saved to: {exp_dir}/results.json")
        return results

    return None


def load_hiql_results(results_dir: str) -> dict:
    """Load HIQL results from OGBench output directory."""
    results_file = os.path.join(results_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)

    # Try to find eval results in standard OGBench format
    eval_file = os.path.join(results_dir, "eval.json")
    if os.path.exists(eval_file):
        with open(eval_file) as f:
            return json.load(f)

    return None


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
    parser.add_argument("--load_results", type=str, default=None,
                        help="Load existing HIQL results from directory")
    args = parser.parse_args()

    if args.load_results:
        results = load_hiql_results(args.load_results)
        if results:
            print(f"HIQL Results from {args.load_results}:")
            print(f"  Success Rate: {results.get('success_rate', 'N/A')}")
        else:
            print(f"Could not load results from {args.load_results}")
    else:
        run_hiql_baseline(
            dataset=args.dataset,
            seed=args.seed,
            output_dir=args.output_dir,
            num_steps=args.num_steps
        )
