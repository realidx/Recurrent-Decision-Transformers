"""Evaluation script for UT-GCDT with test-time scaling experiments."""

import os
import sys
import json
from typing import Dict, List, Any
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import Config
from models import create_model
from data.ogbench_loader import load_ogbench_dataset


def evaluate_with_iterations(
    state,
    env,
    config: Config,
    num_iterations: int,
    num_episodes: int = 100,
    max_steps: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate policy with a specific number of UT iterations.
    
    This enables test-time scaling experiments: train with K iterations,
    evaluate with K' iterations where K' can be different from K.
    """
    results_by_task = {}
    all_successes = []
    all_steps = []
    
    for task_id in range(1, 6):
        task_successes = 0
        task_steps = []
        episodes_per_task = num_episodes // 5
        
        for ep in range(episodes_per_task):
            obs, info = env.reset(options={"task_id": task_id})
            goal = info["goal"]
            
            # Context buffers
            states_buffer = [obs]
            actions_buffer = []
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                ctx_len = config.training.context_len
                
                # Prepare context (same as in train.py)
                if len(states_buffer) < ctx_len:
                    pad_len = ctx_len - len(states_buffer)
                    states = np.array([states_buffer[0]] * pad_len + states_buffer)
                    if len(actions_buffer) == 0:
                        actions = np.zeros((ctx_len, env.action_space.shape[0]))
                    else:
                        pad_actions = [actions_buffer[0]] * pad_len
                        actions = np.array(pad_actions + actions_buffer)
                        actions = actions[-ctx_len:]
                else:
                    states = np.array(states_buffer[-ctx_len:])
                    actions = np.array(actions_buffer[-ctx_len:] if len(actions_buffer) >= ctx_len 
                                      else [np.zeros(env.action_space.shape[0])] * (ctx_len - len(actions_buffer)) + actions_buffer)
                
                # Forward pass with specified number of iterations
                states_input = jnp.array(states[None])
                actions_input = jnp.array(actions[None])
                goals_input = jnp.array(goal[None])
                timesteps_input = jnp.arange(ctx_len)[None]
                
                outputs = state.apply_fn(
                    state.params,
                    states=states_input,
                    actions=actions_input,
                    goals=goals_input,
                    timesteps=timesteps_input,
                    num_iterations=num_iterations,  # Override iterations
                    deterministic=True,
                )
                
                action = np.array(outputs["action_pred"][0])
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                states_buffer.append(obs)
                actions_buffer.append(action)
                steps += 1
                
                if info.get("success", False):
                    task_successes += 1
                    task_steps.append(steps)
                    break
            
            if not info.get("success", False):
                task_steps.append(max_steps)
        
        success_rate = task_successes / episodes_per_task
        results_by_task[f"task_{task_id}"] = {
            "success_rate": success_rate,
            "avg_steps": np.mean(task_steps),
            "num_episodes": episodes_per_task,
        }
        all_successes.append(success_rate)
        all_steps.extend(task_steps)
        
        if verbose:
            print(f"  Task {task_id}: success={success_rate:.3f}, avg_steps={np.mean(task_steps):.1f}")
    
    return {
        "num_iterations": num_iterations,
        "overall_success_rate": np.mean(all_successes),
        "overall_avg_steps": np.mean(all_steps),
        "results_by_task": results_by_task,
    }


def evaluate_goal_distance_buckets(
    state,
    env,
    config: Config,
    num_iterations: int,
    distance_buckets: List[int] = [10, 25, 50, 100],
    episodes_per_bucket: int = 20,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    """
    Evaluate performance stratified by goal distance.
    
    This tests whether UT helps more for longer-horizon goals.
    """
    # Note: This requires custom goal sampling based on distance
    # For now, we use the standard tasks but track the actual distances
    
    results = {}
    
    # Standard evaluation
    for task_id in range(1, 6):
        task_results = []
        
        for ep in range(episodes_per_bucket):
            obs, info = env.reset(options={"task_id": task_id})
            goal = info["goal"]
            
            # Estimate goal distance (L2 for simplicity)
            initial_distance = np.linalg.norm(obs[:2] - goal[:2])  # Assume first 2 dims are position
            
            states_buffer = [obs]
            actions_buffer = []
            done = False
            steps = 0
            success = False
            
            while not done and steps < max_steps:
                ctx_len = config.training.context_len
                
                if len(states_buffer) < ctx_len:
                    pad_len = ctx_len - len(states_buffer)
                    states = np.array([states_buffer[0]] * pad_len + states_buffer)
                    actions = np.zeros((ctx_len, env.action_space.shape[0]))
                    if actions_buffer:
                        actions[-len(actions_buffer):] = np.array(actions_buffer)
                else:
                    states = np.array(states_buffer[-ctx_len:])
                    actions = np.array(actions_buffer[-ctx_len:] if len(actions_buffer) >= ctx_len 
                                      else [np.zeros(env.action_space.shape[0])] * (ctx_len - len(actions_buffer)) + actions_buffer)
                
                states_input = jnp.array(states[None])
                actions_input = jnp.array(actions[None])
                goals_input = jnp.array(goal[None])
                timesteps_input = jnp.arange(ctx_len)[None]
                
                outputs = state.apply_fn(
                    state.params,
                    states=states_input,
                    actions=actions_input,
                    goals=goals_input,
                    timesteps=timesteps_input,
                    num_iterations=num_iterations,
                    deterministic=True,
                )
                
                action = np.array(outputs["action_pred"][0])
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                states_buffer.append(obs)
                actions_buffer.append(action)
                steps += 1
                
                if info.get("success", False):
                    success = True
                    break
            
            task_results.append({
                "initial_distance": initial_distance,
                "success": success,
                "steps": steps,
            })
        
        results[f"task_{task_id}"] = task_results
    
    # Aggregate by distance buckets
    all_results = []
    for task_results in results.values():
        all_results.extend(task_results)
    
    bucket_results = {}
    prev_threshold = 0
    for threshold in distance_buckets:
        bucket_data = [r for r in all_results 
                      if prev_threshold <= r["initial_distance"] < threshold]
        if bucket_data:
            bucket_results[f"dist_{prev_threshold}_{threshold}"] = {
                "success_rate": np.mean([r["success"] for r in bucket_data]),
                "avg_steps": np.mean([r["steps"] for r in bucket_data]),
                "num_episodes": len(bucket_data),
            }
        prev_threshold = threshold
    
    # Handle > max threshold
    bucket_data = [r for r in all_results if r["initial_distance"] >= distance_buckets[-1]]
    if bucket_data:
        bucket_results[f"dist_{distance_buckets[-1]}_plus"] = {
            "success_rate": np.mean([r["success"] for r in bucket_data]),
            "avg_steps": np.mean([r["steps"] for r in bucket_data]),
            "num_episodes": len(bucket_data),
        }
    
    return {
        "num_iterations": num_iterations,
        "bucket_results": bucket_results,
        "raw_results": results,
    }


def run_test_time_scaling_experiment(
    checkpoint_path: str,
    config: Config,
    test_iterations: List[int] = [2, 4, 6, 8],
    num_episodes: int = 100,
) -> Dict[str, Any]:
    """
    Run test-time scaling experiment.
    
    Evaluates same model with different numbers of UT iterations.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load dataset info
    _, _, env_info = load_ogbench_dataset(
        dataset_name=config.data.dataset_name,
        dataset_dir=config.data.dataset_dir,
        context_len=config.training.context_len,
    )
    
    config.state_dim = env_info["state_dim"]
    config.action_dim = env_info["action_dim"]
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    state = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    
    # Create a proper TrainState for apply_fn
    from scripts.train import create_train_state
    rng = jax.random.PRNGKey(0)
    dummy_state = create_train_state(
        rng, model, config, config.state_dim, config.action_dim
    )
    # Replace params with loaded params
    state = dummy_state.replace(params=state["params"])
    
    results = {}
    
    print(f"\nRunning test-time scaling experiment...")
    print(f"Test iterations: {test_iterations}")
    
    for num_iter in test_iterations:
        print(f"\nEvaluating with {num_iter} iterations...")
        
        eval_result = evaluate_with_iterations(
            state,
            env_info["env"],
            config,
            num_iterations=num_iter,
            num_episodes=num_episodes,
            verbose=True,
        )
        
        results[f"iter_{num_iter}"] = eval_result
        print(f"  Overall: success={eval_result['overall_success_rate']:.3f}, "
              f"avg_steps={eval_result['overall_avg_steps']:.1f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to config.json (if not in checkpoint dir)")
    parser.add_argument("--test_iterations", type=int, nargs="+", default=[2, 4, 6, 8],
                       help="Number of iterations to test")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()
    
    # Load config
    config_path = args.config_path or os.path.join(args.checkpoint, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Reconstruct config (simplified - in practice you'd want proper serialization)
    from configs.default import Config
    config = Config()
    # Update config from dict...
    
    # Run experiment
    results = run_test_time_scaling_experiment(
        args.checkpoint,
        config,
        test_iterations=args.test_iterations,
        num_episodes=args.num_episodes,
    )
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()