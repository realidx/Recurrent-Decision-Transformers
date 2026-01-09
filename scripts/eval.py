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

from configs.default import Config, ModelConfig, TrainingConfig, DataConfig, AuxiliaryConfig, EvalConfig
from models import create_model


def evaluate_d4rl(
    state,
    env,
    config: Config,
    num_iterations: int = None,
    num_episodes: int = 100,
    max_steps: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate policy on D4RL environment.

    For antmaze: goal is target_goal from env
    For kitchen: uses task completion
    """
    successes = []
    steps_list = []

    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API

        # Get goal for antmaze
        if hasattr(env, 'target_goal'):
            goal = np.array(env.target_goal)
        elif hasattr(env, 'goal'):
            goal = np.array(env.goal)
        else:
            # For kitchen or other envs, use observation as goal placeholder
            goal = np.array(obs[:2] if len(obs) >= 2 else obs)

        # Context buffers
        states_buffer = [obs]
        actions_buffer = []
        done = False
        steps = 0
        success = False

        while not done and steps < max_steps:
            ctx_len = config.training.context_len
            action_dim = env.action_space.shape[0]

            # Prepare context
            if len(states_buffer) < ctx_len:
                pad_len = ctx_len - len(states_buffer)
                states = np.array([states_buffer[0]] * pad_len + list(states_buffer))
                if len(actions_buffer) == 0:
                    actions = np.zeros((ctx_len, action_dim))
                else:
                    pad_actions = [actions_buffer[0]] * pad_len
                    actions = np.array(pad_actions + list(actions_buffer))
                    actions = actions[-ctx_len:]
            else:
                states = np.array(states_buffer[-ctx_len:])
                if len(actions_buffer) >= ctx_len:
                    actions = np.array(actions_buffer[-ctx_len:])
                else:
                    pad_len = ctx_len - len(actions_buffer)
                    actions = np.concatenate([
                        np.zeros((pad_len, action_dim)),
                        np.array(actions_buffer) if actions_buffer else np.zeros((0, action_dim))
                    ], axis=0)

            # Forward pass - ensure correct shapes
            states_input = jnp.array(states[None])  # (1, ctx_len, state_dim)
            actions_input = jnp.array(actions[None])  # (1, ctx_len, action_dim)
            # Ensure goal is 2D with batch dimension
            goal_2d = goal.reshape(1, -1) if goal.ndim == 1 else goal
            goals_input = jnp.array(goal_2d)  # (1, goal_dim)
            timesteps_input = jnp.arange(ctx_len)[None]

            # Build kwargs for model call
            call_kwargs = {
                "states": states_input,
                "actions": actions_input,
                "goals": goals_input,
                "timesteps": timesteps_input,
                "deterministic": True,
            }
            # Only pass num_iterations for UT models (not GCDT baseline)
            if num_iterations is not None and config.model.use_weight_tying:
                call_kwargs["num_iterations"] = num_iterations

            outputs = state.apply_fn(state.params, **call_kwargs)

            action = np.array(outputs["action_pred"][0])
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            states_buffer.append(obs)
            actions_buffer.append(action)
            steps += 1

            # Check success
            if info.get("success", False) or reward > 0:
                success = True
                break

        successes.append(float(success))
        steps_list.append(steps)

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}: success_rate={np.mean(successes):.3f}")

    return {
        "num_iterations": num_iterations,
        "success_rate": np.mean(successes),
        "avg_steps": np.mean(steps_list),
        "num_episodes": num_episodes,
    }


def run_test_time_scaling_experiment(
    checkpoint_path: str,
    config: Config,
    test_iterations: List[int] = [6, 12, 18, 24],
    num_episodes: int = 100,
) -> Dict[str, Any]:
    """
    Run test-time scaling experiment.

    Evaluates same model with different numbers of UT iterations.
    """
    # Convert to absolute path if needed
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load environment based on dataset type
    dataset_type = getattr(config.data, 'dataset_type', 'd4rl')
    dataset_name = config.data.dataset_name

    if dataset_type == "d4rl":
        import gym
        import d4rl
        env = gym.make(dataset_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        # Goal dim for antmaze is 2 (target xy)
        if "antmaze" in dataset_name:
            goal_dim = 2
        else:
            goal_dim = state_dim
    else:
        from data.ogbench_loader import load_ogbench_dataset
        _, _, env_info = load_ogbench_dataset(
            dataset_name=dataset_name,
            dataset_dir=config.data.dataset_dir,
            context_len=config.training.context_len,
        )
        env = env_info["env"]
        state_dim = env_info["state_dim"]
        action_dim = env_info["action_dim"]
        goal_dim = state_dim

    config.state_dim = state_dim
    config.action_dim = action_dim
    config.goal_dim = goal_dim

    print(f"Environment: {dataset_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Goal dim: {goal_dim}")

    # Create model
    model = create_model(config)

    # Load checkpoint
    ckpt = checkpoints.restore_checkpoint(checkpoint_path, target=None)

    # Create a proper TrainState for apply_fn
    from scripts.train import create_train_state
    rng = jax.random.PRNGKey(0)
    dummy_state = create_train_state(
        rng, model, config, state_dim, action_dim, goal_dim
    )
    # Replace params with loaded params
    state = dummy_state.replace(params=ckpt["params"])

    results = {}

    print(f"\nRunning test-time scaling experiment...")
    print(f"Test iterations: {test_iterations}")
    print(f"Model type: {'U-GCDT' if config.model.use_weight_tying else 'GCDT baseline'}")

    for num_iter in test_iterations:
        print(f"\nEvaluating with {num_iter} iterations...")

        eval_result = evaluate_d4rl(
            state,
            env,
            config,
            num_iterations=num_iter,
            num_episodes=num_episodes,
            verbose=True,
        )

        results[f"iter_{num_iter}"] = eval_result
        print(f"  Result: success_rate={eval_result['success_rate']:.3f}, "
              f"avg_steps={eval_result['avg_steps']:.1f}")

    return results


def load_config_from_json(config_path: str) -> Config:
    """Load config from saved JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Reconstruct config from dict
    config = Config()

    # Update model config
    if "model" in config_dict:
        for key, value in config_dict["model"].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    # Update training config
    if "training" in config_dict:
        for key, value in config_dict["training"].items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

    # Update data config
    if "data" in config_dict:
        for key, value in config_dict["data"].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

    # Update aux config
    if "aux" in config_dict:
        for key, value in config_dict["aux"].items():
            if hasattr(config.aux, key):
                setattr(config.aux, key, value)

    # Update eval config
    if "eval" in config_dict:
        for key, value in config_dict["eval"].items():
            if hasattr(config.eval, key):
                setattr(config.eval, key, value)

    # Top level fields
    if "exp_name" in config_dict:
        config.exp_name = config_dict["exp_name"]
    if "output_dir" in config_dict:
        config.output_dir = config_dict["output_dir"]

    return config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate UT-GCDT models with test-time scaling")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to config.json (if not in checkpoint dir)")
    parser.add_argument("--test_iterations", type=int, nargs="+", default=[6, 12, 18, 24],
                       help="Number of iterations to test")
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="Number of episodes for evaluation")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (default: eval_results.json in checkpoint dir)")
    args = parser.parse_args()

    # Find config file
    if args.config_path:
        config_path = args.config_path
    else:
        # Look in checkpoint directory
        if os.path.isdir(args.checkpoint):
            config_path = os.path.join(args.checkpoint, "config.json")
        else:
            # Checkpoint is a file, look in parent directory
            config_path = os.path.join(os.path.dirname(args.checkpoint), "config.json")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please specify --config_path")
        sys.exit(1)

    print(f"Loading config from {config_path}")
    config = load_config_from_json(config_path)

    # Run experiment
    results = run_test_time_scaling_experiment(
        args.checkpoint,
        config,
        test_iterations=args.test_iterations,
        num_episodes=args.num_episodes,
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        if os.path.isdir(args.checkpoint):
            output_path = os.path.join(args.checkpoint, "eval_results.json")
        else:
            output_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    for key, val in results.items():
        print(f"{key}: success_rate={val['success_rate']:.3f}, avg_steps={val['avg_steps']:.1f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()