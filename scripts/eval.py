"""Corrected evaluation functions with proper normalization - FIXED VERSION.

Key fixes:
1. Observations are normalized during evaluation
2. Goals are normalized during evaluation  
3. Proper handling of action context (last action zeroed)
4. Debug prints for verification
"""

import os
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class NormalizationStats:
    """Statistics for normalizing observations."""
    obs_mean: np.ndarray
    obs_std: np.ndarray
    goal_mean: np.ndarray
    goal_std: np.ndarray


def evaluate_policy_fixed(
    state,  # TrainState with model params
    env,
    config,
    norm_stats: NormalizationStats,
    task_type: str = "antmaze",
    num_episodes: int = 100,
    max_steps: int = 1000,
    num_iterations: Optional[int] = None,  # For UT models
    verbose: bool = False,
    debug_first_episode: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate policy in environment with CORRECT normalization.
    
    CRITICAL FIXES:
    1. Normalize observations using training stats
    2. Normalize goals using training stats
    3. Zero out the last action in context (match training)
    
    Args:
        state: Flax TrainState with model parameters
        env: Gym environment
        config: Config object with training settings
        norm_stats: Normalization statistics from training
        task_type: "antmaze" or "kitchen"
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        num_iterations: Override UT iterations (for test-time scaling)
        verbose: Print progress
        debug_first_episode: Print debug info for first episode
    
    Returns:
        Dictionary with success_rate, avg_steps, etc.
    """
    successes = []
    steps_list = []
    
    ctx_len = config.training.context_len
    action_dim = env.action_space.shape[0]
    
    # Extract normalization stats
    obs_mean = norm_stats.obs_mean
    obs_std = norm_stats.obs_std
    goal_mean = norm_stats.goal_mean
    goal_std = norm_stats.goal_std
    
    if debug_first_episode:
        print("=== Evaluation Debug Info ===")
        print(f"obs_mean shape: {obs_mean.shape}, range: [{obs_mean.min():.2f}, {obs_mean.max():.2f}]")
        print(f"goal_mean shape: {goal_mean.shape}, range: [{goal_mean.min():.2f}, {goal_mean.max():.2f}]")
        print(f"Context length: {ctx_len}")
        print(f"Action dim: {action_dim}")
    
    for ep in range(num_episodes):
        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        # Get goal
        goal_raw = _get_goal(env, obs, task_type)
        
        # CRITICAL FIX: Normalize goal
        goal_normalized = (goal_raw - goal_mean) / goal_std
        
        # CRITICAL FIX: Normalize initial observation
        obs_normalized = (obs - obs_mean) / obs_std
        
        # Debug first episode
        if debug_first_episode and ep == 0:
            print(f"\n--- Episode 0 Debug ---")
            print(f"Raw goal: {goal_raw}")
            print(f"Normalized goal: {goal_normalized}")
            print(f"Raw obs[:5]: {obs[:5]}")
            print(f"Normalized obs[:5]: {obs_normalized[:5]}")
        
        # Context buffers (store NORMALIZED observations)
        states_buffer = [obs_normalized]
        actions_buffer = []
        
        done = False
        steps = 0
        success = False
        
        while not done and steps < max_steps:
            # Prepare context
            states, actions = _prepare_context(
                states_buffer, 
                actions_buffer, 
                ctx_len, 
                action_dim
            )
            
            # Debug first step of first episode
            if debug_first_episode and ep == 0 and steps == 0:
                print(f"States input shape: {states.shape}")
                print(f"Actions input shape: {actions.shape}")
                print(f"Last action in input (should be 0): {actions[-1, :3]}")
                print(f"Goal input shape: {goal_normalized.shape}")
            
            # Forward pass
            states_input = jnp.array(states[None])  # (1, ctx_len, state_dim)
            actions_input = jnp.array(actions[None])  # (1, ctx_len, action_dim)
            goals_input = jnp.array(goal_normalized[None])  # (1, goal_dim)
            timesteps_input = jnp.arange(ctx_len)[None]  # (1, ctx_len)
            
            # Build call kwargs
            call_kwargs = {
                "states": states_input,
                "actions": actions_input,
                "goals": goals_input,
                "timesteps": timesteps_input,
                "deterministic": True,
            }
            
            # Add num_iterations for UT models
            if num_iterations is not None and hasattr(config.model, 'use_weight_tying') and config.model.use_weight_tying:
                call_kwargs["num_iterations"] = num_iterations
            
            outputs = state.apply_fn(state.params, **call_kwargs)
            
            # Get action prediction
            action = np.array(outputs["action_pred"][0])
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            if debug_first_episode and ep == 0 and steps == 0:
                print(f"Predicted action: {action[:3]}")
            
            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            # CRITICAL FIX: Normalize new observation
            obs_normalized = (obs - obs_mean) / obs_std
            
            # Update buffers
            states_buffer.append(obs_normalized)
            actions_buffer.append(action)
            steps += 1
            
            # Check success
            if info.get("success", False):
                success = True
                break
        
        successes.append(float(success))
        steps_list.append(steps)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{num_episodes}: success_rate={np.mean(successes):.3f}, "
                  f"avg_steps={np.mean(steps_list):.1f}")
        
        # Only debug first episode
        debug_first_episode = False
    
    return {
        "success_rate": np.mean(successes),
        "avg_steps": np.mean(steps_list),
        "num_episodes": num_episodes,
        "successes": successes,
    }


def _get_goal(env, obs: np.ndarray, task_type: str) -> np.ndarray:
    """Extract goal from environment."""
    if task_type == "antmaze":
        # Try different ways to get the target goal
        if hasattr(env, 'target_goal'):
            return np.array(env.target_goal)
        elif hasattr(env.unwrapped, 'target_goal'):
            return np.array(env.unwrapped.target_goal)
        elif hasattr(env, 'goal'):
            return np.array(env.goal)
        elif hasattr(env.unwrapped, 'goal'):
            return np.array(env.unwrapped.goal)
        else:
            # Fallback: last 2 dims of observation (some antmaze versions)
            # But this is usually the goal already embedded in obs
            print("WARNING: Could not find target_goal, using obs[-2:]")
            return obs[-2:]
    elif task_type == "kitchen":
        # Kitchen uses full state as goal
        if hasattr(env, 'goal'):
            return np.array(env.goal)
        else:
            return obs.copy()
    else:
        return obs[:2]


def _prepare_context(
    states_buffer: list,
    actions_buffer: list,
    ctx_len: int,
    action_dim: int,
) -> tuple:
    """
    Prepare context for model input.
    
    CRITICAL: Last action in context is ZEROED to match training.
    """
    states_arr = np.array(states_buffer)
    
    # Pad or truncate states
    if len(states_buffer) < ctx_len:
        pad_len = ctx_len - len(states_buffer)
        states = np.concatenate([
            np.tile(states_arr[0:1], (pad_len, 1)),
            states_arr
        ], axis=0)
    else:
        states = states_arr[-ctx_len:]
    
    # Prepare actions with LAST ACTION ZEROED
    if len(actions_buffer) == 0:
        # No actions yet - all zeros
        actions = np.zeros((ctx_len, action_dim))
    elif len(actions_buffer) < ctx_len:
        # Pad with zeros at the beginning
        pad_len = ctx_len - len(actions_buffer) - 1  # -1 because we add a zero at end
        actions_arr = np.array(actions_buffer)
        actions = np.concatenate([
            np.zeros((pad_len, action_dim)),
            actions_arr,
            np.zeros((1, action_dim))  # Zero for current timestep
        ], axis=0)
    else:
        # Take last (ctx_len - 1) actions and add zero
        actions = np.concatenate([
            np.array(actions_buffer[-(ctx_len-1):]),
            np.zeros((1, action_dim))  # Zero for current timestep
        ], axis=0)
    
    return states, actions


def save_norm_stats(norm_stats: NormalizationStats, path: str):
    """Save normalization statistics to file."""
    np.savez(
        path,
        obs_mean=norm_stats.obs_mean,
        obs_std=norm_stats.obs_std,
        goal_mean=norm_stats.goal_mean,
        goal_std=norm_stats.goal_std,
    )
    print(f"Saved normalization stats to {path}")


def load_norm_stats(path: str) -> NormalizationStats:
    """Load normalization statistics from file."""
    data = np.load(path)
    return NormalizationStats(
        obs_mean=data['obs_mean'],
        obs_std=data['obs_std'],
        goal_mean=data['goal_mean'],
        goal_std=data['goal_std'],
    )


# ============================================================================
# INTEGRATION WITH TRAINING SCRIPT
# ============================================================================

def evaluate_policy_wrapper(
    state,
    env,
    config,
    env_info: Dict[str, Any],
    num_episodes: int = 100,
    max_steps: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Wrapper that extracts norm_stats from env_info.
    
    Use this as a drop-in replacement for the original evaluate_policy.
    """
    norm_stats = env_info.get("norm_stats")
    if norm_stats is None:
        raise ValueError("norm_stats not found in env_info! "
                        "Make sure to include it when creating the dataset.")
    
    task_type = env_info.get("task_type", "antmaze")
    
    return evaluate_policy_fixed(
        state=state,
        env=env,
        config=config,
        norm_stats=norm_stats,
        task_type=task_type,
        num_episodes=num_episodes,
        max_steps=max_steps,
        verbose=verbose,
    )


# ============================================================================
# DEBUG/TEST UTILITIES  
# ============================================================================

def test_normalization_consistency(
    env,
    norm_stats: NormalizationStats,
    task_type: str = "antmaze",
    num_tests: int = 5,
):
    """
    Test that normalization is consistent between training and eval.
    
    This helps verify that the goal/obs normalization matches.
    """
    print("=== Testing Normalization Consistency ===")
    
    for i in range(num_tests):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        goal_raw = _get_goal(env, obs, task_type)
        
        # Normalize
        obs_norm = (obs - norm_stats.obs_mean) / norm_stats.obs_std
        goal_norm = (goal_raw - norm_stats.goal_mean) / norm_stats.goal_std
        
        print(f"\nTest {i+1}:")
        print(f"  Raw goal shape: {goal_raw.shape}, values: {goal_raw}")
        print(f"  Normalized goal shape: {goal_norm.shape}, values: {goal_norm}")
        print(f"  Goal norm range: [{goal_norm.min():.2f}, {goal_norm.max():.2f}]")
        print(f"  Obs norm range: [{obs_norm.min():.2f}, {obs_norm.max():.2f}]")
        
        # Check for reasonable normalized values (should be roughly in [-3, 3])
        if np.abs(goal_norm).max() > 10:
            print(f"  WARNING: Goal normalization seems wrong! Max abs value: {np.abs(goal_norm).max():.2f}")
        if np.abs(obs_norm).max() > 10:
            print(f"  WARNING: Obs normalization seems wrong! Max abs value: {np.abs(obs_norm).max():.2f}")


if __name__ == "__main__":
    # Test the evaluation functions
    print("Testing evaluation with normalization fix...")
    
    import gym
    import d4rl
    
    # Create environment and get norm stats
    env = gym.make("antmaze-umaze-v2")
    
    # Simulate loading norm stats (in practice, load from training)
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset["observations"]
    achieved_goals = obs[:, :2]  # xy position for antmaze
    
    norm_stats = NormalizationStats(
        obs_mean=obs.mean(axis=0),
        obs_std=obs.std(axis=0) + 1e-6,
        goal_mean=achieved_goals.mean(axis=0),
        goal_std=achieved_goals.std(axis=0) + 1e-6,
    )
    
    print(f"obs_mean shape: {norm_stats.obs_mean.shape}")
    print(f"goal_mean shape: {norm_stats.goal_mean.shape}")
    
    # Test normalization consistency
    test_normalization_consistency(env, norm_stats, "antmaze")
    
    print("\n✓ Evaluation functions ready to use")