"""Training script for UT-GCDT on D4RL/OGBench with resume support."""

import os
import sys
import time
import json
import dataclasses
from datetime import datetime
from functools import partial
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
import flax
from flax.training import train_state, checkpoints
from flax.jax_utils import replicate, unreplicate
import optax
import numpy as np

# Device setup
NUM_DEVICES = jax.local_device_count()
print(f"JAX devices: {jax.devices()}")
print(f"Number of devices: {NUM_DEVICES}")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import Config, get_ut_gcdt_full_config
from models import create_model, UTGCDT
from data.ogbench_loader import load_ogbench_dataset
from data.ogbench_loader import DataLoader as OGBenchDataLoader
from data.ogbench_loader import batch_to_jax as ogbench_batch_to_jax
from data.d4rl_loader import load_d4rl_dataset
from data.d4rl_loader import DataLoader as D4RLDataLoader
from data.d4rl_loader import batch_to_jax as d4rl_batch_to_jax


class TrainState(train_state.TrainState):
    """Extended train state with additional fields."""
    pass


def create_train_state(
    rng: jax.random.PRNGKey,
    model: UTGCDT,
    config: Config,
    state_dim: int,
    action_dim: int,
    goal_dim: int = None,
) -> TrainState:
    """Initialize model and optimizer."""
    # Goal dim defaults to state_dim if not specified
    if goal_dim is None:
        goal_dim = state_dim

    # Create dummy inputs for initialization
    batch_size = 2
    seq_len = config.training.context_len

    dummy_states = jnp.zeros((batch_size, seq_len, state_dim))
    dummy_actions = jnp.zeros((batch_size, seq_len, action_dim))
    dummy_goals = jnp.zeros((batch_size, goal_dim))
    dummy_timesteps = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Initialize parameters
    params = model.init(
        rng,
        states=dummy_states,
        actions=dummy_actions,
        goals=dummy_goals,
        timesteps=dummy_timesteps,
        deterministic=True,
        return_intermediates=config.aux.deep_supervision,
    )
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    
    # Create optimizer with warmup
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.training.learning_rate,
        transition_steps=config.training.warmup_steps,
    )
    
    decay_fn = optax.cosine_decay_schedule(
        init_value=config.training.learning_rate,
        decay_steps=config.training.max_steps - config.training.warmup_steps,
    )
    
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.training.warmup_steps],
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule_fn,
            weight_decay=config.training.weight_decay,
        ),
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def load_checkpoint(
    checkpoint_dir: str,
    state: TrainState,
    prefix: str = "checkpoint_",
) -> Tuple[TrainState, int]:
    """
    Load the latest checkpoint from directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        state: Initial TrainState (for structure)
        prefix: Checkpoint file prefix
        
    Returns:
        Restored TrainState and step number
    """
    restored_state = checkpoints.restore_checkpoint(
        checkpoint_dir,
        target=state,
        prefix=prefix,
    )
    
    # Get step from restored state
    step = int(restored_state.step)
    
    return restored_state, step


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find max
    steps = []
    for f in checkpoint_files:
        try:
            step = int(f.split("_")[1])
            steps.append(step)
        except (IndexError, ValueError):
            continue
    
    if not steps:
        return None
    
    return checkpoint_dir


def compute_loss(
    params: Dict,
    apply_fn,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
    gc_reg_weight: float,
    gc_margin: float,
    directional_weight: float,
    directional_margin: float,
    rng: jax.random.PRNGKey,
    deterministic: bool = False,
    step: int = 0,  # NEW: current training step for adaptive weighting
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute training loss with adaptive goal-conditioning.
    
    Key fix: Adaptive GC weight that increases when goal-conditioning collapses.
    """
    
    # Forward pass with original goals
    outputs = apply_fn(
        params,
        states=batch["states"],
        actions=batch["actions"],
        goals=batch["goals"],
        timesteps=batch["timesteps"],
        deterministic=deterministic,
        return_intermediates=aux_deep_supervision,
        rngs={"dropout": rng} if rng is not None else None,
    )
    
    action_pred = outputs["action_pred"]
    target_actions = batch["target_actions"]
    action_loss = jnp.mean((action_pred - target_actions) ** 2)

    metrics = {"action_loss": action_loss}

    # Deep supervision loss
    if "intermediate_action_preds" in outputs and aux_deep_supervision:
        preds = outputs["intermediate_action_preds"]
        per_step = jnp.stack([jnp.mean((p - target_actions) ** 2) for p in preds], axis=0)
        deep_action_loss = jnp.mean(per_step)
        deep_weight = 0.1
        total_loss = action_loss + deep_weight * deep_action_loss

        metrics["deep_action_loss"] = deep_action_loss
        metrics["action_loss"] = jnp.mean((preds[-1] - target_actions) ** 2)
    else:
        total_loss = action_loss

    # Waypoint loss with directional component
    if aux_use_waypoint_loss and "waypoint_preds" in outputs and len(outputs["waypoint_preds"]) > 0:
        waypoint_preds = outputs["waypoint_preds"]
        future_states = batch["future_states"]
        
        if aux_deep_supervision and len(waypoint_preds) > 1:
            waypoint_loss = jnp.mean(jnp.stack([
                jnp.mean((wp_pred - future_states) ** 2) 
                for wp_pred in waypoint_preds
            ]))
        else:
            waypoint_loss = jnp.mean((waypoint_preds[-1] - future_states) ** 2)
        
        waypoint_loss = waypoint_loss_weight * waypoint_loss
        metrics["waypoint_loss"] = waypoint_loss
        total_loss = total_loss + waypoint_loss
        
        # Directional waypoint loss (weight can be zeroed for OGBench)
        current_xy = batch["states"][:, -1, :2]
        goal_xy = batch["goals"][:, :2]
        waypoint_pred_xy = waypoint_preds[-1][:, :2]

        pred_delta = waypoint_pred_xy - current_xy
        desired_delta = goal_xy - current_xy

        pred_delta_norm = pred_delta / (jnp.linalg.norm(pred_delta, axis=-1, keepdims=True) + 1e-8)
        desired_delta_norm = desired_delta / (jnp.linalg.norm(desired_delta, axis=-1, keepdims=True) + 1e-8)

        cos_sim = jnp.sum(pred_delta_norm * desired_delta_norm, axis=-1)

        directional_loss = jnp.mean(jnp.maximum(0.0, directional_margin - cos_sim))

        metrics["directional_loss"] = directional_loss
        metrics["waypoint_cos_sim"] = jnp.mean(cos_sim)
        total_loss = total_loss + directional_weight * directional_loss

    # ============================================================
    # Goal-Conditioning Regularizer with Adaptive Weighting
    # ============================================================
    goals_shuffled = jnp.roll(batch["goals"], shift=1, axis=0)
    
    outputs_g2 = apply_fn(
        params,
        states=batch["states"],
        actions=batch["actions"],
        goals=goals_shuffled,
        timesteps=batch["timesteps"],
        deterministic=True,
        return_intermediates=False,
        rngs=None,
    )
    
    action_pred_g1 = outputs["action_pred"]
    action_pred_g2 = outputs_g2["action_pred"]
    
    action_diff = jnp.linalg.norm(action_pred_g1 - action_pred_g2, axis=-1)
    
    gc_loss_per_sample = jnp.maximum(0.0, gc_margin - action_diff)
    gc_loss = jnp.mean(gc_loss_per_sample)
    
    metrics["gc_loss"] = gc_loss
    metrics["gc_action_diff"] = jnp.mean(action_diff)
    
    # ============================================================
    # FIX 4: Adaptive GC weight
    # Increase weight when gc_action_diff is low (model ignoring goal)
    # ============================================================
    if rng is not None:
        # Base weight + adaptive component
        # When action_diff < margin, increase weight
        adaptive_multiplier = jnp.where(
            jnp.mean(action_diff) < gc_margin * 0.5,  # If diff < half the margin
            2.0,  # Double the weight
            1.0   # Normal weight
        )
        effective_gc_weight = gc_reg_weight * adaptive_multiplier
        
        # Also ramp up GC weight over training (schedule)
        # Start with 0.5x, reach 1.0x at step 10000
        warmup_factor = jnp.minimum(1.0, step / 10000.0)
        warmup_factor = jnp.maximum(0.5, warmup_factor)
        
        effective_gc_weight = effective_gc_weight * warmup_factor
        
        metrics["effective_gc_weight"] = effective_gc_weight
        total_loss = total_loss + effective_gc_weight * gc_loss
    
    metrics["total_loss"] = total_loss
    
    return total_loss, metrics


def compute_loss_wrapper(params, apply_fn, batch, aux_use_waypoint_loss, 
                         aux_deep_supervision, waypoint_loss_weight,
                         gc_reg_weight, gc_margin, directional_weight,
                         directional_margin, rng, deterministic, step):
    """Wrapper that passes step to compute_loss."""
    return compute_loss(
        params, apply_fn, batch, aux_use_waypoint_loss,
        aux_deep_supervision, waypoint_loss_weight,
        gc_reg_weight, gc_margin, directional_weight, directional_margin,
        rng, deterministic, step
    )


@partial(jax.jit, static_argnums=(2, 3, 4))
def train_step_single(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
    gc_reg_weight: float,      # New
    gc_margin: float,          # New
    directional_weight: float,
    directional_margin: float,
    rng: jax.random.PRNGKey,
    step: int,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step (single device)."""

    def loss_fn(params):
        return compute_loss(
            params,
            state.apply_fn,
            batch,
            aux_use_waypoint_loss,
            aux_deep_supervision,
            waypoint_loss_weight,
            gc_reg_weight,      # Add this
            gc_margin,          # Add this
            directional_weight,
            directional_margin,
            rng,
            deterministic=False,
            step=step,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, metrics


def train_step_parallel(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
    gc_reg_weight: float,      # Add this
    gc_margin: float,          # Add this
    directional_weight: float,
    directional_margin: float,
    rng: jax.random.PRNGKey,
    step: int,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step with gradient sync across devices."""

    def loss_fn(params):
        return compute_loss(
            params,
            state.apply_fn,
            batch,
            aux_use_waypoint_loss,
            aux_deep_supervision,
            waypoint_loss_weight,
            gc_reg_weight,      # Add this
            gc_margin,          # Add this
            directional_weight,
            directional_margin,
            rng,
            deterministic=False,
            step=step,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Average gradients across devices
    grads = lax.pmean(grads, axis_name="batch")
    metrics = lax.pmean(metrics, axis_name="batch")

    state = state.apply_gradients(grads=grads)

    return state, metrics


def create_train_step_pmap(
    aux_use_waypoint_loss, 
    aux_deep_supervision, 
    waypoint_loss_weight,
    gc_reg_weight,      # Add this
    gc_margin,          # Add this
    directional_weight,
    directional_margin,
    ):
    """Create pmap'd train step with specific aux config."""
    def _train_step(state, batch, rng, step):
        return train_step_parallel(
            state, batch,
            aux_use_waypoint_loss, 
            aux_deep_supervision, 
            waypoint_loss_weight,
            gc_reg_weight,      # Add this
            gc_margin,          # Add this
            directional_weight,
            directional_margin,
            rng,
            step,
        )
    return jax.pmap(_train_step, axis_name="batch", in_axes=(0, 0, 0, None))


def shard_batch(batch: Dict[str, jnp.ndarray], num_devices: int) -> Dict[str, jnp.ndarray]:
    """Shard batch across devices. Shape: (batch,) -> (devices, batch//devices, ...)"""
    def _shard(x):
        batch_size = x.shape[0]
        batch_per_device = batch_size // num_devices
        return x[:num_devices * batch_per_device].reshape((num_devices, batch_per_device) + x.shape[1:])
    return {k: _shard(v) for k, v in batch.items()}


@partial(jax.jit, static_argnums=(2, 3, 4))
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
    gc_reg_weight,      # Add this
    gc_margin,          # Add this
    directional_weight,
    directional_margin,
) -> Dict[str, jnp.ndarray]:
    """Evaluation step (no gradients)."""
    _, metrics = compute_loss(
        state.params,
        state.apply_fn,
        batch,
        aux_use_waypoint_loss,
        aux_deep_supervision,
        waypoint_loss_weight,
        gc_reg_weight,      # Add this
        gc_margin,          # Add this
        directional_weight,
        directional_margin,
        rng=None,
        deterministic=True,
    )
    return metrics


def evaluate_policy(
    state: TrainState,
    env,
    config: Config,
    norm_stats,
    task_type: str = "antmaze",
    num_episodes: int = 10,
    max_steps: int = 1000,
    achieved_goals: np.ndarray = None,
) -> Dict[str, float]:
    """
    Evaluate policy with comprehensive diagnostics.
    
    Fixes:
    1. Proper success detection using env threshold
    2. Waypoint supervision along actual paths
    3. Stuck detection and recovery
    4. Goal-conditioning diagnostics
    """
    
    @jax.jit
    def get_action(params, states, actions, goals, timesteps):
        outputs = state.apply_fn(params, states=states, actions=actions, 
                                goals=goals, timesteps=timesteps, deterministic=True)
        return outputs["action_pred"]

    successes = []
    steps_list = []
    
    # Track success by goal distance buckets
    near_successes = []  # dist < 10
    mid_successes = []   # 10 <= dist < 20
    far_successes = []   # dist >= 20
    
    # Track movement quality
    all_cos_sims = []
    
    ctx_len = config.training.context_len
    action_dim = env.action_space.shape[0]
    
    obs_mean = norm_stats.obs_mean
    obs_std = norm_stats.obs_std
    goal_mean = norm_stats.goal_mean
    goal_std = norm_stats.goal_std
    
    # ============================================================
    # FIX 1: Get success threshold from environment
    # ============================================================
    if hasattr(env.unwrapped, 'goal_threshold'):
        success_threshold = env.unwrapped.goal_threshold
    elif hasattr(env.unwrapped, '_goal_threshold'):
        success_threshold = env.unwrapped._goal_threshold
    elif 'antmaze' in str(type(env.unwrapped)).lower():
        success_threshold = 0.5  # Default for antmaze
    else:
        success_threshold = 0.5
    
    print(f"Success threshold: {success_threshold}")
    
    # Find far-apart goals for counterfactual test
    if achieved_goals is not None and len(achieved_goals) > 100:
        n_samples = min(1000, len(achieved_goals))
        sample_idx = np.random.choice(len(achieved_goals), n_samples, replace=False)
        sampled_goals = achieved_goals[sample_idx]
        
        min_x_idx = np.argmin(sampled_goals[:, 0])
        max_x_idx = np.argmax(sampled_goals[:, 0])
        min_y_idx = np.argmin(sampled_goals[:, 1])
        max_y_idx = np.argmax(sampled_goals[:, 1])
        
        candidates = [min_x_idx, max_x_idx, min_y_idx, max_y_idx]
        max_dist = 0
        goal_far_1, goal_far_2 = sampled_goals[0], sampled_goals[1]
        
        for i in candidates:
            for j in candidates:
                if i != j:
                    dist = np.linalg.norm(sampled_goals[i] - sampled_goals[j])
                    if dist > max_dist:
                        max_dist = dist
                        goal_far_1 = sampled_goals[i]
                        goal_far_2 = sampled_goals[j]
        
        goal_far_1_norm = (goal_far_1 - goal_mean) / goal_std
        goal_far_2_norm = (goal_far_2 - goal_mean) / goal_std
        
        print(f"\n=== Counterfactual Goals (dist={max_dist:.1f}) ===")
        print(f"  Goal 1: {goal_far_1} -> norm: {goal_far_1_norm}")
        print(f"  Goal 2: {goal_far_2} -> norm: {goal_far_2_norm}")
    else:
        goal_far_1_norm = (np.array([0.0, 0.0]) - goal_mean) / goal_std
        goal_far_2_norm = (np.array([20.0, 20.0]) - goal_mean) / goal_std
    
    print("=" * 60)
    
    for ep in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}

        # Get goal
        if task_type == "antmaze":
            if hasattr(env.unwrapped, 'target_goal'):
                goal_raw = np.array(env.unwrapped.target_goal)
            elif hasattr(env, 'target_goal'):
                goal_raw = np.array(env.target_goal)
            else:
                goal_raw = obs[:2]
        else:
            goal_raw = info.get("goal", obs.copy())
        
        # Initial distance for bucketing
        initial_dist = np.linalg.norm(obs[:2] - goal_raw)
        
        goal_normalized = (goal_raw - goal_mean) / goal_std
        obs_normalized = (obs - obs_mean) / obs_std
        
        if ep == 0:
            print(f"\nEpisode 0: goal={goal_raw}, initial_dist={initial_dist:.1f}")

        states_buffer = [obs_normalized]
        actions_buffer = []
        
        done = False
        steps = 0
        success = False
        prev_pos = obs[:2].copy()
        
        # Track for diagnostics
        episode_cos_sims = []
        recent_positions = []
        recent_actions = []  # Track recent actions for smoothing
        counterfactual_steps = [0, 100, 200, 300, 400]
        
        # Stuck detection
        stuck_counter = 0
        
        while not done and steps < max_steps:
            # Prepare context
            states_arr = np.array(states_buffer)
            
            if len(states_buffer) < ctx_len:
                pad_len = ctx_len - len(states_buffer)
                state_pad = np.repeat(states_arr[0:1], pad_len, axis=0)
                states = np.concatenate([state_pad, states_arr], axis=0)
            else:
                states = states_arr[-ctx_len:]
            
            if len(actions_buffer) == 0:
                actions = np.zeros((ctx_len, action_dim))
            else:
                actions_arr = np.array(actions_buffer)
                if len(actions_buffer) < ctx_len - 1:
                    pad_len = ctx_len - len(actions_buffer) - 1
                    actions = np.concatenate([
                        np.zeros((pad_len, action_dim)),
                        actions_arr,
                        np.zeros((1, action_dim))
                    ], axis=0)
                else:
                    actions = np.concatenate([
                        actions_arr[-(ctx_len-1):],
                        np.zeros((1, action_dim))
                    ], axis=0)
            
            states_input = jnp.array(states[None])
            actions_input = jnp.array(actions[None])
            timesteps_input = jnp.arange(ctx_len)[None]
            
            # Counterfactual test at key steps
            if ep == 0 and steps in counterfactual_steps:
                goal_1_input = jnp.array(goal_far_1_norm[None])
                goal_2_input = jnp.array(goal_far_2_norm[None])
                
                action_g1 = np.array(get_action(state.params, states_input, actions_input, 
                                                goal_1_input, timesteps_input)[0])
                action_g2 = np.array(get_action(state.params, states_input, actions_input, 
                                                goal_2_input, timesteps_input)[0])
                
                action_diff = np.linalg.norm(action_g1 - action_g2)
                
                status = "✓ GOOD" if action_diff > 0.2 else ("⚠️ LOW" if action_diff > 0.05 else "❌ BAD")
                print(f"  Step {steps}: ||a(g1)-a(g2)||={action_diff:.3f} {status}")
            
            # Get action
            goals_input = jnp.array(goal_normalized[None])
            action = np.array(get_action(state.params, states_input, actions_input, 
                             goals_input, timesteps_input)[0])
            
            # ============================================================
            # FIX 3: Action smoothing when stuck or high action norm
            # ============================================================
            action_norm = np.linalg.norm(action)
            
            # Smooth actions if stuck
            if len(recent_actions) > 0 and stuck_counter > 10:
                # Blend with previous action to reduce oscillation
                prev_action = recent_actions[-1]
                action = 0.7 * action + 0.3 * prev_action
            
            # Reduce action magnitude if very high (prevents falls)
            if action_norm > 1.5:
                action = action / action_norm * 1.0
            
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Track recent actions
            recent_actions.append(action.copy())
            if len(recent_actions) > 20:
                recent_actions.pop(0)
            
            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            current_pos = obs[:2]
            dist_to_goal = np.linalg.norm(current_pos - goal_raw)
            
            # ============================================================
            # FIX 1: Proper success detection
            # ============================================================
            # Check info dict first (most reliable)
            env_success = info.get("is_success", info.get("success", False))
            dist_success = dist_to_goal <= success_threshold
            
            if ep == 0 and steps % 100 == 0:
                print(f"  Step {steps}: dist={dist_to_goal:.2f}, threshold={success_threshold}, "
                      f"env_success={env_success}, dist_success={dist_success}")
            
            # Cosine similarity tracking
            movement = current_pos - prev_pos
            desired_dir = goal_raw - prev_pos
            
            movement_norm = np.linalg.norm(movement)
            desired_norm = np.linalg.norm(desired_dir)
            
            if movement_norm > 0.01 and desired_norm > 0.01:
                cos_sim = np.dot(movement, desired_dir) / (movement_norm * desired_norm)
                episode_cos_sims.append(cos_sim)
            
            # Stuck detection
            recent_positions.append(current_pos.copy())
            if len(recent_positions) > 50:
                recent_positions.pop(0)
            
            if len(recent_positions) >= 10:
                recent_movement = np.linalg.norm(
                    np.array(recent_positions[-1]) - np.array(recent_positions[-10])
                )
                if recent_movement < 0.5:
                    stuck_counter += 1
                else:
                    stuck_counter = max(0, stuck_counter - 1)
            
            # Detailed diagnostics
            if ep == 0 and steps % 100 == 0:
                pos_variance = np.var(np.array(recent_positions), axis=0).sum() if len(recent_positions) >= 50 else -1
                stuck_status = f"⚠️ STUCK({stuck_counter})" if stuck_counter > 20 else ""
                
                avg_cos_sim = np.mean(episode_cos_sims[-50:]) if len(episode_cos_sims) >= 50 else (
                    np.mean(episode_cos_sims) if episode_cos_sims else 0
                )
                
                print(f"  Step {steps}: pos={current_pos}, dist={dist_to_goal:.1f}, "
                      f"action_norm={action_norm:.2f}, cos_sim={avg_cos_sim:.2f} {stuck_status}")
                
                if avg_cos_sim < 0:
                    print(f"    ❌ Moving AWAY from goal!")
                elif avg_cos_sim < 0.3:
                    print(f"    ⚠️ Weak movement toward goal")
            
            prev_pos = current_pos.copy()
            obs_normalized = (obs - obs_mean) / obs_std
            
            states_buffer.append(obs_normalized)
            actions_buffer.append(action)
            steps += 1
            
            # Success check: use env signal OR distance
            if env_success or dist_success:
                success = True
                if ep == 0:
                    print(f"  ✓ SUCCESS at step {steps}! dist={dist_to_goal:.2f}, "
                          f"env_success={env_success}, dist_success={dist_success}")
                break
        
        if done and not success:
            print(f"Episode {ep} ended: dist={dist_to_goal:.2f}, success={success}")
        
        successes.append(float(success))
        steps_list.append(steps)
        all_cos_sims.extend(episode_cos_sims)
        
        # Bucket by initial distance
        if initial_dist < 10:
            near_successes.append(float(success))
        elif initial_dist < 20:
            mid_successes.append(float(success))
        else:
            far_successes.append(float(success))
        
        if ep == 0 and not success:
            final_dist = np.linalg.norm(obs[:2] - goal_raw)
            print(f"  ✗ FAILED: final_dist={final_dist:.1f}, steps={steps}")
    
    # Summary statistics
    print(f"\n=== Evaluation Summary ===")
    print(f"Overall success: {np.mean(successes):.1%} ({sum(successes):.0f}/{len(successes)})")
    print(f"Avg steps: {np.mean(steps_list):.0f}")
    
    if near_successes:
        print(f"Near goals (<10): {np.mean(near_successes):.1%} ({len(near_successes)} episodes)")
    if mid_successes:
        print(f"Mid goals (10-20): {np.mean(mid_successes):.1%} ({len(mid_successes)} episodes)")
    if far_successes:
        print(f"Far goals (>20): {np.mean(far_successes):.1%} ({len(far_successes)} episodes)")
    
    if all_cos_sims:
        print(f"Avg cos_sim: {np.mean(all_cos_sims):.3f} (>0 = moving toward goal)")
    
    return {
        "success_rate": np.mean(successes),
        "avg_steps": np.mean(steps_list),
        "avg_cos_sim": np.mean(all_cos_sims) if all_cos_sims else 0.0,
        "near_success": np.mean(near_successes) if near_successes else -1,
        "mid_success": np.mean(mid_successes) if mid_successes else -1,
        "far_success": np.mean(far_successes) if far_successes else -1,
    }


def train(config: Config, resume_from: Optional[str] = None):
    """
    Main training loop with multi-GPU support and resume capability.
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint directory to resume from (optional)
    """

    # Allow environment overrides
    max_steps = os.environ.get("MAX_STEPS")
    if max_steps:
        config.training.max_steps = int(max_steps)
    eval_episodes = os.environ.get("EVAL_EPISODES")
    if eval_episodes:
        config.training.eval_episodes = int(eval_episodes)
    log_every = os.environ.get("LOG_EVERY")
    if log_every:
        config.training.log_every = int(log_every)
    eval_every = os.environ.get("EVAL_EVERY")
    if eval_every:
        config.training.eval_every = int(eval_every)
    save_every = os.environ.get("SAVE_EVERY")
    if save_every:
        config.training.save_every = int(save_every)

    # Set random seed
    np.random.seed(config.training.seed)
    rng = jax.random.PRNGKey(config.training.seed)

    # Multi-GPU setup
    use_multi_gpu = NUM_DEVICES > 1
    if use_multi_gpu:
        print(f"\n=== Multi-GPU Training Enabled ({NUM_DEVICES} devices) ===")
        if config.training.batch_size % NUM_DEVICES != 0:
            old_batch_size = config.training.batch_size
            config.training.batch_size = (config.training.batch_size // NUM_DEVICES) * NUM_DEVICES
            print(f"Adjusted batch_size: {old_batch_size} -> {config.training.batch_size}")
    
    # Handle output directory for resume vs new training
    if resume_from:
        output_dir = os.path.abspath(resume_from)
        print(f"\n=== Resuming training from {resume_from} ===")
        
        # Load config from checkpoint directory if exists
        config_path = os.path.join(resume_from, "config.json")
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}")
            # Note: You might want to merge with current config for some overrides
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.output_dir, f"{config.exp_name}_{timestamp}")
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        def config_to_dict(value):
            if dataclasses.is_dataclass(value):
                return {k: config_to_dict(v) for k, v in dataclasses.asdict(value).items()}
            return value

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_to_dict(config), f, indent=2)
    
    print(f"Output directory: {output_dir}")
    print(f"Loading dataset: {config.data.dataset_name} (type: {config.data.dataset_type})")

    if config.data.dataset_type == "ogbench" and config.data.goal_sampling == "her":
        print("OGBench does not support HER goal sampling. Using future goals instead.")
        config.data.goal_sampling = "future"

    # Load dataset
    if config.data.dataset_type == "d4rl":
        train_dataset, val_dataset, env_info = load_d4rl_dataset(
            dataset_name=config.data.dataset_name,
            context_len=config.training.context_len,
            goal_sampling=config.data.goal_sampling,
            min_goal_horizon=config.data.min_goal_horizon,
            max_goal_horizon=config.data.max_goal_horizon,
            waypoint_horizon=config.aux.waypoint_horizon,
            her_relabel_prob=config.data.her_relabel_prob,
        )
        DataLoader = D4RLDataLoader
        batch_to_jax = d4rl_batch_to_jax
        print(f"HER relabeling probability: {config.data.her_relabel_prob}")
    else:
        train_dataset, val_dataset, env_info = load_ogbench_dataset(
            dataset_name=config.data.dataset_name,
            dataset_dir=config.data.dataset_dir,
            context_len=config.training.context_len,
            goal_sampling=config.data.goal_sampling,
            min_goal_horizon=config.data.min_goal_horizon,
            max_goal_horizon=config.data.max_goal_horizon,
            waypoint_horizon=config.aux.waypoint_horizon,
        )
        DataLoader = OGBenchDataLoader
        batch_to_jax = ogbench_batch_to_jax

    # Update config with environment info
    config.state_dim = env_info["state_dim"]
    config.action_dim = env_info["action_dim"]
    config.goal_dim = env_info.get("goal_dim", env_info["state_dim"])

    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}, Goal dim: {config.goal_dim}")
    print(f"norm_stats in env_info: {'norm_stats' in env_info}")
    print(f"norm_stats.goal_mean shape: {env_info['norm_stats'].goal_mean.shape}")

    # Save normalization stats (only for new training)
    if not resume_from and "norm_stats" in env_info and env_info["norm_stats"] is not None:
        norm_stats = env_info["norm_stats"]
        np.savez(
            os.path.join(output_dir, "norm_stats.npz"),
            obs_mean=norm_stats.obs_mean,
            obs_std=norm_stats.obs_std,
            goal_mean=norm_stats.goal_mean,
            goal_std=norm_stats.goal_std,
        )
        print(f"Saved normalization stats to {output_dir}/norm_stats.npz")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        seed=config.training.seed,
        infinite=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        seed=config.training.seed + 1,
        infinite=True,
    )

    # Verify data loader
    b = train_loader.get_batch()
    print(f"Last action: {b.actions[0, -1, :]}") 
    print(f"Target: {b.target_actions[0, :]}")
    
    # Create model and initial state
    rng, init_rng = jax.random.split(rng)
    model = create_model(config)
    state = create_train_state(
        init_rng,
        model,
        config,
        config.state_dim,
        config.action_dim,
        config.goal_dim,
    )

    # Resume from checkpoint if specified
    start_step = 0
    best_success_rate = 0.0
    
    if resume_from:
        print(f"\nLooking for checkpoints in {resume_from}...")
        
        # Try to load the latest checkpoint
        try:
            state, start_step = load_checkpoint(resume_from, state, prefix="checkpoint_")
            print(f"✓ Restored checkpoint at step {start_step}")
        except Exception as e:
            print(f"Could not load regular checkpoint: {e}")
            # Try loading best checkpoint
            try:
                state, start_step = load_checkpoint(resume_from, state, prefix="best_")
                print(f"✓ Restored best checkpoint at step {start_step}")
            except Exception as e2:
                print(f"Could not load best checkpoint either: {e2}")
                print("Starting from scratch...")
                start_step = 0
        
        # Load best success rate from training state file if exists
        training_state_path = os.path.join(resume_from, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
                best_success_rate = training_state.get("best_success_rate", 0.0)
                print(f"✓ Restored best_success_rate: {best_success_rate:.3f}")
        
        # Advance RNG to match resumed step
        for _ in range(start_step):
            rng, _ = jax.random.split(rng)
        print(f"Advanced RNG state to step {start_step}")

    directional_weight = config.aux.directional_weight
    directional_margin = config.aux.directional_margin
    if config.data.dataset_type == "ogbench":
        if directional_weight != 0.0:
            print("Disabling directional waypoint loss for OGBench.")
        directional_weight = 0.0

    # Multi-GPU: replicate state
    if use_multi_gpu:
        state = replicate(state)
        train_step_fn = create_train_step_pmap(
            config.aux.use_waypoint_loss,
            config.aux.deep_supervision,
            config.aux.waypoint_loss_weight,
            config.aux.gc_reg_weight,
            config.aux.gc_margin,
            directional_weight,
            directional_margin,
        )
        print(f"State replicated across {NUM_DEVICES} devices")
    else:
        train_step_fn = None

    # Training loop
    print(f"\nStarting training from step {start_step} to {config.training.max_steps}...")
    
    train_iter = iter(train_loader)
    aux_use_waypoint_loss = config.aux.use_waypoint_loss
    aux_deep_supervision = config.aux.deep_supervision
    waypoint_loss_weight = config.aux.waypoint_loss_weight

    # Skip batches if resuming (to maintain data consistency)
    if start_step > 0:
        print(f"Skipping {start_step} batches to sync data iterator...")
        for _ in range(start_step):
            next(train_iter)
        print("Data iterator synced.")

    gc_reg_weight = config.aux.gc_reg_weight
    gc_margin = config.aux.gc_margin
    
    for step in range(start_step, config.training.max_steps):
        rng, step_rng = jax.random.split(rng)

        # Get batch
        batch = next(train_iter)
        batch = batch_to_jax(batch)

        # Train step
        if use_multi_gpu:
            batch = shard_batch(batch, NUM_DEVICES)
            step_rngs = jax.random.split(step_rng, NUM_DEVICES)
            state, metrics = train_step_fn(state, batch, step_rngs, step)
            metrics = {k: v[0] for k, v in metrics.items()}
        else:
            state, metrics = train_step_single(
                state,
                batch,
                aux_use_waypoint_loss,
                aux_deep_supervision,
                waypoint_loss_weight,
                gc_reg_weight,      # New
                gc_margin,
                directional_weight,
                directional_margin,
                step_rng,
                step,
            )

        # Logging
        if step % config.training.log_every == 0:
            metrics_np = {k: float(v) for k, v in metrics.items()}
            print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics_np.items()))
        
        # Evaluation
        if step > 0 and step % config.training.eval_every == 0:
            print(f"\nEvaluating at step {step}...")

            eval_state = unreplicate(state) if use_multi_gpu else state

            # Validation loss
            val_batch = val_loader.get_batch()
            val_batch = batch_to_jax(val_batch)
            val_metrics = eval_step(
                eval_state,
                val_batch,
                aux_use_waypoint_loss,
                aux_deep_supervision,
                waypoint_loss_weight,
                gc_reg_weight,      # Add this
                gc_margin,          # Add this
                directional_weight,
                directional_margin,
            )
            val_metrics_np = {f"val_{k}": float(v) for k, v in val_metrics.items()}
            print("Validation: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics_np.items()))

            # Policy evaluation
            eval_metrics = evaluate_policy(
                eval_state,
                env_info["env"],
                config,
                norm_stats=env_info["norm_stats"],
                task_type=env_info.get("task_type", "antmaze"),
                num_episodes=config.training.eval_episodes,
                achieved_goals=train_dataset.achieved_goals,
            )
            print(f"Policy eval: success_rate={eval_metrics['success_rate']:.3f}, "
                  f"avg_steps={eval_metrics['avg_steps']:.1f}")

            # Save best model
            if eval_metrics["success_rate"] > best_success_rate:
                best_success_rate = eval_metrics["success_rate"]
                checkpoints.save_checkpoint(
                    output_dir,
                    eval_state,
                    step,
                    prefix="best_",
                    keep=1,
                )
                print(f"New best model saved! Success rate: {best_success_rate:.3f}")
                
                # Save training state
                with open(os.path.join(output_dir, "training_state.json"), "w") as f:
                    json.dump({
                        "best_success_rate": best_success_rate,
                        "best_step": step,
                    }, f)

            print()

        # Save checkpoint
        if step > 0 and step % config.training.save_every == 0:
            save_state = unreplicate(state) if use_multi_gpu else state
            checkpoints.save_checkpoint(
                output_dir,
                save_state,
                step,
                keep=3,
            )
            
            # Also save training state
            with open(os.path.join(output_dir, "training_state.json"), "w") as f:
                json.dump({
                    "best_success_rate": best_success_rate,
                    "current_step": step,
                }, f)
    
    print(f"\nTraining complete! Best success rate: {best_success_rate:.3f}")
    
    return state, best_success_rate


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ut_gcdt_full",
                       choices=["gcdt_baseline", "ut_gcdt", "ut_gcdt_plan", "ut_gcdt_full",
                                "ut_gcdt_gated", "ut_gcdt_gated_full", "ut_gcdt_multiblock"])
    parser.add_argument("--dataset", type=str, default="antmaze-medium-stitch-v0")
    parser.add_argument("--dataset_type", type=str, default=None, choices=["d4rl", "ogbench"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint directory to resume training from")
    
    # Training overrides
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--eval_episodes", type=int, default=None, help="Override eval episodes")
    parser.add_argument("--eval_every", type=int, default=None, help="Override eval frequency")
    
    args = parser.parse_args()
    
    # Get config
    from configs.default import (
        get_gcdt_baseline_config,
        get_ut_gcdt_config,
        get_ut_gcdt_plan_config,
        get_ut_gcdt_full_config,
        get_ut_gcdt_gated_config,
        get_ut_gcdt_gated_full_config,
        get_ut_gcdt_multiblock_config,
    )

    config_map = {
        "gcdt_baseline": get_gcdt_baseline_config,
        "ut_gcdt": get_ut_gcdt_config,
        "ut_gcdt_plan": get_ut_gcdt_plan_config,
        "ut_gcdt_full": get_ut_gcdt_full_config,
        "ut_gcdt_gated": get_ut_gcdt_gated_config,
        "ut_gcdt_gated_full": get_ut_gcdt_gated_full_config,
        "ut_gcdt_multiblock": get_ut_gcdt_multiblock_config,
    }
    
    config = config_map[args.config]()
    config.data.dataset_name = args.dataset
    if args.dataset_type is not None:
        config.data.dataset_type = args.dataset_type
    config.training.seed = args.seed
    config.output_dir = os.path.abspath(args.output_dir)

    resume_path = None
    if args.resume:
        resume_path = os.path.abspath(args.resume)

    # Apply command-line overrides
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.eval_episodes is not None:
        config.training.eval_episodes = args.eval_episodes
    if args.eval_every is not None:
        config.training.eval_every = args.eval_every

    # Train with optional resume
    train(config, resume_from=args.resume)
