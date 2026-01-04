"""Training script for UT-GCDT on OGBench."""

import os
import sys
import time
import json
from datetime import datetime
from functools import partial
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state, checkpoints
import optax
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import Config, get_ut_gcdt_full_config
from models import create_model, UTGCDT
from data.ogbench_loader import load_ogbench_dataset, DataLoader, batch_to_jax


class TrainState(train_state.TrainState):
    """Extended train state with additional fields."""
    pass


def create_train_state(
    rng: jax.random.PRNGKey,
    model: UTGCDT,
    config: Config,
    state_dim: int,
    action_dim: int,
) -> TrainState:
    """Initialize model and optimizer."""
    
    # Create dummy inputs for initialization
    batch_size = 2
    seq_len = config.training.context_len
    
    dummy_states = jnp.zeros((batch_size, seq_len, state_dim))
    dummy_actions = jnp.zeros((batch_size, seq_len, action_dim))
    dummy_goals = jnp.zeros((batch_size, state_dim))
    dummy_timesteps = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    
    # Initialize parameters
    params = model.init(
        rng,
        states=dummy_states,
        actions=dummy_actions,
        goals=dummy_goals,
        timesteps=dummy_timesteps,
        deterministic=True,
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


def compute_loss(
    params: Dict,
    apply_fn,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
    rng: jax.random.PRNGKey,
    deterministic: bool = False,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute training loss.
    
    Args:
        params: Model parameters
        apply_fn: Model forward function
        batch: Dictionary with states, actions, goals, timesteps, target_actions, future_states
        aux_use_waypoint_loss: Whether to include waypoint loss
        aux_deep_supervision: Whether to apply waypoint loss at each iteration
        waypoint_loss_weight: Scale for waypoint loss
        rng: Random key for dropout
        deterministic: Whether to disable dropout
        
    Returns:
        total_loss: Scalar loss
        metrics: Dictionary of individual loss components
    """
    # Forward pass
    outputs = apply_fn(
        params,
        states=batch["states"],
        actions=batch["actions"],
        goals=batch["goals"],
        timesteps=batch["timesteps"],
        deterministic=deterministic,
        return_intermediates=aux_use_waypoint_loss and aux_deep_supervision,
        rngs={"dropout": rng} if not deterministic else None,
    )
    
    # Action prediction loss (MSE)
    action_pred = outputs["action_pred"]
    target_actions = batch["target_actions"]
    action_loss = jnp.mean((action_pred - target_actions) ** 2)
    
    metrics = {"action_loss": action_loss}
    total_loss = action_loss
    
    # Waypoint auxiliary loss
    if aux_use_waypoint_loss and "waypoint_preds" in outputs:
        waypoint_preds = outputs["waypoint_preds"]
        future_states = batch["future_states"]
        
        if aux_deep_supervision:
            # Apply loss at each iteration
            waypoint_loss = 0.0
            for wp_pred in waypoint_preds:
                waypoint_loss += jnp.mean((wp_pred - future_states) ** 2)
            waypoint_loss /= len(waypoint_preds)
        else:
            # Only final iteration
            waypoint_loss = jnp.mean((waypoint_preds[-1] - future_states) ** 2)
        
        waypoint_loss = waypoint_loss_weight * waypoint_loss
        metrics["waypoint_loss"] = waypoint_loss
        total_loss += waypoint_loss
    
    metrics["total_loss"] = total_loss
    
    return total_loss, metrics


@partial(jax.jit, static_argnums=(2, 3, 4))
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
    rng: jax.random.PRNGKey,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step."""
    
    def loss_fn(params):
        return compute_loss(
            params,
            state.apply_fn,
            batch,
            aux_use_waypoint_loss,
            aux_deep_supervision,
            waypoint_loss_weight,
            rng,
            deterministic=False,
        )
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


@partial(jax.jit, static_argnums=(2, 3, 4))
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
) -> Dict[str, jnp.ndarray]:
    """Evaluation step (no gradients)."""
    _, metrics = compute_loss(
        state.params,
        state.apply_fn,
        batch,
        aux_use_waypoint_loss,
        aux_deep_supervision,
        waypoint_loss_weight,
        rng=None,
        deterministic=True,
    )
    return metrics


def evaluate_policy(
    state: TrainState,
    env,
    config: Config,
    num_episodes: int = 10,
    max_steps: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate policy in environment.
    
    Returns:
        Dictionary with success_rate, avg_steps, etc.
    """
    successes = []
    steps_list = []
    
    for task_id in range(1, 6):  # OGBench has 5 evaluation tasks
        task_successes = 0
        task_steps = []
        
        for _ in range(num_episodes // 5):
            obs, info = env.reset(options={"task_id": task_id})
            goal = info["goal"]
            
            # Initialize context buffers
            states_buffer = [obs]
            actions_buffer = []
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # Prepare input for model
                # Pad context if needed
                ctx_len = config.training.context_len
                
                if len(states_buffer) < ctx_len:
                    # Pad with first observation
                    pad_len = ctx_len - len(states_buffer)
                    states = np.array([states_buffer[0]] * pad_len + states_buffer)
                    if len(actions_buffer) == 0:
                        actions = np.zeros((ctx_len, env.action_space.shape[0]))
                    else:
                        actions = np.array([actions_buffer[0]] * (pad_len + 1) + actions_buffer[:-1])
                        actions = actions[-ctx_len:]
                else:
                    states = np.array(states_buffer[-ctx_len:])
                    actions = np.array(actions_buffer[-(ctx_len-1):] + [actions_buffer[-1]] if actions_buffer else [[0]*env.action_space.shape[0]])
                    if len(actions) < ctx_len:
                        actions = np.concatenate([
                            np.zeros((ctx_len - len(actions), env.action_space.shape[0])),
                            actions
                        ])
                
                # Add batch dimension
                states_input = jnp.array(states[None])
                actions_input = jnp.array(actions[None])
                goals_input = jnp.array(goal[None])
                timesteps_input = jnp.arange(ctx_len)[None]
                
                # Get action prediction
                outputs = state.apply_fn(
                    state.params,
                    states=states_input,
                    actions=actions_input,
                    goals=goals_input,
                    timesteps=timesteps_input,
                    deterministic=True,
                )
                action = np.array(outputs["action_pred"][0])
                
                # Clip action to valid range
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update buffers
                states_buffer.append(obs)
                actions_buffer.append(action)
                steps += 1
                
                if info.get("success", False):
                    task_successes += 1
                    task_steps.append(steps)
                    break
            
            if not info.get("success", False):
                task_steps.append(max_steps)
        
        successes.append(task_successes / (num_episodes // 5))
        steps_list.extend(task_steps)
    
    return {
        "success_rate": np.mean(successes),
        "avg_steps": np.mean(steps_list),
        "success_per_task": successes,
    }


def train(config: Config):
    """Main training loop."""
    
    # Set random seed
    np.random.seed(config.training.seed)
    rng = jax.random.PRNGKey(config.training.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"{config.exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    print(f"Output directory: {output_dir}")
    print(f"Loading dataset: {config.data.dataset_name}")
    
    # Load dataset
    train_dataset, val_dataset, env_info = load_ogbench_dataset(
        dataset_name=config.data.dataset_name,
        dataset_dir=config.data.dataset_dir,
        context_len=config.training.context_len,
        goal_sampling=config.data.goal_sampling,
        min_goal_horizon=config.data.min_goal_horizon,
        max_goal_horizon=config.data.max_goal_horizon,
        waypoint_horizon=config.aux.waypoint_horizon,
    )
    
    # Update config with environment info
    config.state_dim = env_info["state_dim"]
    config.action_dim = env_info["action_dim"]
    
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}")
    
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
    
    # Create model
    rng, init_rng = jax.random.split(rng)
    model = create_model(config)
    state = create_train_state(
        init_rng,
        model,
        config,
        config.state_dim,
        config.action_dim,
    )
    
    # Training loop
    print(f"\nStarting training for {config.training.max_steps} steps...")
    
    train_iter = iter(train_loader)
    best_success_rate = 0.0
    aux_use_waypoint_loss = config.aux.use_waypoint_loss
    aux_deep_supervision = config.aux.deep_supervision
    waypoint_loss_weight = config.aux.waypoint_loss_weight
    
    for step in range(config.training.max_steps):
        rng, step_rng = jax.random.split(rng)
        
        # Get batch
        batch = next(train_iter)
        batch = batch_to_jax(batch)
        
        # Train step
        state, metrics = train_step(
            state,
            batch,
            aux_use_waypoint_loss,
            aux_deep_supervision,
            waypoint_loss_weight,
            step_rng,
        )
        
        # Logging
        if step % config.training.log_every == 0:
            metrics_np = {k: float(v) for k, v in metrics.items()}
            print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics_np.items()))
        
        # Evaluation
        if step > 0 and step % config.training.eval_every == 0:
            print(f"\nEvaluating at step {step}...")
            
            # Validation loss
            val_batch = val_loader.get_batch()
            val_batch = batch_to_jax(val_batch)
            val_metrics = eval_step(
                state,
                val_batch,
                aux_use_waypoint_loss,
                aux_deep_supervision,
                waypoint_loss_weight,
            )
            val_metrics_np = {f"val_{k}": float(v) for k, v in val_metrics.items()}
            print("Validation: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics_np.items()))
            
            # Policy evaluation
            eval_metrics = evaluate_policy(
                state,
                env_info["env"],
                config,
                num_episodes=config.training.eval_episodes,
            )
            print(f"Policy eval: success_rate={eval_metrics['success_rate']:.3f}, "
                  f"avg_steps={eval_metrics['avg_steps']:.1f}")
            
            # Save best model
            if eval_metrics["success_rate"] > best_success_rate:
                best_success_rate = eval_metrics["success_rate"]
                checkpoints.save_checkpoint(
                    output_dir,
                    state,
                    step,
                    prefix="best_",
                    keep=1,
                )
                print(f"New best model saved! Success rate: {best_success_rate:.3f}")
            
            print()
        
        # Save checkpoint
        if step > 0 and step % config.training.save_every == 0:
            checkpoints.save_checkpoint(
                output_dir,
                state,
                step,
                keep=3,
            )
    
    print(f"\nTraining complete! Best success rate: {best_success_rate:.3f}")
    
    return state, best_success_rate


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ut_gcdt_full",
                       choices=["gcdt_baseline", "ut_gcdt", "ut_gcdt_plan", "ut_gcdt_full"])
    parser.add_argument("--dataset", type=str, default="antmaze-medium-stitch-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    
    # Get config
    from configs.default import (
        get_gcdt_baseline_config,
        get_ut_gcdt_config,
        get_ut_gcdt_plan_config,
        get_ut_gcdt_full_config,
    )
    
    config_map = {
        "gcdt_baseline": get_gcdt_baseline_config,
        "ut_gcdt": get_ut_gcdt_config,
        "ut_gcdt_plan": get_ut_gcdt_plan_config,
        "ut_gcdt_full": get_ut_gcdt_full_config,
    }
    
    config = config_map[args.config]()
    config.data.dataset_name = args.dataset
    config.training.seed = args.seed
    config.output_dir = args.output_dir
    
    train(config)
