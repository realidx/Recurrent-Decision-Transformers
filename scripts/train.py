"""Training script for UT-GCDT on OGBench."""

import os
import sys
import time
import json
import dataclasses
from datetime import datetime
from functools import partial
from typing import Dict, Any, Tuple

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
    # return_intermediates=True enables deep supervision for both:
    # - UTGCDT: waypoint predictions at each iteration
    # - GCDT: action predictions from each layer's independent head
    outputs = apply_fn(
        params,
        states=batch["states"],
        actions=batch["actions"],
        goals=batch["goals"],
        timesteps=batch["timesteps"],
        deterministic=deterministic,
        return_intermediates=aux_deep_supervision,
        rngs={"dropout": rng} if not deterministic else None,
    )
    
    # Action prediction loss (MSE)
    action_pred = outputs["action_pred"]
    target_actions = batch["target_actions"]
    action_loss = jnp.mean((action_pred - target_actions) ** 2)

    metrics = {"action_loss": action_loss}
    total_loss = action_loss

    # Deep supervision for stacked GCDT with independent heads
    # Per protocol: Each layer l has its own head H_l with supervision
    if "intermediate_action_preds" in outputs and aux_deep_supervision:
        intermediate_preds = outputs["intermediate_action_preds"]
        if len(intermediate_preds) > 1:
            # Apply loss to all intermediate heads (weighted towards later layers)
            deep_action_loss = 0.0
            num_layers = len(intermediate_preds)
            for i, pred in enumerate(intermediate_preds[:-1]):  # Exclude final (already in action_loss)
                weight = (i + 1) / num_layers  # Later layers matter more
                deep_action_loss += weight * jnp.mean((pred - target_actions) ** 2)
            deep_action_loss /= (num_layers - 1) if num_layers > 1 else 1
            metrics["deep_action_loss"] = deep_action_loss
            total_loss += 0.5 * deep_action_loss  # Weighted contribution

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
def train_step_single(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    aux_use_waypoint_loss: bool,
    aux_deep_supervision: bool,
    waypoint_loss_weight: float,
    rng: jax.random.PRNGKey,
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
            rng,
            deterministic=False,
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
    rng: jax.random.PRNGKey,
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
            rng,
            deterministic=False,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Average gradients across devices
    grads = lax.pmean(grads, axis_name="batch")
    metrics = lax.pmean(metrics, axis_name="batch")

    state = state.apply_gradients(grads=grads)

    return state, metrics


def create_train_step_pmap(aux_use_waypoint_loss, aux_deep_supervision, waypoint_loss_weight):
    """Create pmap'd train step with specific aux config."""
    def _train_step(state, batch, rng):
        return train_step_parallel(
            state, batch,
            aux_use_waypoint_loss, aux_deep_supervision, waypoint_loss_weight,
            rng
        )
    return jax.pmap(_train_step, axis_name="batch")


def shard_batch(batch: Dict[str, jnp.ndarray], num_devices: int) -> Dict[str, jnp.ndarray]:
    """Shard batch across devices. Shape: (batch,) -> (devices, batch//devices, ...)"""
    def _shard(x):
        # Reshape to (num_devices, batch_per_device, ...)
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
                action_dim = env.action_space.shape[0]

                states_arr = np.array(states_buffer)
                if actions_buffer:
                    actions_arr = np.array(actions_buffer)
                    actions_seq = np.concatenate([actions_arr, np.zeros((1, action_dim))], axis=0)
                    pad_action = np.array(actions_buffer[0])
                else:
                    actions_seq = np.zeros((1, action_dim))
                    pad_action = np.zeros(action_dim)

                if len(states_buffer) < ctx_len:
                    # Pad with first observation/action
                    pad_len = ctx_len - len(states_buffer)
                    state_pad = np.repeat(states_arr[0:1], pad_len, axis=0)
                    states = np.concatenate([state_pad, states_arr], axis=0)

                    action_pad = np.repeat(pad_action[None, :], pad_len, axis=0)
                    actions = np.concatenate([action_pad, actions_seq], axis=0)
                else:
                    states = states_arr[-ctx_len:]
                    actions = actions_seq[-ctx_len:]
                
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
    """Main training loop with multi-GPU support."""

    # Allow environment overrides for fast iteration on clusters
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
        # Ensure batch size is divisible by number of devices
        if config.training.batch_size % NUM_DEVICES != 0:
            old_batch_size = config.training.batch_size
            config.training.batch_size = (config.training.batch_size // NUM_DEVICES) * NUM_DEVICES
            print(f"Adjusted batch_size: {old_batch_size} -> {config.training.batch_size} (divisible by {NUM_DEVICES})")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"{config.exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    def config_to_dict(value):
        """Recursively convert dataclasses to dicts for JSON serialization."""
        if dataclasses.is_dataclass(value):
            return {k: config_to_dict(v) for k, v in dataclasses.asdict(value).items()}
        return value

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_to_dict(config), f, indent=2)
    
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

    # Multi-GPU: replicate state and create pmap'd train step
    if use_multi_gpu:
        state = replicate(state)
        train_step_fn = create_train_step_pmap(
            config.aux.use_waypoint_loss,
            config.aux.deep_supervision,
            config.aux.waypoint_loss_weight,
        )
        print(f"State replicated across {NUM_DEVICES} devices")
    else:
        train_step_fn = None  # Use train_step_single

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

        # Train step (multi-GPU or single GPU)
        if use_multi_gpu:
            # Shard batch and RNG across devices
            batch = shard_batch(batch, NUM_DEVICES)
            step_rngs = jax.random.split(step_rng, NUM_DEVICES)
            state, metrics = train_step_fn(state, batch, step_rngs)
            # Get metrics from first device (they're averaged via pmean)
            metrics = {k: v[0] for k, v in metrics.items()}
        else:
            state, metrics = train_step_single(
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

            # Get unreplicated state for evaluation (multi-GPU)
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
            )
            val_metrics_np = {f"val_{k}": float(v) for k, v in val_metrics.items()}
            print("Validation: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics_np.items()))

            # Policy evaluation
            eval_metrics = evaluate_policy(
                eval_state,
                env_info["env"],
                config,
                num_episodes=config.training.eval_episodes,
            )
            print(f"Policy eval: success_rate={eval_metrics['success_rate']:.3f}, "
                  f"avg_steps={eval_metrics['avg_steps']:.1f}")

            # Save best model (use unreplicated state)
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

            print()

        # Save checkpoint (use unreplicated state for multi-GPU)
        if step > 0 and step % config.training.save_every == 0:
            save_state = unreplicate(state) if use_multi_gpu else state
            checkpoints.save_checkpoint(
                output_dir,
                save_state,
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
    # Training overrides for fast iteration
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

    # Apply command-line overrides
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.eval_episodes is not None:
        config.training.eval_episodes = args.eval_episodes
    if args.eval_every is not None:
        config.training.eval_every = args.eval_every

    train(config)
