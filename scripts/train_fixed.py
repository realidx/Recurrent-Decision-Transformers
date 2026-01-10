"""Training script for UT-GCDT on D4RL/OGBench."""

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
    
    # Action prediction loss (MSE) - final layer/step only
    action_pred = outputs["action_pred"]
    target_actions = batch["target_actions"]
    action_loss = jnp.mean((action_pred - target_actions) ** 2)

    metrics = {"action_loss": action_loss}

    # V3 Protocol: Sum of MSE losses from ALL heads/steps (not weighted)
    # For GCDT: independent heads at each layer L
    # For U-GCDT: shared head applied at each step K
    # Loss = sum_{l=1}^{L} MSE(Head_l(x_l), a_true)
    if "intermediate_action_preds" in outputs and aux_deep_supervision:
        intermediate_preds = outputs["intermediate_action_preds"]
        if len(intermediate_preds) >= 1:
            # Progressive weighting: later iterations weighted more
            total_loss = 0.0
            num_preds = len(intermediate_preds)
            for k, pred in enumerate(intermediate_preds):
                step_weight = (k + 1) / num_preds
                total_loss += step_weight * jnp.mean((pred - target_actions) ** 2)
            metrics["deep_action_loss"] = total_loss
            metrics["action_loss"] = jnp.mean((intermediate_preds[-1] - target_actions) ** 2)
        else:
            total_loss = action_loss
    else:
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
    """Evaluate the current policy in the environment.

    Notes:
      - Supports both Gym (old API) and Gymnasium (new API).
      - Supports both OGBench-style multi-task envs (reset(options={task_id}))
        and D4RL envs (no options kwarg, no info['goal']).
    """

    def _safe_reset(task_id: int | None = None):
        kwargs = {}
        # OGBench-style: reset(options={"task_id": ...})
        if task_id is not None:
            kwargs["options"] = {"task_id": int(task_id)}
        try:
            out = env.reset(**kwargs)
        except TypeError:
            # D4RL AntMaze (and many Gym envs) do not accept 'options'
            out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def _safe_step(action):
        out = env.step(action)
        # Gymnasium API: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        # Gym API: (obs, reward, done, info)
        obs, reward, done, info = out
        return obs, reward, bool(done), info

    def _unwrap_env(e):
        # Try to reach the base env; robust across wrappers
        base = getattr(e, "unwrapped", e)
        for attr in ("_wrapped_env", "env"):
            # unwrap repeatedly if nested
            while hasattr(base, attr):
                base = getattr(base, attr)
        return base

    def _get_goal(obs, info):
        # 1) From info dict (OGBench-style)
        if isinstance(info, dict) and "goal" in info:
            g = info["goal"]
        else:
            g = None

        # 2) From dict observations (Gymnasium goal envs)
        if g is None and isinstance(obs, dict):
            for k in ("desired_goal", "goal"):
                if k in obs:
                    g = obs[k]
                    break

        # 3) From common env attributes (D4RL AntMaze keeps internal target goal)
        if g is None:
            base = _unwrap_env(env)
            for attr in ("target_goal", "_target_goal", "goal", "_goal"):
                if hasattr(base, attr):
                    cand = getattr(base, attr)
                    # Ignore empty tuples like ()
                    try:
                        if isinstance(cand, tuple) and len(cand) == 0:
                            continue
                    except Exception:
                        pass
                    g = cand
                    break

        goal_dim = getattr(config, "goal_dim", None)
        if g is None:
            if goal_dim is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros((goal_dim,), dtype=np.float32)

        g_arr = np.asarray(g, dtype=np.float32).reshape(-1)

        # Normalize goal dimension if we know it
        if goal_dim is not None:
            if g_arr.size > goal_dim:
                g_arr = g_arr[:goal_dim]
            elif g_arr.size < goal_dim:
                pad = np.zeros((goal_dim - g_arr.size,), dtype=np.float32)
                g_arr = np.concatenate([g_arr, pad], axis=0)

        return g_arr

    def _extract_obs_vec(obs):
        # Some envs return dict obs; prefer "observation" if present
        if isinstance(obs, dict) and "observation" in obs:
            return obs["observation"]
        return obs

    def _is_success(info, reward):
        if isinstance(info, dict):
            if info.get("success", False):
                return True
            if info.get("is_success", False):
                return True
            if info.get("goal_achieved", False):
                return True
        # AntMaze often gives reward 1 on success (sparse reward)
        try:
            if reward is not None and float(reward) > 0.0:
                return True
        except Exception:
            pass
        return False

    # Decide evaluation mode
    dataset_type = getattr(getattr(config, "data", None), "dataset_type", None)
    ogbench_mode = (dataset_type == "ogbench")

    # OGBench: 5 tasks; D4RL: single task
    task_ids = list(range(1, 6)) if ogbench_mode else [None]
    episodes_per_task = max(1, num_episodes // len(task_ids))

    all_success_flags: list[float] = []
    all_steps: list[int] = []

    for task_id in task_ids:
        for _ in range(episodes_per_task):
            obs, info = _safe_reset(task_id=task_id)
            obs_vec = _extract_obs_vec(obs)
            goal = _get_goal(obs, info)

            # Initialize context buffers
            states_buffer = [np.asarray(obs_vec)]
            actions_buffer = []
            done = False
            steps = 0
            succeeded = False

            while (not done) and steps < max_steps:
                # Prepare input for model
                ctx_len = config.training.context_len
                action_dim = env.action_space.shape[0]

                states_arr = np.asarray(states_buffer, dtype=np.float32)
                if actions_buffer:
                    actions_arr = np.asarray(actions_buffer, dtype=np.float32)
                    actions_seq = np.concatenate([actions_arr, np.zeros((1, action_dim), dtype=np.float32)], axis=0)
                    pad_action = np.asarray(actions_buffer[0], dtype=np.float32)
                else:
                    actions_seq = np.zeros((1, action_dim), dtype=np.float32)
                    pad_action = np.zeros((action_dim,), dtype=np.float32)

                if len(states_buffer) < ctx_len:
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

                # Step environment (Gym/Gymnasium compatible)
                obs, reward, done, info = _safe_step(action)
                obs_vec = _extract_obs_vec(obs)

                # Update buffers
                states_buffer.append(np.asarray(obs_vec))
                actions_buffer.append(action)
                steps += 1

                if _is_success(info, reward):
                    succeeded = True
                    break

            all_success_flags.append(1.0 if succeeded else 0.0)
            all_steps.append(steps if succeeded else max_steps)

    success_rate = float(np.mean(all_success_flags)) if all_success_flags else 0.0
    avg_steps = float(np.mean(all_steps)) if all_steps else float(max_steps)

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
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
    print(f"Loading dataset: {config.data.dataset_name} (type: {config.data.dataset_type})")

    # Load dataset based on type
    if config.data.dataset_type == "d4rl":
        train_dataset, val_dataset, env_info = load_d4rl_dataset(
            dataset_name=config.data.dataset_name,
            context_len=config.training.context_len,
            goal_sampling=config.data.goal_sampling,
            min_goal_horizon=config.data.min_goal_horizon,
            max_goal_horizon=config.data.max_goal_horizon,
            waypoint_horizon=config.aux.waypoint_horizon,
            her_relabel_prob=config.data.her_relabel_prob,  # HER for stitching
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
    # For D4RL antmaze, goal_dim may differ from state_dim
    config.goal_dim = env_info.get("goal_dim", env_info["state_dim"])

    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}, Goal dim: {config.goal_dim}")

    # Save normalization stats for eval
    if "norm_stats" in env_info and env_info["norm_stats"] is not None:
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
    
    # Create model
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
                       choices=["gcdt_baseline", "ut_gcdt", "ut_gcdt_plan", "ut_gcdt_full",
                                "ut_gcdt_gated", "ut_gcdt_gated_full"])
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
        get_ut_gcdt_gated_config,
        get_ut_gcdt_gated_full_config,
    )

    config_map = {
        "gcdt_baseline": get_gcdt_baseline_config,
        "ut_gcdt": get_ut_gcdt_config,
        "ut_gcdt_plan": get_ut_gcdt_plan_config,
        "ut_gcdt_full": get_ut_gcdt_full_config,
        "ut_gcdt_gated": get_ut_gcdt_gated_config,
        "ut_gcdt_gated_full": get_ut_gcdt_gated_full_config,
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
