"""Data loading utilities for D4RL datasets."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Iterator
from dataclasses import dataclass


@dataclass
class TrajectoryBatch:
    """Batch of trajectory data for GCDT training."""
    states: np.ndarray       # (batch, context_len, state_dim)
    actions: np.ndarray      # (batch, context_len, action_dim)
    goals: np.ndarray        # (batch, goal_dim)
    timesteps: np.ndarray    # (batch, context_len)
    target_actions: np.ndarray  # (batch, action_dim) - action to predict
    future_states: np.ndarray   # (batch, state_dim) - for waypoint loss


class D4RLDataset:
    """
    Dataset wrapper for D4RL that provides trajectory sampling for GCDT.

    Supports:
    - AntMaze tasks (goal-conditioned navigation)
    - Kitchen tasks (sequential manipulation)
    """

    def __init__(
        self,
        dataset: Dict[str, np.ndarray],
        env,
        context_len: int = 20,
        goal_sampling: str = "future",
        min_goal_horizon: int = 1,
        max_goal_horizon: int = 50,
        waypoint_horizon: int = 10,
        task_type: str = "antmaze",  # "antmaze" or "kitchen"
    ):
        """
        Args:
            dataset: D4RL dataset dict with 'observations', 'actions', 'terminals', etc.
            env: Gym environment for goal extraction
            context_len: Number of (state, action) pairs in context
            goal_sampling: "future" (sample from same trajectory) or "random"
            min_goal_horizon: Minimum steps ahead for goal
            max_goal_horizon: Maximum steps ahead for goal
            waypoint_horizon: Steps ahead for waypoint prediction target
            task_type: Type of D4RL task ("antmaze" or "kitchen")
        """
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.terminals = dataset.get("terminals", dataset.get("dones", np.zeros(len(self.observations))))
        self.timeouts = dataset.get("timeouts", np.zeros(len(self.observations)))

        # For antmaze, extract goal positions from observations
        # AntMaze obs format: [qpos, qvel, goal_xy] where goal is last 2 dims
        self.task_type = task_type
        self.env = env

        if task_type == "antmaze":
            # Goals are the target xy position (last 2 dims of observation)
            self.goal_dim = 2
            # Extract achieved positions (first 2 dims are xy position)
            self.achieved_goals = self.observations[:, :2]
        elif task_type == "kitchen":
            # Kitchen uses full state as goal
            self.goal_dim = self.observations.shape[-1]
            self.achieved_goals = self.observations
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.context_len = context_len
        self.goal_sampling = goal_sampling
        self.min_goal_horizon = min_goal_horizon
        self.max_goal_horizon = max_goal_horizon
        self.waypoint_horizon = waypoint_horizon

        self.state_dim = self.observations.shape[-1]
        self.action_dim = self.actions.shape[-1]

        # Build trajectory index for efficient sampling
        self._build_trajectory_index()

    def _build_trajectory_index(self):
        """Build index of trajectory boundaries for efficient sampling."""
        # Episode ends when terminal OR timeout
        episode_ends = np.logical_or(self.terminals, self.timeouts)
        end_indices = np.where(episode_ends)[0]

        self.trajectory_starts = [0]
        self.trajectory_ends = []

        for end_idx in end_indices:
            self.trajectory_ends.append(end_idx + 1)
            if end_idx + 1 < len(self.observations):
                self.trajectory_starts.append(end_idx + 1)

        # Handle case where last trajectory doesn't end
        if len(self.trajectory_ends) < len(self.trajectory_starts):
            self.trajectory_ends.append(len(self.observations))

        self.trajectory_starts = np.array(self.trajectory_starts)
        self.trajectory_ends = np.array(self.trajectory_ends)
        self.trajectory_lengths = self.trajectory_ends - self.trajectory_starts

        # Filter out trajectories that are too short
        min_len = self.context_len + max(self.max_goal_horizon, self.waypoint_horizon)
        valid_trajs = self.trajectory_lengths >= min_len
        self.trajectory_starts = self.trajectory_starts[valid_trajs]
        self.trajectory_ends = self.trajectory_ends[valid_trajs]
        self.trajectory_lengths = self.trajectory_lengths[valid_trajs]

        self.num_trajectories = len(self.trajectory_starts)

        if self.num_trajectories == 0:
            # Fallback: use shorter minimum length
            min_len = self.context_len + 1
            self.trajectory_starts = np.array([0])
            self.trajectory_ends = np.array([len(self.observations)])
            self.trajectory_lengths = self.trajectory_ends - self.trajectory_starts
            valid_trajs = self.trajectory_lengths >= min_len
            self.trajectory_starts = self.trajectory_starts[valid_trajs]
            self.trajectory_ends = self.trajectory_ends[valid_trajs]
            self.trajectory_lengths = self.trajectory_lengths[valid_trajs]
            self.num_trajectories = len(self.trajectory_starts)

        # Create sampling weights proportional to usable length
        usable_lengths = np.maximum(self.trajectory_lengths - min_len + 1, 1)
        self.traj_weights = usable_lengths / usable_lengths.sum()

        print(f"Built trajectory index: {self.num_trajectories} valid trajectories")
        if self.num_trajectories > 0:
            print(f"Trajectory length stats: min={self.trajectory_lengths.min()}, "
                  f"max={self.trajectory_lengths.max()}, mean={self.trajectory_lengths.mean():.1f}")

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> TrajectoryBatch:
        """
        Sample a batch of trajectory contexts with goals.
        """
        # Sample trajectories
        traj_indices = rng.choice(
            self.num_trajectories,
            size=batch_size,
            p=self.traj_weights
        )

        states_batch = []
        actions_batch = []
        goals_batch = []
        timesteps_batch = []
        target_actions_batch = []
        future_states_batch = []

        for traj_idx in traj_indices:
            traj_start = self.trajectory_starts[traj_idx]
            traj_end = self.trajectory_ends[traj_idx]
            traj_len = traj_end - traj_start

            # Sample starting point within trajectory
            max_start = max(0, traj_len - self.context_len - 1)
            start_offset = rng.integers(0, max_start + 1)

            # Extract context
            ctx_start = traj_start + start_offset
            ctx_end = min(ctx_start + self.context_len, traj_end)
            actual_ctx_len = ctx_end - ctx_start

            states = self.observations[ctx_start:ctx_end]
            actions = self.actions[ctx_start:ctx_end]

            # Pad if needed
            if actual_ctx_len < self.context_len:
                pad_len = self.context_len - actual_ctx_len
                states = np.concatenate([
                    np.tile(states[0:1], (pad_len, 1)),
                    states
                ], axis=0)
                actions = np.concatenate([
                    np.tile(actions[0:1], (pad_len, 1)),
                    actions
                ], axis=0)

            timesteps = np.arange(self.context_len)

            # Target action is the last action in context
            target_action = actions[-1]

            # Sample goal from future in same trajectory
            if self.goal_sampling == "future":
                max_horizon = min(self.max_goal_horizon, traj_end - ctx_end)
                if max_horizon > self.min_goal_horizon:
                    goal_horizon = rng.integers(self.min_goal_horizon, max_horizon + 1)
                else:
                    goal_horizon = max(1, max_horizon)
                goal_idx = min(ctx_end - 1 + goal_horizon, traj_end - 1)
                goal = self.achieved_goals[goal_idx]
            else:
                # Random goal from dataset
                goal_idx = rng.integers(0, len(self.achieved_goals))
                goal = self.achieved_goals[goal_idx]

            # Future state for waypoint loss
            waypoint_idx = min(ctx_end - 1 + self.waypoint_horizon, traj_end - 1)
            future_state = self.observations[waypoint_idx]

            states_batch.append(states)
            actions_batch.append(actions)
            goals_batch.append(goal)
            timesteps_batch.append(timesteps)
            target_actions_batch.append(target_action)
            future_states_batch.append(future_state)

        return TrajectoryBatch(
            states=np.stack(states_batch),
            actions=np.stack(actions_batch),
            goals=np.stack(goals_batch),
            timesteps=np.stack(timesteps_batch),
            target_actions=np.stack(target_actions_batch),
            future_states=np.stack(future_states_batch),
        )

    def __len__(self) -> int:
        """Return approximate number of samples."""
        return int((self.trajectory_lengths - self.context_len + 1).clip(min=1).sum())


class DataLoader:
    """Iterator that yields batches from D4RLDataset."""

    def __init__(
        self,
        dataset: D4RLDataset,
        batch_size: int,
        seed: int = 42,
        infinite: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.infinite = infinite
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[TrajectoryBatch]:
        while True:
            yield self.dataset.sample_batch(self.batch_size, self.rng)
            if not self.infinite:
                break

    def get_batch(self) -> TrajectoryBatch:
        """Get a single batch."""
        return self.dataset.sample_batch(self.batch_size, self.rng)


def get_task_type(dataset_name: str) -> str:
    """Determine task type from dataset name."""
    if "antmaze" in dataset_name:
        return "antmaze"
    elif "kitchen" in dataset_name:
        return "kitchen"
    else:
        # Default to treating as generic (full state as goal)
        return "kitchen"


def load_d4rl_dataset(
    dataset_name: str,
    context_len: int = 20,
    train_split: float = 0.9,
    **kwargs
) -> Tuple[D4RLDataset, D4RLDataset, Dict]:
    """
    Load D4RL dataset and create train/val datasets.

    Args:
        dataset_name: D4RL dataset name (e.g., "antmaze-umaze-v2")
        context_len: Number of (state, action) pairs in context
        train_split: Fraction of data for training
        **kwargs: Additional arguments for D4RLDataset

    Returns:
        train_dataset: D4RLDataset for training
        val_dataset: D4RLDataset for validation
        env_info: Dictionary with state_dim, action_dim, goal_dim, env
    """
    import gym
    import d4rl

    # Create environment
    env = gym.make(dataset_name)

    # Get dataset
    dataset = d4rl.qlearning_dataset(env)

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    task_type = get_task_type(dataset_name)

    # Split into train/val
    n_samples = len(dataset["observations"])
    n_train = int(n_samples * train_split)

    # Split by trajectory boundaries for cleaner separation
    terminals = dataset.get("terminals", dataset.get("dones", np.zeros(n_samples)))
    timeouts = dataset.get("timeouts", np.zeros(n_samples))
    episode_ends = np.where(np.logical_or(terminals, timeouts))[0]

    if len(episode_ends) > 1:
        # Find split point near train_split fraction
        split_idx = int(len(episode_ends) * train_split)
        split_point = episode_ends[split_idx] + 1
    else:
        split_point = n_train

    train_data = {
        k: v[:split_point] for k, v in dataset.items()
    }
    val_data = {
        k: v[split_point:] for k, v in dataset.items()
    }

    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {n_samples}, Train: {len(train_data['observations'])}, Val: {len(val_data['observations'])}")

    train_dataset = D4RLDataset(
        dataset=train_data,
        env=env,
        context_len=context_len,
        task_type=task_type,
        **kwargs
    )

    val_dataset = D4RLDataset(
        dataset=val_data,
        env=env,
        context_len=context_len,
        task_type=task_type,
        **kwargs
    )

    # Goal dim depends on task type
    if task_type == "antmaze":
        goal_dim = 2
    else:
        goal_dim = state_dim

    env_info = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "goal_dim": goal_dim,
        "env": env,
        "task_type": task_type,
    }

    return train_dataset, val_dataset, env_info


def batch_to_jax(batch: TrajectoryBatch) -> Dict[str, jnp.ndarray]:
    """Convert numpy batch to JAX arrays."""
    return {
        "states": jnp.array(batch.states),
        "actions": jnp.array(batch.actions),
        "goals": jnp.array(batch.goals),
        "timesteps": jnp.array(batch.timesteps),
        "target_actions": jnp.array(batch.target_actions),
        "future_states": jnp.array(batch.future_states),
    }
