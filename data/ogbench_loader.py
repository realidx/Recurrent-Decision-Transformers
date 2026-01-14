"""Data loading utilities for OGBench datasets."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Iterator
from dataclasses import dataclass


@dataclass
class TrajectoryBatch:
    """Batch of trajectory data for GCDT training."""
    states: np.ndarray       # (batch, context_len, state_dim)
    actions: np.ndarray      # (batch, context_len, action_dim)
    goals: np.ndarray        # (batch, state_dim)
    timesteps: np.ndarray    # (batch, context_len)
    target_actions: np.ndarray  # (batch, action_dim) - action to predict
    future_states: np.ndarray   # (batch, state_dim) - for waypoint loss


@dataclass
class NormalizationStats:
    """Statistics for normalizing observations."""
    obs_mean: np.ndarray
    obs_std: np.ndarray
    goal_mean: np.ndarray
    goal_std: np.ndarray


class OGBenchDataset:
    """
    Dataset wrapper for OGBench that provides trajectory sampling for GCDT.
    
    Handles:
    - Loading OGBench datasets
    - Sampling trajectory contexts
    - Goal sampling (future states from same trajectory)
    - Future state sampling for waypoint auxiliary loss
    """
    
    def __init__(
        self,
        dataset: Dict[str, np.ndarray],
        context_len: int = 20,
        goal_sampling: str = "future",
        min_goal_horizon: int = 1,
        max_goal_horizon: int = 50,
        waypoint_horizon: int = 10,
        task_type: str = "antmaze",
        normalize: bool = True,
        norm_stats: Optional[NormalizationStats] = None,
    ):
        """
        Args:
            dataset: OGBench dataset dict with 'observations', 'actions', 'terminals', 'valids'
            context_len: Number of (state, action) pairs in context
            goal_sampling: "future" (sample from same trajectory) or "random"
            min_goal_horizon: Minimum steps ahead for goal
            max_goal_horizon: Maximum steps ahead for goal  
            waypoint_horizon: Steps ahead for waypoint prediction target
        """
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.terminals = dataset.get("terminals", np.zeros(len(self.observations)))
        self.valids = dataset.get("valids", np.ones(len(self.observations)))
        
        self.context_len = context_len
        self.goal_sampling = goal_sampling
        self.min_goal_horizon = min_goal_horizon
        self.max_goal_horizon = max_goal_horizon
        self.waypoint_horizon = waypoint_horizon
        self.task_type = task_type
        self.normalize = normalize
        
        self.state_dim = self.observations.shape[-1]
        self.action_dim = self.actions.shape[-1]

        if task_type == "antmaze":
            self.goal_dim = 2
            self.achieved_goals = self.observations[:, :2]
        else:
            self.goal_dim = self.state_dim
            self.achieved_goals = self.observations

        if normalize:
            self.norm_stats = norm_stats or self._compute_norm_stats()
        else:
            self.norm_stats = None
        
        # Build trajectory index for efficient sampling
        self._build_trajectory_index()
        
    def _build_trajectory_index(self):
        """Build index of trajectory boundaries for efficient sampling."""
        # Find trajectory boundaries from terminals
        terminal_indices = np.where(self.terminals == 1)[0]
        
        self.trajectory_starts = [0]
        self.trajectory_ends = []
        
        for term_idx in terminal_indices:
            self.trajectory_ends.append(term_idx + 1)
            if term_idx + 1 < len(self.observations):
                self.trajectory_starts.append(term_idx + 1)
        
        # Handle case where last trajectory doesn't end with terminal
        if len(self.trajectory_ends) < len(self.trajectory_starts):
            self.trajectory_ends.append(len(self.observations))
            
        self.trajectory_starts = np.array(self.trajectory_starts)
        self.trajectory_ends = np.array(self.trajectory_ends)
        self.trajectory_lengths = self.trajectory_ends - self.trajectory_starts
        
        # Filter out trajectories that are too short
        min_len = self.context_len + 1 + self.min_goal_horizon
        valid_trajs = self.trajectory_lengths >= min_len
        self.trajectory_starts = self.trajectory_starts[valid_trajs]
        self.trajectory_ends = self.trajectory_ends[valid_trajs]
        self.trajectory_lengths = self.trajectory_lengths[valid_trajs]
        
        self.num_trajectories = len(self.trajectory_starts)
        
        # Create sampling weights proportional to usable length
        usable_lengths = self.trajectory_lengths - min_len + 1
        self.traj_weights = usable_lengths / usable_lengths.sum()
        
        print(f"Built trajectory index: {self.num_trajectories} valid trajectories")
        print(f"Trajectory length stats: min={self.trajectory_lengths.min()}, "
              f"max={self.trajectory_lengths.max()}, mean={self.trajectory_lengths.mean():.1f}")

    def _compute_norm_stats(self) -> NormalizationStats:
        """Compute normalization statistics from observations and achieved goals."""
        obs_mean = self.observations.mean(axis=0)
        obs_std = self.observations.std(axis=0) + 1e-6

        goal_mean = self.achieved_goals.mean(axis=0)
        goal_std = self.achieved_goals.std(axis=0) + 1e-6

        return NormalizationStats(
            obs_mean=obs_mean,
            obs_std=obs_std,
            goal_mean=goal_mean,
            goal_std=goal_std,
        )

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using stored statistics."""
        if self.norm_stats is None:
            return obs
        return (obs - self.norm_stats.obs_mean) / self.norm_stats.obs_std

    def normalize_goal(self, goal: np.ndarray) -> np.ndarray:
        """Normalize goals using stored statistics."""
        if self.norm_stats is None:
            return goal
        return (goal - self.norm_stats.goal_mean) / self.norm_stats.goal_std
    
    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> TrajectoryBatch:
        """
        Sample a batch of trajectory contexts with goals.
        
        Returns:
            TrajectoryBatch with:
            - states: (batch, context_len, state_dim)
            - actions: (batch, context_len, action_dim) 
            - goals: (batch, state_dim)
            - timesteps: (batch, context_len)
            - target_actions: (batch, action_dim)
            - future_states: (batch, state_dim)
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
            # Need room for context + target action + goal horizon
            max_start = max(0, traj_len - self.context_len - 1 - self.min_goal_horizon)
            start_offset = rng.integers(0, max_start + 1)

            # Extract context
            ctx_start = traj_start + start_offset
            ctx_end = ctx_start + self.context_len
            
            states = self.observations[ctx_start:ctx_end]
            timesteps = np.arange(self.context_len)

            # Actions are shifted by one to prevent leakage (last action zeroed)
            actions = np.zeros((self.context_len, self.action_dim))
            if self.context_len > 1:
                actions[1:] = self.actions[ctx_start:ctx_end - 1]

            # Target action is the action after the last state in context
            target_action = self.actions[ctx_end - 1]
            
            # Sample goal from future in same trajectory
            if self.goal_sampling == "future":
                goal_horizon = rng.integers(
                    self.min_goal_horizon, 
                    min(self.max_goal_horizon + 1, traj_end - ctx_end + 1)
                )
                goal_idx = ctx_end - 1 + goal_horizon  # -1 because ctx_end is exclusive
                goal_full = self.observations[min(goal_idx, traj_end - 1)]
            else:
                # Random goal from dataset
                goal_idx = rng.integers(0, len(self.observations))
                goal_full = self.observations[goal_idx]

            if self.task_type == "antmaze":
                goal = goal_full[:2]
            else:
                goal = goal_full
            
            # Future state for waypoint loss
            waypoint_idx = min(ctx_end - 1 + self.waypoint_horizon, traj_end - 1)
            future_state = self.observations[waypoint_idx]

            if self.normalize:
                states = self.normalize_obs(states)
                goal = self.normalize_goal(goal)
                future_state = self.normalize_obs(future_state)
            
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
        """Return approximate number of samples (sum of usable positions)."""
        return int((self.trajectory_lengths - self.context_len - self.max_goal_horizon + 1).sum())


class DataLoader:
    """Iterator that yields batches from OGBenchDataset."""
    
    def __init__(
        self,
        dataset: OGBenchDataset,
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


def load_ogbench_dataset(
    dataset_name: str,
    dataset_dir: str = "~/.ogbench/data",
    context_len: int = 20,
    **kwargs
) -> Tuple[OGBenchDataset, OGBenchDataset, Dict]:
    """
    Load OGBench dataset and create train/val datasets.
    
    Returns:
        train_dataset: OGBenchDataset for training
        val_dataset: OGBenchDataset for validation
        env_info: Dictionary with state_dim, action_dim, etc.
    """
    import ogbench
    
    env, train_data, val_data = ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=dataset_dir,
    )
    
    # Get dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    task_type = "antmaze" if "antmaze" in dataset_name else "generic"
    goal_dim = 2 if task_type == "antmaze" else state_dim

    train_dataset = OGBenchDataset(
        dataset=train_data,
        context_len=context_len,
        task_type=task_type,
        **kwargs
    )

    val_dataset = OGBenchDataset(
        dataset=val_data,
        context_len=context_len,
        task_type=task_type,
        norm_stats=train_dataset.norm_stats,
        **kwargs
    )
    
    env_info = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "goal_dim": goal_dim,
        "task_type": task_type,
        "norm_stats": train_dataset.norm_stats,
        "env": env,
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
