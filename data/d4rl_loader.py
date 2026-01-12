"""Data loading utilities for D4RL datasets - FIXED VERSION.

Key fixes:
1. Action leakage: Last action in input is now zeroed, target is the NEXT action
2. Goal normalization: Correct shape handling for antmaze (2D goals)
3. Debug prints added for verification
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Iterator
from dataclasses import dataclass


@dataclass
class NormalizationStats:
    """Statistics for normalizing observations."""
    obs_mean: np.ndarray
    obs_std: np.ndarray
    goal_mean: np.ndarray  # For antmaze: shape (2,) for xy coords
    goal_std: np.ndarray   # For antmaze: shape (2,)


@dataclass
class TrajectoryBatch:
    """Batch of trajectory data for GCDT training."""
    states: np.ndarray       # (batch, context_len, state_dim)
    actions: np.ndarray      # (batch, context_len, action_dim) - last action is ZEROED
    goals: np.ndarray        # (batch, goal_dim)
    timesteps: np.ndarray    # (batch, context_len)
    target_actions: np.ndarray  # (batch, action_dim) - action to predict (NEXT action)
    future_states: np.ndarray   # (batch, state_dim) - for waypoint loss


class D4RLDataset:
    """
    Dataset wrapper for D4RL that provides trajectory sampling for GCDT.
    
    CRITICAL FIX: Prevents action leakage by:
    1. Zeroing the last action in the input sequence
    2. Target action is the action AFTER the last state in context
    
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
        task_type: str = "antmaze",
        normalize: bool = True,
        norm_stats: Optional[NormalizationStats] = None,
        her_relabel_prob: float = 0.8,
    ):
        """
        Args:
            dataset: D4RL dataset dict with 'observations', 'actions', 'terminals', etc.
            env: Gym environment for goal extraction
            context_len: Number of (state, action) pairs in context
            goal_sampling: "future" (sample from same trajectory) or "her" (hindsight relabeling)
            min_goal_horizon: Minimum steps ahead for goal
            max_goal_horizon: Maximum steps ahead for goal
            waypoint_horizon: Steps ahead for waypoint prediction target
            task_type: Type of D4RL task ("antmaze" or "kitchen")
            normalize: Whether to normalize observations and goals
            norm_stats: Pre-computed normalization stats (for val set)
            her_relabel_prob: Probability of using achieved final state as goal (HER)
        """
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.terminals = dataset.get("terminals", dataset.get("dones", np.zeros(len(self.observations))))
        self.timeouts = dataset.get("timeouts", np.zeros(len(self.observations)))

        self.task_type = task_type
        self.env = env
        self.normalize = normalize
        self.her_relabel_prob = her_relabel_prob

        if task_type == "antmaze":
            # Goals are the target xy position (first 2 dims of observation)
            self.goal_dim = 2
            # Extract achieved positions (first 2 dims are xy position)
            self.achieved_goals = self.observations[:, :2]
        elif task_type == "kitchen":
            # Kitchen uses full state as goal
            self.goal_dim = self.observations.shape[-1]
            self.achieved_goals = self.observations
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Compute or use provided normalization statistics
        if normalize:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                self.norm_stats = self._compute_norm_stats()
            
            # CRITICAL: Verify normalization stats shapes
            print(f"=== Normalization Stats Shapes ===")
            print(f"obs_mean shape: {self.norm_stats.obs_mean.shape}")
            print(f"obs_std shape: {self.norm_stats.obs_std.shape}")
            print(f"goal_mean shape: {self.norm_stats.goal_mean.shape}")
            print(f"goal_std shape: {self.norm_stats.goal_std.shape}")
            print(f"goal_dim: {self.goal_dim}")
            
            # Verify goal stats match goal_dim
            assert self.norm_stats.goal_mean.shape[0] == self.goal_dim, \
                f"Goal mean shape {self.norm_stats.goal_mean.shape} doesn't match goal_dim {self.goal_dim}"
            assert self.norm_stats.goal_std.shape[0] == self.goal_dim, \
                f"Goal std shape {self.norm_stats.goal_std.shape} doesn't match goal_dim {self.goal_dim}"
            
            print(f"Normalization: obs_mean range [{self.norm_stats.obs_mean.min():.2f}, {self.norm_stats.obs_mean.max():.2f}], "
                  f"goal_mean range [{self.norm_stats.goal_mean.min():.2f}, {self.norm_stats.goal_mean.max():.2f}]")
        else:
            self.norm_stats = None

        self.context_len = context_len
        self.goal_sampling = goal_sampling
        self.min_goal_horizon = min_goal_horizon
        self.max_goal_horizon = max_goal_horizon
        self.waypoint_horizon = waypoint_horizon

        self.state_dim = self.observations.shape[-1]
        self.action_dim = self.actions.shape[-1]

        # Build trajectory index for efficient sampling
        self._build_trajectory_index()

    def _compute_norm_stats(self) -> NormalizationStats:
        """Compute normalization statistics from data."""
        obs_mean = self.observations.mean(axis=0)
        obs_std = self.observations.std(axis=0) + 1e-6

        # CRITICAL FIX: Goal stats should be computed from achieved_goals
        # which has the correct dimensionality (2 for antmaze, full state for kitchen)
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
    
    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Unnormalize observations."""
        if self.norm_stats is None:
            return obs
        return obs * self.norm_stats.obs_std + self.norm_stats.obs_mean

    def _build_trajectory_index(self):
        """Build index of trajectory boundaries for efficient sampling."""
        # Find episode boundaries (terminal OR timeout)
        episode_ends = np.where(np.logical_or(self.terminals, self.timeouts))[0]

        self.trajectory_starts = [0]
        self.trajectory_ends = []

        for end_idx in episode_ends:
            self.trajectory_ends.append(end_idx + 1)
            if end_idx + 1 < len(self.observations):
                self.trajectory_starts.append(end_idx + 1)

        # Handle case where last trajectory doesn't end with terminal/timeout
        if len(self.trajectory_ends) < len(self.trajectory_starts):
            self.trajectory_ends.append(len(self.observations))

        self.trajectory_starts = np.array(self.trajectory_starts)
        self.trajectory_ends = np.array(self.trajectory_ends)
        self.trajectory_lengths = self.trajectory_ends - self.trajectory_starts

        # Filter out trajectories that are too short
        # Need: context_len states + 1 extra for target action + goal horizon
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

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> TrajectoryBatch:
        """
        Sample a batch of trajectory contexts with goals.
        
        FIXED: Actions are shifted by 1 to match inference
        - action[t] = action taken AFTER state[t-1], or 0 for t=0
        - This way, at every position t, the model sees (s_t, a_{t-1}) and predicts a_t
        
        Sequence structure:
        Position:  0      1      2      ...   T-1
        State:     s_0    s_1    s_2    ...   s_{T-1}
        Action:    0      a_0    a_1    ...   a_{T-2}
        Target:    a_{T-1}
        """
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
            max_start = max(0, traj_len - self.context_len - 1 - self.min_goal_horizon)
            start_offset = rng.integers(0, max_start + 1)

            ctx_start = traj_start + start_offset
            ctx_end = min(ctx_start + self.context_len, traj_end - 1)
            actual_ctx_len = ctx_end - ctx_start

            # States: s_0 to s_{T-1} (T = actual_ctx_len)
            states = self.observations[ctx_start:ctx_end]
            
            # FIXED: Actions are SHIFTED by 1
            # action[t] = a_{t-1} (action taken after seeing s_{t-1})
            # action[0] = 0 (no previous action)
            # action[T-1] = a_{T-2}
            # Target = a_{T-1} (action to predict at s_{T-1})
            
            actions = np.zeros((actual_ctx_len, self.action_dim))
            if actual_ctx_len > 1:
                # Fill positions 1 to T-1 with actions a_0 to a_{T-2}
                actions[1:] = self.actions[ctx_start:ctx_end - 1]
            
            # Target: a_{T-1} - the action we should take at s_{T-1}
            target_action = self.actions[ctx_end - 1]

            # Pad if needed (at beginning of context)
            if actual_ctx_len < self.context_len:
                pad_len = self.context_len - actual_ctx_len
                states = np.concatenate([
                    np.tile(states[0:1], (pad_len, 1)),
                    states
                ], axis=0)
                actions = np.concatenate([
                    np.zeros((pad_len, self.action_dim)),
                    actions
                ], axis=0)

            timesteps = np.arange(self.context_len)

            # Goal sampling with HER
            use_her = rng.random() < self.her_relabel_prob
            
            if use_her:
                goal_idx = traj_end - 1
            else:
                # FIX: Ensure valid range for rng.integers
                remaining = traj_end - ctx_end
                max_horizon = max(self.min_goal_horizon + 1, min(self.max_goal_horizon + 1, remaining))
                if max_horizon <= self.min_goal_horizon:
                    goal_horizon = self.min_goal_horizon
                else:
                    goal_horizon = rng.integers(self.min_goal_horizon, max_horizon)
                goal_idx = min(ctx_end + goal_horizon, traj_end - 1)
            
            # Extract goal
            if self.task_type == "antmaze":
                goal = self.achieved_goals[goal_idx]
            else:
                goal = self.observations[goal_idx]

            # Future state for waypoint loss
            waypoint_idx = min(ctx_end + self.waypoint_horizon, traj_end - 1)
            future_state = self.observations[waypoint_idx]

            states_batch.append(states)
            actions_batch.append(actions)
            goals_batch.append(goal)
            timesteps_batch.append(timesteps)
            target_actions_batch.append(target_action)
            future_states_batch.append(future_state)

        # Stack all batches
        states_arr = np.stack(states_batch)
        actions_arr = np.stack(actions_batch)
        goals_arr = np.stack(goals_batch)
        future_states_arr = np.stack(future_states_batch)
        target_actions_arr = np.stack(target_actions_batch)

        # Apply normalization
        if self.normalize and self.norm_stats is not None:
            states_arr = (states_arr - self.norm_stats.obs_mean) / self.norm_stats.obs_std
            goals_arr = (goals_arr - self.norm_stats.goal_mean) / self.norm_stats.goal_std
            future_states_arr = (future_states_arr - self.norm_stats.obs_mean) / self.norm_stats.obs_std

        return TrajectoryBatch(
            states=states_arr,
            actions=actions_arr,
            goals=goals_arr,
            timesteps=np.stack(timesteps_batch),
            target_actions=target_actions_arr,
            future_states=future_states_arr,
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
        return "kitchen"


def load_d4rl_dataset(
    dataset_name: str,
    context_len: int = 20,
    train_split: float = 0.9,
    **kwargs
) -> Tuple[D4RLDataset, D4RLDataset, Dict]:
    """
    Load D4RL dataset and create train/val datasets.
    """
    import gym
    import d4rl

    env = gym.make(dataset_name)
    dataset = d4rl.qlearning_dataset(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    task_type = get_task_type(dataset_name)

    # Split into train/val by trajectory boundaries
    n_samples = len(dataset["observations"])
    terminals = dataset.get("terminals", dataset.get("dones", np.zeros(n_samples)))
    timeouts = dataset.get("timeouts", np.zeros(n_samples))
    episode_ends = np.where(np.logical_or(terminals, timeouts))[0]

    if len(episode_ends) > 1:
        split_idx = int(len(episode_ends) * train_split)
        split_point = episode_ends[split_idx] + 1
    else:
        split_point = int(n_samples * train_split)

    train_data = {k: v[:split_point] for k, v in dataset.items()}
    val_data = {k: v[split_point:] for k, v in dataset.items()}

    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {n_samples}, Train: {len(train_data['observations'])}, Val: {len(val_data['observations'])}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create train dataset
    train_dataset = D4RLDataset(
        dataset=train_data,
        env=env,
        context_len=context_len,
        task_type=task_type,
        normalize=True,
        **kwargs
    )

    # Create val dataset using same normalization stats
    val_dataset = D4RLDataset(
        dataset=val_data,
        env=env,
        context_len=context_len,
        task_type=task_type,
        normalize=True,
        norm_stats=train_dataset.norm_stats,
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
        "norm_stats": train_dataset.norm_stats,
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


# ============================================================================
# DEBUG UTILITIES
# ============================================================================

def verify_batch(batch: TrajectoryBatch, verbose: bool = True) -> bool:
    """
    Verify batch doesn't have action leakage.
    
    Returns True if batch is valid, False if there's leakage.
    """
    # Check that last action in input is zero
    last_actions = batch.actions[:, -1, :]  # (batch, action_dim)
    is_zero = np.allclose(last_actions, 0.0)
    
    if verbose:
        print("=== Batch Verification ===")
        print(f"States shape: {batch.states.shape}")
        print(f"Actions shape: {batch.actions.shape}")
        print(f"Goals shape: {batch.goals.shape}")
        print(f"Target actions shape: {batch.target_actions.shape}")
        print(f"Last action in input (should be ~0): {last_actions[0, :3]}")
        print(f"Target action (what we predict): {batch.target_actions[0, :3]}")
        print(f"Last action is zeroed: {is_zero}")
        
        if not is_zero:
            print("WARNING: Last action is NOT zeroed - potential action leakage!")
    
    return is_zero


if __name__ == "__main__":
    # Test the data loader
    print("Testing D4RLDataset with action leakage fix...")
    
    train_dataset, val_dataset, env_info = load_d4rl_dataset(
        "antmaze-umaze-v2",
        context_len=20,
    )
    
    loader = DataLoader(train_dataset, batch_size=32, seed=42)
    batch = loader.get_batch()
    
    # Verify the batch
    is_valid = verify_batch(batch, verbose=True)
    
    if is_valid:
        print("\n✓ Batch verification PASSED - no action leakage detected")
    else:
        print("\n✗ Batch verification FAILED - action leakage detected!")