"""Default configuration for UT-GCDT experiments."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Embedding dimensions
    hidden_dim: int = 256

    # Transformer block
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1

    # For standard GCDT (untied) - V3 Protocol: L=12 layers
    num_layers: int = 12

    # For UT-GCDT (tied) - V3 Protocol: K=12 iterations
    num_iterations: int = 12  # K in the paper
    use_weight_tying: bool = True

    # For Multi-Block UT-GCDT: multiple unique blocks cycled repeatedly
    # Total updates = num_blocks * num_cycles
    # E.g., num_blocks=2, num_cycles=6 -> 12 total updates
    num_blocks: int = 1  # Number of unique transformer blocks
    num_cycles: int = 1  # Number of times to cycle through all blocks

    # Step embeddings for UT iterations
    use_step_embeddings: bool = True
    step_embedding_type: str = "learned"  # "learned" or "sinusoidal"

    # Gated recurrent loop (URM-style)
    # H_{k+1} = (1-g) * H_k + g * Block(H_k)
    # Helps with gradient flow and allows model to control update magnitude
    use_gated_loop: bool = False

    # Plan token
    use_plan_token: bool = True
    plan_token_bidirectional: bool = False
    plan_token_block_last_action: bool = True
    
    # Sequence modeling
    max_seq_len: int = 256  # Maximum context length
    

@dataclass
class AuxiliaryConfig:
    """Auxiliary loss configuration."""
    # Waypoint prediction
    use_waypoint_loss: bool = True
    waypoint_horizon: int = 10  # H steps ahead
    waypoint_loss_weight: float = 0.1
    
    # Deep supervision: apply aux loss at each UT iteration
    deep_supervision: bool = True
    # If not deep supervision, only apply at final iteration
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 0.0001
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batch size
    batch_size: int = 256
    
    # Context window for trajectory sampling
    context_len: int = 20  # Number of (s, a) pairs in context
    
    # Evaluation
    eval_every: int = 10000
    eval_episodes: int = 50
    
    # Logging
    log_every: int = 1000
    save_every: int = 50000
    
    # Reproducibility
    seed: int = 42
    

@dataclass
class DataConfig:
    """Dataset configuration."""
    # Dataset type: "d4rl" or "ogbench"
    dataset_type: str = "d4rl"

    # V3 Protocol D4RL datasets:
    #   - Debug/Sanity: antmaze-umaze-v2 (target: >50% success)
    #   - Primary (Stitching): antmaze-medium-play-v2 (target: beat baseline by >10%)
    #   - Secondary: kitchen-mixed-v0 (long-horizon manipulation)
    dataset_name: str = "antmaze-umaze-v2"

    # For OGBench (legacy)
    dataset_dir: str = field(
        default_factory=lambda: os.environ.get("OGBENCH_DATASET_DIR", "./ogbench_data")
    )

    # Goal sampling strategy
    # "future": sample goals from future in same trajectory
    # "random": sample random goals from dataset
    goal_sampling: str = "future"

    # Future goal horizon range (for "future" sampling)
    min_goal_horizon: int = 1
    max_goal_horizon: int = 50

    # Hindsight Experience Replay (HER) relabeling probability
    # 80% of time: use trajectory's FINAL achieved state as goal (enables stitching)
    # 20% of time: use original goal sampling (maintains global context)
    her_relabel_prob: float = 0.8


@dataclass
class EvalConfig:
    """Evaluation configuration for test-time scaling experiments."""
    # Test with different numbers of UT iterations (V3: K=12 training)
    test_iterations: List[int] = field(default_factory=lambda: [12, 24])

    # Evaluate on different goal distance buckets
    goal_distance_buckets: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    

@dataclass
class Config:
    """Full experiment configuration."""
    # Experiment name
    exp_name: str = "ut_gcdt_default"
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    aux: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Output directory
    output_dir: str = "./outputs"


# Preset configurations for experiments
def get_gcdt_baseline_config() -> Config:
    """
    Standard GCDT baseline (untied layers, no plan token).

    Per protocol ("Steel-Manning"):
    - Independent action heads: Each layer l has its own projection head H_l
    - Deep supervision on all heads
    - This gives the baseline maximum flexibility to learn layer-specific representations
    """
    config = Config(exp_name="gcdt_baseline")
    config.model.use_weight_tying = False
    config.model.use_plan_token = False
    config.model.use_step_embeddings = False
    config.aux.use_waypoint_loss = False
    # Enable deep supervision for independent heads
    config.aux.deep_supervision = True
    return config


def get_ut_gcdt_config() -> Config:
    """
    U-GCDT with weight tying only.

    Per protocol:
    - Shared action head (one H used for all steps k)
    - Sinusoidal step embeddings (no learned params for step counter)
    """
    config = Config(exp_name="ut_gcdt_tied")
    config.model.use_weight_tying = True
    config.model.use_plan_token = False
    config.model.use_step_embeddings = True
    config.model.step_embedding_type = "sinusoidal"  # Fixed, no param leak
    config.aux.use_waypoint_loss = False
    return config


def get_ut_gcdt_plan_config() -> Config:
    """
    U-GCDT with plan token.

    Per protocol:
    - Shared action head
    - Sinusoidal step embeddings (no param leak)
    - Plan token for iterative refinement
    """
    config = Config(exp_name="ut_gcdt_plan")
    config.model.use_weight_tying = True
    config.model.use_plan_token = True
    config.model.use_step_embeddings = True
    config.model.step_embedding_type = "sinusoidal"  # Fixed, no param leak
    config.aux.use_waypoint_loss = False
    return config


def get_ut_gcdt_full_config() -> Config:
    """
    U-GCDT Full: Plan token + Waypoint loss + Deep supervision.

    Per protocol:
    - Shared action head
    - Sinusoidal step embeddings (no param leak)
    - Plan token for iterative refinement
    - Waypoint auxiliary loss with deep supervision
    """
    config = Config(exp_name="ut_gcdt_full")
    config.model.use_weight_tying = True
    config.model.use_plan_token = True
    config.model.use_step_embeddings = True
    config.model.step_embedding_type = "sinusoidal"  # Fixed, no param leak
    config.aux.use_waypoint_loss = True
    config.aux.deep_supervision = True
    return config


def get_ut_gcdt_gated_config() -> Config:
    """
    U-GCDT with Gated Recurrent Loop (URM-style).

    Key change: Uses gated update instead of naive residual:
    H_{k+1} = (1-g) * H_k + g * Block(H_k)

    Benefits:
    - Better gradient flow through iterations
    - Model can control update magnitude per position
    - Stabilizes training for deeper iteration counts
    - Inspired by GRU/LSTM gating and Universal Recurrent Memory

    Config: Weight tying + Plan token + Gated loop (no waypoint loss)
    """
    config = Config(exp_name="ut_gcdt_gated")
    config.model.use_weight_tying = True
    config.model.use_plan_token = True
    config.model.use_step_embeddings = True
    config.model.step_embedding_type = "sinusoidal"  # Fixed, no param leak
    config.model.use_gated_loop = True
    config.aux.use_waypoint_loss = False
    return config


def get_ut_gcdt_gated_full_config() -> Config:
    """
    U-GCDT Full + Gated Recurrent Loop.

    Combines all features:
    - Weight tying (Universal Transformer)
    - Plan token for iterative refinement
    - Gated recurrent loop (URM-style)
    - Waypoint auxiliary loss with deep supervision
    """
    config = Config(exp_name="ut_gcdt_gated_full")
    config.model.use_weight_tying = True
    config.model.use_plan_token = True
    config.model.use_step_embeddings = True
    config.model.step_embedding_type = "sinusoidal"
    config.model.use_gated_loop = True
    config.aux.use_waypoint_loss = True
    config.aux.deep_supervision = True
    return config


def get_ut_gcdt_multiblock_config() -> Config:
    """
    Multi-Block U-GCDT: 2 unique blocks cycled 6 times (12 total updates).

    This is a hybrid between:
    - Full weight tying (1 block, K iterations) - maximum parameter efficiency
    - No weight tying (L independent layers) - maximum expressivity

    Key idea: Use 2 unique transformer blocks and cycle through them 6 times.
    This allows:
    - More expressivity than single-block UT (different weights per block)
    - Better parameter efficiency than 12 independent layers
    - Potential for learned block specialization (e.g., refinement vs prediction)

    Config: 2 blocks x 6 cycles = 12 total updates
    - Plan token for iterative refinement
    - Sinusoidal step embeddings (no param leak)
    - Waypoint auxiliary loss with deep supervision
    """
    config = Config(exp_name="ut_gcdt_multiblock_2x6")
    config.model.use_weight_tying = True  # Indicates UT-style (not GCDT baseline)
    config.model.num_blocks = 2  # 2 unique transformer blocks
    config.model.num_cycles = 6  # Cycle through blocks 6 times
    config.model.num_iterations = 12  # Total: 2 * 6 = 12 (for compatibility)
    config.model.use_plan_token = True
    config.model.use_step_embeddings = True
    config.model.step_embedding_type = "sinusoidal"
    config.model.use_gated_loop = True  # Enable gating for better gradient flow
    config.aux.use_waypoint_loss = True
    config.aux.deep_supervision = True
    return config