"""Default configuration for UT-GCDT experiments."""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Embedding dimensions
    hidden_dim: int = 256
    
    # Transformer block
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    # For standard GCDT (untied)
    num_layers: int = 4
    
    # For UT-GCDT (tied)
    num_iterations: int = 4  # K in the paper
    use_weight_tying: bool = True
    
    # Step embeddings for UT iterations
    use_step_embeddings: bool = True
    step_embedding_type: str = "learned"  # "learned" or "sinusoidal"
    
    # Plan token
    use_plan_token: bool = True
    
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
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batch size
    batch_size: int = 256
    
    # Context window for trajectory sampling
    context_len: int = 20  # Number of (s, a) pairs in context
    
    # Evaluation
    eval_every: int = 5000
    eval_episodes: int = 100
    
    # Logging
    log_every: int = 100
    save_every: int = 10000
    
    # Reproducibility
    seed: int = 42
    

@dataclass  
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "antmaze-medium-stitch-v0"
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


@dataclass
class EvalConfig:
    """Evaluation configuration for test-time scaling experiments."""
    # Test with different numbers of UT iterations
    test_iterations: List[int] = field(default_factory=lambda: [4, 6, 8])
    
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
    """Standard GCDT baseline (untied layers, no plan token)."""
    config = Config(exp_name="gcdt_baseline")
    config.model.use_weight_tying = False
    config.model.use_plan_token = False
    config.model.use_step_embeddings = False
    config.aux.use_waypoint_loss = False
    return config


def get_ut_gcdt_config() -> Config:
    """UT-GCDT with weight tying only."""
    config = Config(exp_name="ut_gcdt_tied")
    config.model.use_weight_tying = True
    config.model.use_plan_token = False
    config.aux.use_waypoint_loss = False
    return config


def get_ut_gcdt_plan_config() -> Config:
    """UT-GCDT with plan token."""
    config = Config(exp_name="ut_gcdt_plan")
    config.model.use_weight_tying = True
    config.model.use_plan_token = True
    config.aux.use_waypoint_loss = False
    return config


def get_ut_gcdt_full_config() -> Config:
    """UT-GCDT with plan token and waypoint loss (full model)."""
    config = Config(exp_name="ut_gcdt_full")
    config.model.use_weight_tying = True
    config.model.use_plan_token = True
    config.aux.use_waypoint_loss = True
    config.aux.deep_supervision = True
    return config
