"""Universal Transformer Goal-Conditioned Decision Transformer (UT-GCDT)."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
from functools import partial

from .components import (
    TokenEmbedding,
    TransformerBlock,
    SinusoidalEmbedding,
    ActionHead,
    WaypointHead,
    RecurrentGate,
)


class UTGCDT(nn.Module):
    """
    Universal Transformer for Goal-Conditioned Decision Transformers.
    
    Key features:
    - Weight-tied transformer block applied K times (Universal Transformer style)
    - Optional plan token for iterative refinement
    - Step embeddings to distinguish iterations
    - Waypoint auxiliary loss for deep supervision
    """
    # Dimensions
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    
    # Transformer config
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    # UT config
    num_iterations: int = 4  # K: number of times to apply transformer block
    use_step_embeddings: bool = True
    step_embedding_type: str = "learned"  # "learned" or "sinusoidal"

    # Gated recurrent loop (URM-style)
    # Instead of H_{k+1} = Block(H_k), uses:
    # H_{k+1} = (1-g) * H_k + g * Block(H_k)
    # where g = sigmoid(W_gate * [H_k, Block(H_k)])
    use_gated_loop: bool = False

    # Plan token
    use_plan_token: bool = True
    plan_token_bidirectional: bool = False
    plan_token_block_last_action: bool = True
    
    # Sequence config
    max_seq_len: int = 256
    
    # Auxiliary heads
    use_waypoint_head: bool = True
    
    def setup(self):
        # Token embeddings
        self.token_embed = TokenEmbedding(
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            use_plan_token=self.use_plan_token,
        )
        
        # Single transformer block (weight-tied across iterations)
        self.transformer_block = TransformerBlock(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
        )
        
        # Step embeddings for each iteration
        if self.use_step_embeddings:
            if self.step_embedding_type == "sinusoidal":
                self.step_embed = SinusoidalEmbedding(dim=self.hidden_dim)
            else:
                # Learned step embeddings
                self.step_embeddings = self.param(
                    "step_embeddings",
                    nn.initializers.normal(stddev=0.02),
                    (self.num_iterations, self.hidden_dim),
                )

        # Gated recurrent mechanism (optional)
        if self.use_gated_loop:
            self.recurrent_gate = RecurrentGate(hidden_dim=self.hidden_dim)

        # Final layer norm
        self.final_ln = nn.LayerNorm()
        
        # Prediction heads
        self.action_head = ActionHead(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )
        
        if self.use_waypoint_head:
            self.waypoint_head = WaypointHead(
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
            )
    
    def get_step_embedding(self, step: int, batch_size: int, seq_len: int) -> jnp.ndarray:
        """Get step embedding for iteration k."""
        if not self.use_step_embeddings:
            return jnp.zeros((batch_size, seq_len, self.hidden_dim))
        
        if self.step_embedding_type == "sinusoidal":
            # Same step for all positions in sequence
            steps = jnp.full((batch_size,), step)
            step_emb = self.step_embed(steps)  # (batch, hidden_dim)
            step_emb = step_emb[:, None, :]  # (batch, 1, hidden_dim)
            step_emb = jnp.broadcast_to(step_emb, (batch_size, seq_len, self.hidden_dim))
        else:
            # Learned step embeddings
            step_emb = self.step_embeddings[step]  # (hidden_dim,)
            step_emb = jnp.broadcast_to(step_emb, (batch_size, seq_len, self.hidden_dim))
        
        return step_emb
    
    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,      # (batch, seq_len, state_dim)
        actions: jnp.ndarray,     # (batch, seq_len, action_dim)
        goals: jnp.ndarray,       # (batch, state_dim)
        timesteps: jnp.ndarray,   # (batch, seq_len)
        num_iterations: Optional[int] = None,  # Override for test-time scaling
        deterministic: bool = True,
        return_intermediates: bool = False,  # Return intermediate states for deep supervision
    ) -> Dict[str, Any]:
        """
        Forward pass with K iterations of the transformer block.
        
        Returns:
            Dictionary containing:
            - action_pred: Predicted action (batch, action_dim)
            - waypoint_preds: List of waypoint predictions at each iteration
            - plan_states: List of plan token states at each iteration
            - hidden_states: Final hidden states (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = states.shape
        K = num_iterations if num_iterations is not None else self.num_iterations
        
        # Embed tokens
        hidden, plan_token_idx = self.token_embed(states, actions, goals, timesteps)
        total_seq_len = hidden.shape[1]
        plan_offset = 1 if self.use_plan_token else 0

        # Build attention mask for plan token bidirectional attention
        attn_mask = None
        if self.use_plan_token and self.plan_token_bidirectional:
            attn_mask = jnp.tril(jnp.ones((total_seq_len, total_seq_len)))
            attn_mask = attn_mask.at[0, :].set(1)
            if self.plan_token_block_last_action:
                last_action_idx = plan_offset + 2 * seq_len
                attn_mask = attn_mask.at[0, last_action_idx].set(0)
            attn_mask = attn_mask[None, None, :, :]

        # Index of last state token for action prediction
        # Sequence: [PLAN] [GOAL] s_0 a_0 s_1 a_1 ... s_t a_t
        last_state_idx = plan_offset + 1 + 2 * (seq_len - 1)

        # Store intermediate outputs for deep supervision
        waypoint_preds = []
        plan_states = []
        gate_values = []  # Store gate values for analysis
        intermediate_action_preds = []  # Action predictions at each step k (shared head)

        # Apply transformer block K times (Universal Transformer loop)
        for k in range(K):
            # Add step embedding
            step_emb = self.get_step_embedding(k, batch_size, total_seq_len)
            hidden_with_step = hidden + step_emb

            # Apply transformer block (same weights each iteration)
            proposal = self.transformer_block(
                hidden_with_step,
                mask=attn_mask,
                deterministic=deterministic
            )

            # Gated update vs naive update
            if self.use_gated_loop:
                # Gated: H_{k+1} = (1-g) * H_k + g * proposal
                # Gate sees both old state and new proposal
                hidden, gate = self.recurrent_gate(hidden, proposal)
                if return_intermediates:
                    gate_values.append(gate)
            else:
                # Naive: H_{k+1} = proposal (transformer block has internal residuals)
                hidden = proposal

            # Deep supervision: predict action at each step k using SHARED head
            # This enables summed loss across all iterations (like independent heads for baseline)
            if return_intermediates or k < K - 1:
                # Apply layer norm before action prediction
                step_hidden = self.final_ln(hidden)
                last_state_hidden = step_hidden[:, last_state_idx, :]
                step_action_pred = self.action_head(last_state_hidden)
                intermediate_action_preds.append(step_action_pred)

            # Extract plan token state and predict waypoint (for deep supervision)
            if self.use_plan_token and self.use_waypoint_head:
                plan_state = hidden[:, plan_token_idx, :]  # (batch, hidden_dim)
                plan_states.append(plan_state)

                if return_intermediates or k == K - 1:
                    waypoint_pred = self.waypoint_head(plan_state)
                    waypoint_preds.append(waypoint_pred)
        
        # Final layer norm (already applied in loop for action predictions)
        hidden = self.final_ln(hidden)
        last_state_hidden = hidden[:, last_state_idx, :]
        action_pred = self.action_head(last_state_hidden)
        intermediate_action_preds.append(action_pred)

        # Final action prediction is the last intermediate prediction
        action_pred = intermediate_action_preds[-1]

        return {
            "action_pred": action_pred,
            "intermediate_action_preds": intermediate_action_preds,  # For deep supervision
            "waypoint_preds": waypoint_preds,
            "plan_states": plan_states,
            "hidden_states": hidden,
            "plan_token_idx": plan_token_idx,
            "gate_values": gate_values,  # Gate activations for analysis
        }


class MultiBlockUTGCDT(nn.Module):
    """
    Multi-Block Universal Transformer for Goal-Conditioned Decision Transformers.

    Key features:
    - Multiple unique transformer blocks (e.g., 2 blocks)
    - Blocks are cycled through repeatedly (e.g., 6 cycles)
    - Total updates = num_blocks * num_cycles (e.g., 2 * 6 = 12)
    - Hybrid between full weight tying (1 block) and no tying (L blocks)
    - Optional plan token, step embeddings, gated loop, etc.
    """
    # Dimensions
    state_dim: int
    action_dim: int
    hidden_dim: int = 256

    # Transformer config
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1

    # Multi-block cycling config
    num_blocks: int = 2  # Number of unique transformer blocks
    num_cycles: int = 6  # Number of cycles through all blocks
    # Total iterations = num_blocks * num_cycles

    # Step embeddings
    use_step_embeddings: bool = True
    step_embedding_type: str = "sinusoidal"

    # Gated recurrent loop
    use_gated_loop: bool = False

    # Plan token
    use_plan_token: bool = True
    plan_token_bidirectional: bool = False
    plan_token_block_last_action: bool = True

    # Sequence config
    max_seq_len: int = 256

    # Auxiliary heads
    use_waypoint_head: bool = True

    def setup(self):
        # Token embeddings
        self.token_embed = TokenEmbedding(
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            use_plan_token=self.use_plan_token,
        )

        # Multiple transformer blocks (each with unique weights)
        self.transformer_blocks = [
            TransformerBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )
            for i in range(self.num_blocks)
        ]

        # Total iterations for step embeddings
        self.total_iterations = self.num_blocks * self.num_cycles

        # Step embeddings for each iteration (total = num_blocks * num_cycles)
        if self.use_step_embeddings:
            if self.step_embedding_type == "sinusoidal":
                self.step_embed = SinusoidalEmbedding(dim=self.hidden_dim)
            else:
                # Learned step embeddings for each unique step
                self.step_embeddings = self.param(
                    "step_embeddings",
                    nn.initializers.normal(stddev=0.02),
                    (self.total_iterations, self.hidden_dim),
                )

        # Gated recurrent mechanism (optional)
        if self.use_gated_loop:
            self.recurrent_gate = RecurrentGate(hidden_dim=self.hidden_dim)

        # Final layer norm
        self.final_ln = nn.LayerNorm()

        # Prediction heads
        self.action_head = ActionHead(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )

        if self.use_waypoint_head:
            self.waypoint_head = WaypointHead(
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
            )

    def get_step_embedding(self, step: int, batch_size: int, seq_len: int) -> jnp.ndarray:
        """Get step embedding for iteration k."""
        if not self.use_step_embeddings:
            return jnp.zeros((batch_size, seq_len, self.hidden_dim))

        if self.step_embedding_type == "sinusoidal":
            steps = jnp.full((batch_size,), step)
            step_emb = self.step_embed(steps)
            step_emb = step_emb[:, None, :]
            step_emb = jnp.broadcast_to(step_emb, (batch_size, seq_len, self.hidden_dim))
        else:
            # Learned step embeddings
            step_emb = self.step_embeddings[step]
            step_emb = jnp.broadcast_to(step_emb, (batch_size, seq_len, self.hidden_dim))

        return step_emb

    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        goals: jnp.ndarray,
        timesteps: jnp.ndarray,
        num_cycles: Optional[int] = None,  # Override for test-time scaling
        deterministic: bool = True,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass cycling through blocks num_cycles times.

        Returns:
            Dictionary containing:
            - action_pred: Predicted action (batch, action_dim)
            - waypoint_preds: List of waypoint predictions at each iteration
            - plan_states: List of plan token states at each iteration
            - hidden_states: Final hidden states (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = states.shape
        cycles = num_cycles if num_cycles is not None else self.num_cycles
        total_steps = self.num_blocks * cycles

        # Embed tokens
        hidden, plan_token_idx = self.token_embed(states, actions, goals, timesteps)
        total_seq_len = hidden.shape[1]
        plan_offset = 1 if self.use_plan_token else 0

        # Build attention mask for plan token bidirectional attention
        attn_mask = None
        if self.use_plan_token and self.plan_token_bidirectional:
            attn_mask = jnp.tril(jnp.ones((total_seq_len, total_seq_len)))
            attn_mask = attn_mask.at[0, :].set(1)
            if self.plan_token_block_last_action:
                last_action_idx = plan_offset + 2 * seq_len
                attn_mask = attn_mask.at[0, last_action_idx].set(0)
            attn_mask = attn_mask[None, None, :, :]

        # Index of last state token for action prediction
        last_state_idx = plan_offset + 1 + 2 * (seq_len - 1)

        # Store intermediate outputs for deep supervision
        waypoint_preds = []
        plan_states = []
        gate_values = []
        intermediate_action_preds = []

        # Cycle through blocks: block_0, block_1, block_0, block_1, ...
        step_counter = 0
        for cycle in range(cycles):
            for block_idx in range(self.num_blocks):
                # Add step embedding (unique for each step in the cycle)
                step_emb = self.get_step_embedding(step_counter, batch_size, total_seq_len)
                hidden_with_step = hidden + step_emb

                # Apply transformer block
                proposal = self.transformer_blocks[block_idx](
                    hidden_with_step,
                    mask=attn_mask,
                    deterministic=deterministic,
                )

                # Gated update vs naive update
                if self.use_gated_loop:
                    hidden, gate = self.recurrent_gate(hidden, proposal)
                    if return_intermediates:
                        gate_values.append(gate)
                else:
                    hidden = proposal

                # Deep supervision: predict action at each step using shared head
                if return_intermediates or step_counter < total_steps - 1:
                    step_hidden = self.final_ln(hidden)
                    last_state_hidden = step_hidden[:, last_state_idx, :]
                    step_action_pred = self.action_head(last_state_hidden)
                    intermediate_action_preds.append(step_action_pred)

                # Extract plan token state and predict waypoint
                if self.use_plan_token and self.use_waypoint_head:
                    plan_state = hidden[:, plan_token_idx, :]
                    plan_states.append(plan_state)

                    if return_intermediates or step_counter == total_steps - 1:
                        waypoint_pred = self.waypoint_head(plan_state)
                        waypoint_preds.append(waypoint_pred)

                step_counter += 1

        # Final layer norm
        hidden = self.final_ln(hidden)
        last_state_hidden = hidden[:, last_state_idx, :]
        action_pred = self.action_head(last_state_hidden)
        intermediate_action_preds.append(action_pred)

        action_pred = intermediate_action_preds[-1]

        return {
            "action_pred": action_pred,
            "intermediate_action_preds": intermediate_action_preds,
            "waypoint_preds": waypoint_preds,
            "plan_states": plan_states,
            "hidden_states": hidden,
            "plan_token_idx": plan_token_idx,
            "gate_values": gate_values,
        }


class GCDT(nn.Module):
    """
    Standard Goal-Conditioned Decision Transformer (baseline).

    Uses stacked transformer layers with untied weights.
    """
    # Dimensions
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    
    # Transformer config
    num_heads: int = 4
    num_layers: int = 4  # L: number of stacked layers
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    # Plan token (for fair comparison)
    use_plan_token: bool = False
    plan_token_bidirectional: bool = False
    plan_token_block_last_action: bool = True
    
    # Sequence config
    max_seq_len: int = 256
    
    def setup(self):
        # Token embeddings
        self.token_embed = TokenEmbedding(
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            use_plan_token=self.use_plan_token,
        )

        # Stacked transformer layers (untied weights)
        self.transformer_layers = [
            TransformerBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f"layer_{i}"
            )
            for i in range(self.num_layers)
        ]

        # Final layer norm
        self.final_ln = nn.LayerNorm()

        # Layer norms for intermediate supervision (one per non-final layer)
        self.intermediate_lns = [
            nn.LayerNorm(name=f"intermediate_ln_{i}")
            for i in range(self.num_layers - 1)
        ]

        # Independent action heads for each layer (per protocol: "steel-manning" the baseline)
        # Each layer l has its own projection head H_l to predict actions
        self.action_heads = [
            ActionHead(
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                name=f"action_head_{i}"
            )
            for i in range(self.num_layers)
        ]
    
    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        goals: jnp.ndarray,
        timesteps: jnp.ndarray,
        deterministic: bool = True,
        return_intermediates: bool = False,  # For deep supervision with independent heads
    ) -> Dict[str, Any]:
        """Forward pass through stacked transformer layers."""
        batch_size, seq_len, _ = states.shape

        # Embed tokens
        hidden, plan_token_idx = self.token_embed(states, actions, goals, timesteps)
        total_seq_len = hidden.shape[1]
        plan_offset = 1 if self.use_plan_token else 0

        # Build attention mask
        attn_mask = None
        if self.use_plan_token and self.plan_token_bidirectional:
            attn_mask = jnp.tril(jnp.ones((total_seq_len, total_seq_len)))
            attn_mask = attn_mask.at[0, :].set(1)
            if self.plan_token_block_last_action:
                last_action_idx = plan_offset + 1 + 2 * seq_len - 1
                attn_mask = attn_mask.at[0, last_action_idx].set(0)
            attn_mask = attn_mask[None, None, :, :]

        # Index of last state token for action prediction
        last_state_idx = plan_offset + 1 + 2 * (seq_len - 1)

        # Store intermediate action predictions for deep supervision
        intermediate_action_preds = []

        # Apply transformer layers sequentially (different weights each layer)
        for i, layer in enumerate(self.transformer_layers):
            hidden = layer(hidden, mask=attn_mask, deterministic=deterministic)

            # Independent action head per layer (for deep supervision)
            if return_intermediates or i == len(self.transformer_layers) - 1:
                layer_hidden = self.final_ln(hidden) if i == len(self.transformer_layers) - 1 else self.intermediate_lns[i](hidden)
                last_state_hidden = layer_hidden[:, last_state_idx, :]
                action_pred = self.action_heads[i](last_state_hidden)
                intermediate_action_preds.append(action_pred)

        # Final layer norm
        hidden = self.final_ln(hidden)

        # Final action prediction (from the last layer's head)
        action_pred = intermediate_action_preds[-1]

        return {
            "action_pred": action_pred,
            "intermediate_action_preds": intermediate_action_preds,
            "hidden_states": hidden,
            "plan_token_idx": plan_token_idx,
        }


def create_model(config) -> nn.Module:
    """Factory function to create model from config."""
    if config.model.use_weight_tying:
        # Check if using multi-block configuration
        num_blocks = getattr(config.model, 'num_blocks', 1)
        if num_blocks > 1:
            # Multi-block UT-GCDT: multiple unique blocks cycled repeatedly
            return MultiBlockUTGCDT(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dim=config.model.hidden_dim,
                num_heads=config.model.num_heads,
                mlp_ratio=config.model.mlp_ratio,
                dropout_rate=config.model.dropout_rate,
                num_blocks=config.model.num_blocks,
                num_cycles=config.model.num_cycles,
                use_step_embeddings=config.model.use_step_embeddings,
                step_embedding_type=config.model.step_embedding_type,
                use_gated_loop=config.model.use_gated_loop,
                use_plan_token=config.model.use_plan_token,
                plan_token_bidirectional=config.model.plan_token_bidirectional,
                plan_token_block_last_action=config.model.plan_token_block_last_action,
                max_seq_len=config.model.max_seq_len,
                use_waypoint_head=config.aux.use_waypoint_loss,
            )
        else:
            # Standard UT-GCDT: single block repeated K times
            return UTGCDT(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dim=config.model.hidden_dim,
                num_heads=config.model.num_heads,
                mlp_ratio=config.model.mlp_ratio,
                dropout_rate=config.model.dropout_rate,
                num_iterations=config.model.num_iterations,
                use_step_embeddings=config.model.use_step_embeddings,
                step_embedding_type=config.model.step_embedding_type,
                use_gated_loop=config.model.use_gated_loop,
                use_plan_token=config.model.use_plan_token,
                plan_token_bidirectional=config.model.plan_token_bidirectional,
                plan_token_block_last_action=config.model.plan_token_block_last_action,
                max_seq_len=config.model.max_seq_len,
                use_waypoint_head=config.aux.use_waypoint_loss,
            )
    else:
        return GCDT(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            mlp_ratio=config.model.mlp_ratio,
            dropout_rate=config.model.dropout_rate,
            use_plan_token=config.model.use_plan_token,
            plan_token_bidirectional=config.model.plan_token_bidirectional,
            plan_token_block_last_action=config.model.plan_token_block_last_action,
            max_seq_len=config.model.max_seq_len,
        )
