"""Shared model components for GCDT and UT-GCDT."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
import math


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional/step embeddings."""
    dim: int
    max_len: int = 1000
    
    @nn.compact
    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            positions: Integer positions of shape (batch, seq_len) or (batch,)
        Returns:
            Embeddings of shape (batch, seq_len, dim) or (batch, dim)
        """
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb_scale)
        
        # Handle both 1D and 2D position inputs
        if positions.ndim == 1:
            positions = positions[:, None]  # (batch, 1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        emb = positions[..., None] * emb[None, None, :]  # (batch, seq_len, half_dim)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)  # (batch, seq_len, dim)
        
        if squeeze_output:
            emb = emb.squeeze(1)
            
        return emb


class TokenEmbedding(nn.Module):
    """Embedding layer for states, actions, and special tokens."""
    hidden_dim: int
    max_seq_len: int
    state_dim: int
    action_dim: int
    use_plan_token: bool = True
    
    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,      # (batch, seq_len, state_dim)
        actions: jnp.ndarray,     # (batch, seq_len, action_dim)  
        goals: jnp.ndarray,       # (batch, state_dim)
        timesteps: jnp.ndarray,   # (batch, seq_len)
    ) -> Tuple[jnp.ndarray, int]:
        """
        Embed and concatenate tokens into sequence.
        
        Returns:
            embeddings: (batch, total_seq_len, hidden_dim)
            plan_token_idx: Index of plan token in sequence (0 if used, -1 if not)
        """
        batch_size, seq_len, _ = states.shape
        
        # Project states, actions, goals to hidden_dim
        state_embed = nn.Dense(self.hidden_dim, name="state_embed")(states)
        action_embed = nn.Dense(self.hidden_dim, name="action_embed")(actions)
        goal_embed = nn.Dense(self.hidden_dim, name="goal_embed")(goals)  # (batch, hidden_dim)
        
        # Positional embeddings
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_seq_len, self.hidden_dim)
        )
        
        # Token type embeddings (state=0, action=1, goal=2, plan=3)
        token_type_embed = self.param(
            "token_type_embed",
            nn.initializers.normal(stddev=0.02),
            (4, self.hidden_dim)
        )
        
        # Build sequence: [PLAN] [GOAL] s_0 a_0 s_1 a_1 ... s_t a_t
        # or without plan: [GOAL] s_0 a_0 s_1 a_1 ... s_t a_t
        
        tokens = []
        token_types = []
        
        # Plan token (learnable)
        if self.use_plan_token:
            plan_token = self.param(
                "plan_token",
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.hidden_dim)
            )
            plan_token = jnp.broadcast_to(plan_token, (batch_size, 1, self.hidden_dim))
            tokens.append(plan_token)
            token_types.append(jnp.full((batch_size, 1), 3))  # type 3 = plan
            plan_token_idx = 0
        else:
            plan_token_idx = -1

        # Goal token - ensure proper 3D shape (batch, 1, hidden_dim)
        if goal_embed.ndim == 2:
            goal_token = jnp.expand_dims(goal_embed, axis=1)
        else:
            # Already 3D, just ensure middle dim is 1
            goal_token = goal_embed[:, :1, :]
        tokens.append(goal_token)
        token_types.append(jnp.full((batch_size, 1), 2))  # type 2 = goal

        # Interleave states and actions: s_0 a_0 s_1 a_1 ...
        for t in range(seq_len):
            tokens.append(state_embed[:, t:t+1, :])
            token_types.append(jnp.full((batch_size, 1), 0))  # type 0 = state
            tokens.append(action_embed[:, t:t+1, :])
            token_types.append(jnp.full((batch_size, 1), 1))  # type 1 = action
        
        # Concatenate all tokens
        embeddings = jnp.concatenate(tokens, axis=1)  # (batch, total_len, hidden_dim)
        token_types_concat = jnp.concatenate(token_types, axis=1)  # (batch, total_len)

        total_len = embeddings.shape[1]

        # Ensure token_types matches embeddings length (handle any edge cases)
        token_types_concat = token_types_concat[:, :total_len]

        # Add positional embeddings
        embeddings = embeddings + pos_embed[:, :total_len, :]

        # Add token type embeddings
        type_embeds = token_type_embed[token_types_concat]  # (batch, total_len, hidden_dim)
        embeddings = embeddings + type_embeds
        
        return embeddings, plan_token_idx


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    num_heads: int
    hidden_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            x: Input of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask (batch, 1, seq_len, seq_len)
            deterministic: Whether to apply dropout
        Returns:
            Output of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads
        
        # QKV projection
        qkv = nn.Dense(3 * self.hidden_dim, name="qkv")(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scale = head_dim ** -0.5
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        
        # Use caller-provided mask if given; otherwise default to causal.
        if mask is None:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            mask = mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
        attn = jnp.where(mask == 0, -1e9, attn)
        
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)
        
        # Attention output
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        out = nn.Dense(self.hidden_dim, name="out_proj")(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=deterministic)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    hidden_dim: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        mlp_dim = int(self.hidden_dim * self.mlp_ratio)
        
        x = nn.Dense(mlp_dim, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim, name="fc2")(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer block (attention + FFN with pre-norm)."""
    num_heads: int
    hidden_dim: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        # Pre-norm attention
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            name="attn"
        )(x, mask=mask, deterministic=deterministic)
        x = residual + x
        
        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = FeedForward(
            hidden_dim=self.hidden_dim,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="ffn"
        )(x, deterministic=deterministic)
        x = residual + x
        
        return x


class RecurrentGate(nn.Module):
    """
    Gated recurrent update mechanism for Universal Transformer.

    Instead of naive residual: H_{k+1} = Block(H_k)
    Uses sigmoid-weighted mixing: H_{k+1} = (1-g) * H_k + g * Block(H_k)

    This allows the model to control information flow:
    - g ≈ 0: preserve previous state (keep memory)
    - g ≈ 1: fully update to new proposal (write new thoughts)

    Inspired by GRU/LSTM gating and Universal Recurrent Memory (URM).
    """
    hidden_dim: int

    @nn.compact
    def __call__(
        self,
        h_prev: jnp.ndarray,      # Previous hidden state (batch, seq_len, hidden_dim)
        h_proposal: jnp.ndarray,  # New proposal from transformer block
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute gated update.

        Args:
            h_prev: Previous hidden state
            h_proposal: Proposed new state from transformer block

        Returns:
            h_next: Updated hidden state
            gate: Gate values for analysis/visualization (batch, seq_len, hidden_dim)
        """
        # Concatenate previous state and proposal for gate computation
        # Gate sees both old and new to decide how much to update
        gate_input = jnp.concatenate([h_prev, h_proposal], axis=-1)

        # Project to gate logits
        gate_logits = nn.Dense(
            self.hidden_dim,
            name="gate_proj",
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(gate_input)

        # Sigmoid activation to get values in [0, 1]
        gate = jax.nn.sigmoid(gate_logits)

        # Gated update: soft blend between old and new
        h_next = (1 - gate) * h_prev + gate * h_proposal

        return h_next, gate


class ActionHead(nn.Module):
    """Action prediction head."""
    action_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Hidden state of shape (batch, hidden_dim)
        Returns:
            Action prediction of shape (batch, action_dim)
        """
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.action_dim, name="fc2")(x)
        return x


class WaypointHead(nn.Module):
    """Waypoint (future state) prediction head."""
    state_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Hidden state (e.g., plan token) of shape (batch, hidden_dim)
        Returns:
            Future state prediction of shape (batch, state_dim)
        """
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.state_dim, name="fc2")(x)
        return x