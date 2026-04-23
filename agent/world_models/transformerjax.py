"""
JAX/Equinox implementation of Transformer architecture.
Converted from transformer.py (PyTorch version).
"""

from typing import Optional, List, Tuple, NamedTuple
import math

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from .kv_cachingjax import KeysValuesState, create_kv_cache, update_kv_cache, get_kv_cache


def get_sinusoid_encoding_table(n_positions: int, d_hid: int) -> jnp.ndarray:
    """Generate sinusoidal position encoding table."""
    def get_position_angle_vec(position):
        return jnp.array([
            position / jnp.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
    
    sinusoid_table = jnp.array([get_position_angle_vec(pos_i) for pos_i in range(n_positions)])
    sinusoid_table = sinusoid_table.at[:, 0::2].set(jnp.sin(sinusoid_table[:, 0::2]))
    sinusoid_table = sinusoid_table.at[:, 1::2].set(jnp.cos(sinusoid_table[:, 1::2]))
    return jnp.expand_dims(sinusoid_table, 0)


class FeedForward(eqx.Module):
    """Feed-forward network with GELU activation."""
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 2)
        self.fc1 = eqx.nn.Linear(embed_dim, hidden_dim, key=keys[0])
        self.fc2 = eqx.nn.Linear(hidden_dim, embed_dim, key=keys[1])
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x: jnp.ndarray, *, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        if key is not None:
            x = self.dropout(x, key=key)
        # If key is None, skip dropout (useful when vmap doesn't provide per-element keys)
        x = self.fc2(x)
        return x


class SelfAttention(eqx.Module):
    """Multi-head self-attention with optional KV caching."""
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    proj_dropout: eqx.nn.Dropout

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, *, key: jax.random.PRNGKey):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        keys = jax.random.split(key, 2)
        self.qkv = eqx.nn.Linear(embed_dim, 3 * embed_dim, key=keys[0])
        self.proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.proj_dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        x: jnp.ndarray,
        kv_state: Optional[KeysValuesState] = None,
        layer_idx: int = 0,
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, Optional[KeysValuesState]]:
        B, T, C = x.shape
        
        # Apply qkv linear with vmap over batch and sequence dimensions
        qkv = jax.vmap(jax.vmap(self.qkv))(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        new_kv_state = kv_state
        if kv_state is not None:
            prev_k, prev_v = get_kv_cache(kv_state, layer_idx)
            prev_T = prev_k.shape[2]
            new_kv_state = update_kv_cache(kv_state, layer_idx, k, v, prev_T)
            k = jnp.concatenate([prev_k, k], axis=2)
            v = jnp.concatenate([prev_v, v], axis=2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        if attn_mask is not None:
            # Expand mask to (B, 1, T, T) for broadcasting with (B, H, T, T)
            if attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            attn = jnp.where(attn_mask, attn, -1e9)
        
        attn = jax.nn.softmax(attn, axis=-1)
        if key is not None:
            key, subkey = jax.random.split(key)
            attn = self.attn_dropout(attn, key=subkey)
        
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        # Apply proj linear with vmap over batch and sequence dimensions
        out = jax.vmap(jax.vmap(self.proj))(out)
        if key is not None:
            out = self.proj_dropout(out, key=key)
        
        return out, new_kv_state


class Block(eqx.Module):
    """Transformer block with pre-norm architecture."""
    ln1: eqx.nn.LayerNorm
    attn: SelfAttention
    ln2: eqx.nn.LayerNorm
    ff: FeedForward

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 2)
        self.ln1 = eqx.nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, dropout, key=keys[0])
        self.ln2 = eqx.nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim, dropout, key=keys[1])

    def __call__(
        self,
        x: jnp.ndarray,
        kv_state: Optional[KeysValuesState] = None,
        layer_idx: int = 0,
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, Optional[KeysValuesState]]:
        keys = jax.random.split(key, 2) if key is not None else (None, None)
        
        # x shape: (B, T, C) - need to vmap over both batch and sequence for LayerNorm
        normed = jax.vmap(jax.vmap(self.ln1))(x)  # (B, T, C)
        attn_out, new_kv_state = self.attn(normed, kv_state, layer_idx, attn_mask, key=keys[0])
        x = x + attn_out
        
        normed = jax.vmap(jax.vmap(self.ln2))(x)  # (B, T, C)
        # Note: We pass key=None to disable dropout during training
        # since vmap with dropout requires per-element keys which is complex
        ff_out = jax.vmap(jax.vmap(lambda xi: self.ff(xi, key=None)))(normed)
        x = x + ff_out
        
        return x, new_kv_state


class Transformer(eqx.Module):
    """Transformer encoder with optional KV caching."""
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)
    
    blocks: List[Block]
    # pos_emb removed - position embedding handled by TransRewEndModel
    ln_out: eqx.nn.LayerNorm

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        keys = jax.random.split(key, num_layers)
        self.blocks = [
            Block(embed_dim, num_heads, hidden_dim, dropout, key=k)
            for k in keys
        ]
        # pos_emb removed - position embedding handled by TransRewEndModel
        self.ln_out = eqx.nn.LayerNorm(embed_dim)

    def __call__(
        self,
        x: jnp.ndarray,
        kv_state: Optional[KeysValuesState] = None,
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, Optional[KeysValuesState]]:
        # Position embedding already added by TransRewEndModel
        
        new_kv_state = kv_state
        for i, block in enumerate(self.blocks):
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x, new_kv_state = block(x, new_kv_state, i, attn_mask, key=subkey)
        
        x = jax.vmap(jax.vmap(self.ln_out))(x)
        return x, new_kv_state

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValuesState:
        """Generate empty keys and values for KV caching (matches PyTorch API)."""
        return create_kv_cache(
            num_samples=n, max_tokens=max_tokens,
            num_heads=self.num_heads, embed_dim=self.embed_dim // self.num_heads
        )


class PerAttention(eqx.Module):
    """Perceiver-style cross-attention module."""
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    
    q_proj: eqx.nn.Linear
    kv_proj: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    proj_dropout: eqx.nn.Dropout

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, *, key: jax.random.PRNGKey):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        keys = jax.random.split(key, 3)
        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.kv_proj = eqx.nn.Linear(embed_dim, 2 * embed_dim, key=keys[1])
        self.proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.proj_dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        latent: jnp.ndarray,
        context: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        B, T_q, C = latent.shape
        _, T_kv, _ = context.shape
        
        q = self.q_proj(latent).reshape(B, T_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, T_kv, 2, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        if attn_mask is not None:
            # Expand mask to (B, 1, T, T) for broadcasting with (B, H, T, T)
            if attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            attn = jnp.where(attn_mask, attn, -1e9)
        
        attn = jax.nn.softmax(attn, axis=-1)
        if key is not None:
            key, subkey = jax.random.split(key)
            attn = self.attn_dropout(attn, key=subkey)
        
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T_q, C)
        out = self.proj(out)
        if key is not None:
            out = self.proj_dropout(out, key=key)
        
        return out


class PerceiverBlock(eqx.Module):
    """Perceiver block with cross-attention and feed-forward."""
    ln1: eqx.nn.LayerNorm
    cross_attn: PerAttention
    ln2: eqx.nn.LayerNorm
    ff: FeedForward

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 2)
        self.ln1 = eqx.nn.LayerNorm(embed_dim)
        self.cross_attn = PerAttention(embed_dim, num_heads, dropout, key=keys[0])
        self.ln2 = eqx.nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim, dropout, key=keys[1])

    def __call__(
        self,
        latent: jnp.ndarray,
        context: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        keys = jax.random.split(key, 2) if key is not None else (None, None)
        
        normed = jax.vmap(self.ln1)(latent)
        attn_out = self.cross_attn(normed, context, attn_mask, key=keys[0])
        latent = latent + attn_out
        
        normed = jax.vmap(self.ln2)(latent)
        ff_out = jax.vmap(lambda xi, k=keys[1]: self.ff(xi, key=k))(normed)
        latent = latent + ff_out
        
        return latent


class Perceiver(eqx.Module):
    """Perceiver architecture with learnable latents."""
    embed_dim: int = eqx.field(static=True)
    num_latents: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    
    latents: Float[Array, "1 num_latents embed_dim"]
    blocks: List[PerceiverBlock]
    ln_out: eqx.nn.LayerNorm

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_latents: int,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        key: jax.random.PRNGKey
    ):
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.num_layers = num_layers
        
        keys = jax.random.split(key, num_layers + 1)
        self.latents = jax.random.normal(keys[0], (1, num_latents, embed_dim)) * 0.02
        self.blocks = [
            PerceiverBlock(embed_dim, num_heads, hidden_dim, dropout, key=k)
            for k in keys[1:]
        ]
        self.ln_out = eqx.nn.LayerNorm(embed_dim)

    def __call__(
        self,
        context: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        B = context.shape[0]
        latent = jnp.tile(self.latents, (B, 1, 1))
        
        for block in self.blocks:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            latent = block(latent, context, attn_mask, key=subkey)
        
        latent = jax.vmap(self.ln_out)(latent)
        return latent
