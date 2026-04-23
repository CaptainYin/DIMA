"""
JAX/Equinox implementation of Slicer and related modules.
Converted from slicer.py (PyTorch version).
"""

from typing import List, Tuple, Callable
import math

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int

from .transformerjax import get_sinusoid_encoding_table


class Slicer(eqx.Module):
    """Base slicer for computing indices from block masks."""
    block_size: int = eqx.field(static=True)
    num_kept_tokens: int = eqx.field(static=True)
    indices: Int[Array, "max_indices"]
    max_blocks: int = eqx.field(static=True)

    def __init__(self, max_blocks: int, block_mask: jnp.ndarray):
        self.block_size = block_mask.shape[0]
        self.num_kept_tokens = int(block_mask.sum())
        self.max_blocks = max_blocks
        kept_indices = jnp.where(block_mask)[0]
        kept_indices = jnp.tile(kept_indices, max_blocks)
        offsets = jnp.repeat(jnp.arange(max_blocks), self.num_kept_tokens)
        self.indices = kept_indices + self.block_size * offsets

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> jnp.ndarray:
        """Compute slice indices for the given step range."""
        total_steps = num_steps + prev_steps
        num_blocks = math.ceil(total_steps / self.block_size)
        indices = self.indices[:num_blocks * self.num_kept_tokens]
        mask = jnp.logical_and(prev_steps <= indices, indices < total_steps)
        valid_indices = jnp.where(mask, indices - prev_steps, -1)
        return valid_indices[valid_indices >= 0]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Head(eqx.Module):
    """Head module with slicing."""
    slicer: Slicer
    head_module: eqx.Module

    def __init__(self, max_blocks: int, block_mask: jnp.ndarray, head_module: eqx.Module):
        self.slicer = Slicer(max_blocks, block_mask)
        self.head_module = head_module

    def __call__(self, x: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        slice_indices = self.slicer.compute_slice(num_steps, prev_steps)
        x_sliced = x[:, slice_indices]
        return self.head_module(x_sliced)


class SpecialHead(eqx.Module):
    """Head with concatenated features."""
    slicer: Slicer
    head_module: eqx.Module

    def __init__(self, max_blocks: int, block_mask: jnp.ndarray, head_module: eqx.Module):
        self.slicer = Slicer(max_blocks, block_mask)
        self.head_module = head_module

    def __call__(self, x: jnp.ndarray, perattn_feat: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        slice_indices = self.slicer.compute_slice(num_steps, prev_steps)
        x_sliced = x[:, slice_indices]
        feat = jnp.concatenate([x_sliced, perattn_feat], axis=-1)
        return self.head_module(feat)


class AgentWiseHead(eqx.Module):
    """Head with agent-wise sinusoidal embeddings."""
    slicer: Slicer
    head_module: eqx.Module
    num_agents: int = eqx.field(static=True)
    agent_id_emb: Float[Array, "1 num_agents embed_dim"]

    def __init__(self, max_blocks: int, block_mask: jnp.ndarray, head_module: eqx.Module, num_agents: int, embed_dim: int):
        self.slicer = Slicer(max_blocks, block_mask)
        self.head_module = head_module
        self.num_agents = num_agents
        self.agent_id_emb = get_sinusoid_encoding_table(num_agents, embed_dim)

    def __call__(self, x: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        slice_indices = self.slicer.compute_slice(num_steps, prev_steps)
        x_sliced = x[:, slice_indices]
        x_expanded = jnp.expand_dims(x_sliced, axis=2)
        x_expanded = jnp.tile(x_expanded, (1, 1, self.num_agents, 1))
        agent_emb = self.agent_id_emb[:, :self.num_agents]
        x_with_agent = x_expanded + jnp.expand_dims(agent_emb, axis=1)
        return self.head_module(x_with_agent)


class DiscreteDist(eqx.Module):
    """MLP for discrete categorical distributions."""
    n_categoricals: int = eqx.field(static=True)
    n_classes: int = eqx.field(static=True)
    layers: List[eqx.Module]

    def __init__(self, in_dim: int, n_categoricals: int, n_classes: int, hidden_size: int, *, key: jax.random.PRNGKey):
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_dim, hidden_size, key=keys[0]),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            eqx.nn.Linear(hidden_size, n_classes * n_categoricals, key=keys[2]),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.layers[0](x)
        x = jax.nn.relu(x)
        x = self.layers[1](x)
        x = jax.nn.relu(x)
        x = self.layers[2](x)
        shape = x.shape[:-1] + (self.n_categoricals, self.n_classes)
        return x.reshape(shape)


class Embedder(eqx.Module):
    """Token embedding with slicing."""
    embedding_dim: int = eqx.field(static=True)
    embedding_tables: List[eqx.nn.Embedding]
    slicers: List[Slicer]

    def __init__(self, max_blocks: int, block_masks: List[jnp.ndarray], embedding_tables: List[eqx.nn.Embedding]):
        assert len(block_masks) == len(embedding_tables)
        self.embedding_dim = embedding_tables[0].embedding.shape[1]
        assert all([e.embedding.shape[1] == self.embedding_dim for e in embedding_tables])
        self.embedding_tables = embedding_tables
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def __call__(self, tokens: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        assert tokens.ndim == 2
        output = jnp.zeros((*tokens.shape, self.embedding_dim))
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            embedded = emb(tokens[:, s])
            output = output.at[:, s].set(embedded)
        return output


class GeneralEmbedder(eqx.Module):
    """General embedder with explicit embedding dimension."""
    embedding_dim: int = eqx.field(static=True)
    embedding_tables: List[eqx.Module]
    slicers: List[Slicer]

    def __init__(self, max_blocks: int, block_masks: List[jnp.ndarray], embedding_tables: List[eqx.Module], embedding_dim: int):
        assert len(block_masks) == len(embedding_tables)
        self.embedding_dim = embedding_dim
        self.embedding_tables = list(embedding_tables)
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def __call__(self, tokens: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        B, T = tokens.shape[:2]
        output = jnp.zeros((B, T, self.embedding_dim))
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            embedded = emb(tokens[:, s])
            output = output.at[:, s].set(embedded)
        return output


class MergeEmbedder(eqx.Module):
    """Merged embedding from multiple slicers."""
    embedding_dim: int = eqx.field(static=True)
    merged_tokens: int = eqx.field(static=True)
    slicers_num: int = eqx.field(static=True)
    embedding_tables: List[eqx.nn.Embedding]
    slicers: List[Slicer]

    def __init__(self, max_blocks: int, block_masks: List[jnp.ndarray], embedding_tables: List[eqx.nn.Embedding]):
        assert len(block_masks) == len(embedding_tables)
        assert jnp.all(sum(block_masks) == 1)
        self.embedding_dim = embedding_tables[0].embedding.shape[1]
        assert all([e.embedding.shape[1] == self.embedding_dim for e in embedding_tables])
        self.merged_tokens = int(block_masks[0].sum())
        assert all([int(m.sum()) == self.merged_tokens for m in block_masks])
        self.slicers_num = len(block_masks)
        self.embedding_tables = embedding_tables
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def __call__(self, tokens: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        assert tokens.ndim == 2
        outputs = []
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            outputs.append(emb(tokens[:, s]))
        return jnp.concatenate(outputs, axis=-1)


def create_mlp_head(in_dim: int, hidden_dim: int, out_dim: int, key: jax.random.PRNGKey) -> eqx.nn.Sequential:
    """Create a simple MLP head."""
    keys = jax.random.split(key, 3)
    return eqx.nn.Sequential([
        eqx.nn.Linear(in_dim, hidden_dim, key=keys[0]),
        eqx.nn.Lambda(jax.nn.relu),
        eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
        eqx.nn.Lambda(jax.nn.relu),
        eqx.nn.Linear(hidden_dim, out_dim, key=keys[2]),
    ])
