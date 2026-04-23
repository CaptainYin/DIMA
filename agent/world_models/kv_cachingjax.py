"""
JAX/Equinox implementation of KV Caching for Transformers.
Converted from kv_caching.py (PyTorch version).
"""

from typing import Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class CacheState(NamedTuple):
    """Immutable state for a single cache (keys or values)."""
    cache: Float[Array, "batch num_heads max_tokens head_dim"]
    size: int


class KVCacheState(NamedTuple):
    """Immutable state for key-value cache pair."""
    k_cache: CacheState
    v_cache: CacheState


class KeysValuesState(NamedTuple):
    """Immutable state for all transformer layers' KV caches."""
    kv_caches: Tuple[KVCacheState, ...]


def create_cache(
    num_samples: int,
    num_heads: int,
    max_tokens: int,
    embed_dim: int,
) -> CacheState:
    """Create an empty cache state."""
    assert embed_dim % num_heads == 0
    head_dim = embed_dim // num_heads
    cache = jnp.zeros((num_samples, num_heads, max_tokens, head_dim))
    return CacheState(cache=cache, size=0)


def reset_cache(cache_state: CacheState) -> CacheState:
    """Reset cache to empty state."""
    return CacheState(
        cache=jnp.zeros_like(cache_state.cache),
        size=0
    )


def prune_cache(cache_state: CacheState, mask: jnp.ndarray) -> CacheState:
    """Prune cache based on boolean mask."""
    return CacheState(
        cache=cache_state.cache[mask],
        size=cache_state.size
    )


def get_cache(cache_state: CacheState) -> jnp.ndarray:
    """Get the valid portion of the cache."""
    return jax.lax.dynamic_slice(
        cache_state.cache,
        (0, 0, 0, 0),
        (cache_state.cache.shape[0], cache_state.cache.shape[1], cache_state.size, cache_state.cache.shape[3])
    )


def update_cache(cache_state: CacheState, x: jnp.ndarray) -> CacheState:
    """Update cache with new values and return new state."""
    new_size = cache_state.size + x.shape[2]
    new_cache = jax.lax.dynamic_update_slice(
        cache_state.cache,
        x,
        (0, 0, cache_state.size, 0)
    )
    return CacheState(cache=new_cache, size=new_size)


# KVCache functions
def create_kv_cache(
    num_samples: int,
    num_heads: int,
    max_tokens: int,
    embed_dim: int,
) -> KVCacheState:
    """Create an empty KV cache state."""
    k_cache = create_cache(num_samples, num_heads, max_tokens, embed_dim)
    v_cache = create_cache(num_samples, num_heads, max_tokens, embed_dim)
    return KVCacheState(k_cache=k_cache, v_cache=v_cache)


def reset_kv_cache(kv_state: KVCacheState) -> KVCacheState:
    """Reset KV cache to empty state."""
    return KVCacheState(
        k_cache=reset_cache(kv_state.k_cache),
        v_cache=reset_cache(kv_state.v_cache)
    )


def prune_kv_cache(kv_state: KVCacheState, mask: jnp.ndarray) -> KVCacheState:
    """Prune KV cache based on boolean mask."""
    return KVCacheState(
        k_cache=prune_cache(kv_state.k_cache, mask),
        v_cache=prune_cache(kv_state.v_cache, mask)
    )


def get_kv_cache(kv_state: KVCacheState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get the valid portion of both K and V caches."""
    return get_cache(kv_state.k_cache), get_cache(kv_state.v_cache)


def update_kv_cache(
    kv_state: KVCacheState,
    k: jnp.ndarray,
    v: jnp.ndarray
) -> KVCacheState:
    """Update both K and V caches with new values."""
    return KVCacheState(
        k_cache=update_cache(kv_state.k_cache, k),
        v_cache=update_cache(kv_state.v_cache, v)
    )


def kv_cache_shape(kv_state: KVCacheState) -> Tuple[int, int, int, int]:
    """Get shape of KV cache (n, num_heads, size, head_dim)."""
    cache = kv_state.k_cache
    return (
        cache.cache.shape[0],
        cache.cache.shape[1],
        cache.size,
        cache.cache.shape[3]
    )


# KeysValues functions (multi-layer)
def create_keys_values(
    num_samples: int,
    num_heads: int,
    max_tokens: int,
    embed_dim: int,
    num_layers: int,
) -> KeysValuesState:
    """Create empty KV caches for all layers."""
    kv_caches = tuple(
        create_kv_cache(num_samples, num_heads, max_tokens, embed_dim)
        for _ in range(num_layers)
    )
    return KeysValuesState(kv_caches=kv_caches)


def reset_keys_values(kv_state: KeysValuesState) -> KeysValuesState:
    """Reset all KV caches."""
    return KeysValuesState(
        kv_caches=tuple(reset_kv_cache(kv) for kv in kv_state.kv_caches)
    )


def prune_keys_values(kv_state: KeysValuesState, mask: jnp.ndarray) -> KeysValuesState:
    """Prune all KV caches based on boolean mask."""
    return KeysValuesState(
        kv_caches=tuple(prune_kv_cache(kv, mask) for kv in kv_state.kv_caches)
    )


def get_keys_values_size(kv_state: KeysValuesState) -> int:
    """Get the current size of the cache."""
    return kv_state.kv_caches[0].k_cache.size


def update_layer_kv_cache(
    kv_state: KeysValuesState,
    layer_idx: int,
    k: jnp.ndarray,
    v: jnp.ndarray
) -> KeysValuesState:
    """Update KV cache for a specific layer."""
    new_kv_caches = list(kv_state.kv_caches)
    new_kv_caches[layer_idx] = update_kv_cache(kv_state.kv_caches[layer_idx], k, v)
    return KeysValuesState(kv_caches=tuple(new_kv_caches))
