"""
JAX/Equinox utilities.
Converted from utils.py (PyTorch version).
"""

from typing import Tuple, Dict, Any, Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PyTree


LossAndLogs = Tuple[Float[Array, ""], Dict[str, Any]]


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric logarithm transformation."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def init_lstm_weights(key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Initialize weights using Xavier uniform."""
    fan_in = shape[0] if len(shape) > 1 else shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_lstm_hidden_weights(key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Initialize hidden weights using orthogonal initialization."""
    if len(shape) < 2:
        return jax.random.normal(key, shape) * 0.02
    a = jax.random.normal(key, shape)
    q, r = jnp.linalg.qr(a)
    d = jnp.diag(r)
    ph = jnp.sign(d)
    q = q * ph
    return q


def init_linear_weight(key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Initialize linear layer weights with normal distribution (std=0.02)."""
    return jax.random.normal(key, shape) * 0.02


def init_linear_bias(shape: Tuple[int, ...]) -> jnp.ndarray:
    """Initialize bias to zeros."""
    return jnp.zeros(shape)


def init_layernorm_weight(shape: Tuple[int, ...]) -> jnp.ndarray:
    """Initialize LayerNorm weight to ones."""
    return jnp.ones(shape)


def init_layernorm_bias(shape: Tuple[int, ...]) -> jnp.ndarray:
    """Initialize LayerNorm bias to zeros."""
    return jnp.zeros(shape)


def init_embedding_weight(key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Initialize embedding weights with normal distribution (std=0.02)."""
    return jax.random.normal(key, shape) * 0.02


def init_weights(model: PyTree, key: jax.random.PRNGKey) -> PyTree:
    """Initialize weights of a model."""
    def init_leaf(leaf, subkey):
        if isinstance(leaf, jnp.ndarray):
            if leaf.ndim >= 2:
                return jax.random.normal(subkey, leaf.shape) * 0.02
            else:
                return leaf
        return leaf
    
    leaves, treedef = jax.tree_util.tree_flatten(model)
    keys = jax.random.split(key, len(leaves))
    new_leaves = [init_leaf(leaf, k) for leaf, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


class GRUCell(eqx.Module):
    """GRU Cell implementation."""
    hidden_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    weight_ih: Float[Array, "3*hidden_size input_size"]
    weight_hh: Float[Array, "3*hidden_size hidden_size"]
    bias_ih: Float[Array, "3*hidden_size"]
    bias_hh: Float[Array, "3*hidden_size"]

    def __init__(self, input_size: int, hidden_size: int, *, key: jax.random.PRNGKey):
        self.input_size = input_size
        self.hidden_size = hidden_size
        keys = jax.random.split(key, 4)
        limit = jnp.sqrt(6.0 / (input_size + hidden_size))
        self.weight_ih = jax.random.uniform(keys[0], (3 * hidden_size, input_size), minval=-limit, maxval=limit)
        weight_hh = jax.random.normal(keys[1], (3 * hidden_size, hidden_size))
        chunks = jnp.split(weight_hh, 3, axis=0)
        orth_chunks = []
        for i, chunk in enumerate(chunks):
            q, r = jnp.linalg.qr(chunk.T)
            d = jnp.diag(r)
            ph = jnp.sign(d)
            q = q * ph
            orth_chunks.append(q.T)
        self.weight_hh = jnp.concatenate(orth_chunks, axis=0)
        self.bias_ih = jnp.zeros(3 * hidden_size)
        self.bias_hh = jnp.zeros(3 * hidden_size)

    def __call__(self, x: jnp.ndarray, hx: jnp.ndarray) -> jnp.ndarray:
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = hx @ self.weight_hh.T + self.bias_hh
        i_r, i_z, i_n = jnp.split(gi, 3, axis=-1)
        h_r, h_z, h_n = jnp.split(gh, 3, axis=-1)
        r = jax.nn.sigmoid(i_r + h_r)
        z = jax.nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        hy = (1 - z) * n + z * hx
        return hy


class LSTMCell(eqx.Module):
    """LSTM Cell implementation."""
    hidden_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    weight_ih: Float[Array, "4*hidden_size input_size"]
    weight_hh: Float[Array, "4*hidden_size hidden_size"]
    bias_ih: Float[Array, "4*hidden_size"]
    bias_hh: Float[Array, "4*hidden_size"]

    def __init__(self, input_size: int, hidden_size: int, *, key: jax.random.PRNGKey):
        self.input_size = input_size
        self.hidden_size = hidden_size
        keys = jax.random.split(key, 4)
        limit = jnp.sqrt(6.0 / (input_size + hidden_size))
        self.weight_ih = jax.random.uniform(keys[0], (4 * hidden_size, input_size), minval=-limit, maxval=limit)
        weight_hh = jax.random.normal(keys[1], (4 * hidden_size, hidden_size))
        chunks = jnp.split(weight_hh, 4, axis=0)
        orth_chunks = []
        for chunk in chunks:
            q, r = jnp.linalg.qr(chunk.T)
            d = jnp.diag(r)
            ph = jnp.sign(d)
            q = q * ph
            orth_chunks.append(q.T)
        self.weight_hh = jnp.concatenate(orth_chunks, axis=0)
        bias_ih = jnp.zeros(4 * hidden_size)
        bias_ih = bias_ih.at[hidden_size:2*hidden_size].set(1.0)
        self.bias_ih = bias_ih
        self.bias_hh = jnp.zeros(4 * hidden_size)

    def __call__(self, x: jnp.ndarray, hx: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        h, c = hx
        gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)
        return new_h, (new_h, new_c)


def compute_lambda_return(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray, gamma: float = 0.99, lambda_: float = 0.95) -> jnp.ndarray:
    """Compute lambda returns (GAE-style)."""
    T = rewards.shape[1]
    def scan_fn(carry, t):
        next_value, returns = carry
        reward = rewards[:, T - 1 - t]
        value = values[:, T - 1 - t]
        done = dones[:, T - 1 - t]
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambda_ * (1 - done) * returns
        return (value, gae), gae
    _, returns = jax.lax.scan(scan_fn, (values[:, -1], jnp.zeros_like(rewards[:, 0])), jnp.arange(T))
    returns = jnp.flip(returns, axis=0).T
    return returns + values[:, :-1]
