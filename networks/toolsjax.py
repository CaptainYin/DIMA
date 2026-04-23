"""
JAX/Equinox implementation of loss functions and tools.
Converted from networks/tools.py (PyTorch version).
"""

from typing import Dict, Type, Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric logarithm transformation."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


class HLGaussLoss(eqx.Module):
    """
    HL-Gauss loss for regression as classification.
    Returns (..., num_bins) probability labels.
    """
    min_value: float = eqx.field(static=True)
    max_value: float = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    support: Float[Array, "num_bins+1"]
    output_dim: int = eqx.field(static=True)

    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = jnp.linspace(min_value, max_value, num_bins + 1)
        self.output_dim = num_bins

    def __repr__(self):
        return f"HLGaussLoss(min={self.min_value}, max={self.max_value}, bins={self.num_bins}, sigma={self.sigma})"

    def __call__(self, logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Compute cross-entropy loss with soft labels."""
        probs = self.transform_to_probs(target)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(probs * log_probs, axis=-1).mean()

    def transform_to_probs(self, target: jnp.ndarray) -> jnp.ndarray:
        """Transform continuous targets to probability distributions."""
        cdf_evals = jax.scipy.special.erf(
            (self.support - jnp.expand_dims(target, -1))
            / (jnp.sqrt(2.0) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / jnp.expand_dims(z, -1)

    def transform_from_probs(self, probs: jnp.ndarray) -> jnp.ndarray:
        """Transform probability distribution back to continuous value."""
        centers = (self.support[:-1] + self.support[1:]) / 2
        return jnp.sum(probs * centers, axis=-1)


class TwoHotCategoricalLoss(eqx.Module):
    """
    Two-hot categorical loss for regression as classification.
    Returns (..., num_bins + 1) probability labels.
    """
    min_value: float = eqx.field(static=True)
    max_value: float = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    support: Float[Array, "num_bins+1"]
    output_dim: int = eqx.field(static=True)

    def __init__(self, min_value: float, max_value: float, num_bins: int, **kwargs):
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.support = jnp.linspace(min_value, max_value, num_bins + 1)
        self.output_dim = num_bins + 1

    def __repr__(self):
        return f"TwoHotCategoricalLoss(min={self.min_value}, max={self.max_value}, bins={self.num_bins})"

    def __call__(self, logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Compute cross-entropy loss with two-hot labels."""
        probs = self.transform_to_probs(target)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(probs * log_probs, axis=-1).mean()

    def transform_to_probs(self, target: jnp.ndarray) -> jnp.ndarray:
        """Transform continuous targets to two-hot probability distributions."""
        shape = target.shape
        target_flat = target.reshape(-1)
        batch_size = target_flat.shape[0]
        indices = jnp.sum(self.support < jnp.expand_dims(target_flat, -1), axis=-1)
        indices = jnp.clip(indices, 1, self.num_bins)
        probs = jnp.zeros((batch_size, self.num_bins + 1))
        lower_idx = indices - 1
        upper_idx = indices
        lower_support = self.support[lower_idx]
        upper_support = self.support[upper_idx]
        bin_width = upper_support - lower_support
        bin_width = jnp.where(bin_width == 0, 1.0, bin_width)
        upper_weight = (target_flat - lower_support) / bin_width
        lower_weight = 1.0 - upper_weight
        probs = probs.at[jnp.arange(batch_size), lower_idx].set(lower_weight)
        probs = probs.at[jnp.arange(batch_size), upper_idx].add(upper_weight)
        return probs.reshape(*shape, self.num_bins + 1)

    def transform_from_probs(self, probs: jnp.ndarray) -> jnp.ndarray:
        """Transform probability distribution back to continuous value."""
        return jnp.sum(probs * self.support, axis=-1)


LOSSES_DICT: Dict[str, Type] = {
    'hlgauss': HLGaussLoss,
    'twohot': TwoHotCategoricalLoss,
}


def create_loss(loss_type: str, min_value: float, max_value: float, num_bins: int, sigma: float = None):
    """Factory function to create loss instances."""
    if loss_type == 'hlgauss':
        if sigma is None:
            bin_width = (max_value - min_value) / num_bins
            sigma = bin_width * 0.75
        return HLGaussLoss(min_value, max_value, num_bins, sigma)
    elif loss_type == 'twohot':
        return TwoHotCategoricalLoss(min_value, max_value, num_bins)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def smooth_l1_loss(pred: jnp.ndarray, target: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """Smooth L1 loss (Huber loss)."""
    diff = jnp.abs(pred - target)
    return jnp.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta).mean()


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss for classification."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if labels.ndim == logits.ndim:
        return -jnp.sum(labels * log_probs, axis=-1).mean()
    else:
        num_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(labels, num_classes)
        return -jnp.sum(one_hot * log_probs, axis=-1).mean()


def binary_cross_entropy_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss with logits."""
    max_val = jnp.maximum(-logits, 0)
    loss = logits - logits * labels + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))
    return loss.mean()


def focal_loss(logits: jnp.ndarray, labels: jnp.ndarray, alpha: float = 0.25, gamma: float = 2.0) -> jnp.ndarray:
    """Focal loss for handling class imbalance."""
    probs = jax.nn.sigmoid(logits)
    p_t = jnp.where(labels == 1, probs, 1 - probs)
    alpha_t = jnp.where(labels == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    bce = binary_cross_entropy_with_logits(logits, labels)
    return (focal_weight * bce).mean()
