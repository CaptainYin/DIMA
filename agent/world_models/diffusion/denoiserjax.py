from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .inner_model import StateInnerModelConfig
from .inner_modeljax import InnerModel as InnerModelJax
from ..perceiverjax import PerceiverConfig


def add_dims(input: jnp.ndarray, n: int) -> jnp.ndarray:
    """Expand dimensions of input array to have n dimensions."""
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: jnp.ndarray
    c_out: jnp.ndarray
    c_skip: jnp.ndarray
    c_noise: jnp.ndarray


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: StateInnerModelConfig
    perceiver: PerceiverConfig
    sigma_data: float
    sigma_offset_noise: float


class Denoiser(eqx.Module):
    """Denoiser module for diffusion models, implemented in JAX/Equinox."""
    cfg: DenoiserConfig = eqx.static_field()
    device: str = eqx.static_field()
    inner_model: InnerModelJax
    sigma_data: float = eqx.static_field()
    sigma_offset_noise: float = eqx.static_field()
    num_agents: int = eqx.static_field()
    is_continuous_act: bool = eqx.static_field()
    clip_denoised: bool = eqx.static_field()
    steps_condition: int = eqx.static_field()
    
    # Sigma distribution parameters (set via setup_training)
    sigma_loc: Optional[float] = eqx.static_field()
    sigma_scale: Optional[float] = eqx.static_field()
    sigma_min: Optional[float] = eqx.static_field()
    sigma_max: Optional[float] = eqx.static_field()

    def __init__(
        self,
        cfg: DenoiserConfig,
        num_agents: int,
        key: jax.random.PRNGKey,
        clip_denoised: bool = False,
        is_continuous_act: bool = False,
        device: str ='cpu'
    ) -> None:
        self.cfg = cfg
        self.device=device
        self.inner_model = InnerModelJax(
            cfg.inner_model,
            cfg.perceiver,
            num_agents=num_agents,
            is_continuous_act=is_continuous_act,
            key=key,
        )
        self.sigma_data = cfg.sigma_data
        self.sigma_offset_noise = cfg.sigma_offset_noise
        self.num_agents = num_agents
        self.is_continuous_act = is_continuous_act
        self.clip_denoised = clip_denoised
        self.steps_condition = cfg.inner_model.num_steps_conditioning
        
        # Initialize sigma distribution params as None (set via setup_training)
        self.sigma_loc = None
        self.sigma_scale = None
        self.sigma_min = None
        self.sigma_max = None

    def setup_training(self, cfg: SigmaDistributionConfig) -> "Denoiser":
        """Set up training sigma distribution. Returns a new Denoiser with updated config.
        
        Since sigma parameters are static fields, we use object.__setattr__ to update them.
        """
        # For static fields, we need to create a modified copy manually
        new_denoiser = object.__new__(Denoiser)
        # Copy all fields
        object.__setattr__(new_denoiser, 'inner_model', self.inner_model)
        object.__setattr__(new_denoiser, 'sigma_data', self.sigma_data)
        object.__setattr__(new_denoiser, 'sigma_offset_noise', self.sigma_offset_noise)
        object.__setattr__(new_denoiser, 'num_agents', self.num_agents)
        object.__setattr__(new_denoiser, 'is_continuous_act', self.is_continuous_act)
        object.__setattr__(new_denoiser, 'clip_denoised', self.clip_denoised)
        object.__setattr__(new_denoiser, 'steps_condition', self.steps_condition)
        # Set new sigma params
        object.__setattr__(new_denoiser, 'sigma_loc', cfg.loc)
        object.__setattr__(new_denoiser, 'sigma_scale', cfg.scale)
        object.__setattr__(new_denoiser, 'sigma_min', cfg.sigma_min)
        object.__setattr__(new_denoiser, 'sigma_max', cfg.sigma_max)
        return new_denoiser

    def sample_sigma(self, n: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample sigma values for training."""
        assert self.sigma_loc is not None, "Call setup_training first"
        s = jax.random.normal(key, (n,)) * self.sigma_scale + self.sigma_loc
        return jnp.clip(jnp.exp(s), self.sigma_min, self.sigma_max)

    def encode(self, latent: jnp.ndarray) -> jnp.ndarray:
        """Encode latent (scale by sigma_data)."""
        return latent * self.sigma_data

    def decode(self, latent: jnp.ndarray) -> jnp.ndarray:
        """Decode latent (scale by 1/sigma_data)."""
        return latent / self.sigma_data

    def apply_noise(
        self,
        x: jnp.ndarray,
        sigma: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Apply noise to input x with given sigma."""
        key1, key2 = jax.random.split(key)
        offset_noise = self.sigma_offset_noise * jax.random.normal(key1, x.shape)
        noise = jax.random.normal(key2, x.shape) * add_dims(sigma, x.ndim)
        return x + offset_noise + noise

    def compute_conditioners(self, sigma: jnp.ndarray) -> Conditioners:
        """Compute conditioning scalars from sigma."""
        sigma = jnp.sqrt(sigma**2 + self.sigma_offset_noise**2)
        c_in = 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * jnp.sqrt(c_skip)
        c_noise = jnp.log(sigma) / 4
        return Conditioners(
            c_in=add_dims(c_in, 3),
            c_out=add_dims(c_out, 3),
            c_skip=add_dims(c_skip, 3),
            c_noise=add_dims(c_noise, 1),
        )

    def compute_model_output(
        self,
        noisy_next_obs: jnp.ndarray,
        obs: jnp.ndarray,
        act: jnp.ndarray,
        cs: Conditioners,
        act_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Compute denoising model output."""
        # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
        innermodel_input = noisy_next_obs * cs.c_in
        return self.inner_model( innermodel_input, cs.c_noise, obs, act, act_mask, key )

    def wrap_model_output(
        self,
        noisy_next_obs: jnp.ndarray,
        model_output: jnp.ndarray,
        cs: Conditioners,
    ) -> jnp.ndarray:
        """Wrap model output with skip connection and scaling."""
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        if self.clip_denoised:
            d = jnp.clip(d, -1.0, 1.0)
        return d

    def denoise( self, noisy_next_obs: jnp.ndarray, sigma: jnp.ndarray, obs: jnp.ndarray, act: jnp.ndarray, 
                act_mask: jnp.ndarray, key: jax.random.PRNGKey, ) -> jnp.ndarray:
        """Denoise noisy_next_obs given conditioning."""
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output( noisy_next_obs, obs, act, cs, act_mask, key )
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def __call__( self, shared_obs: jnp.ndarray, act: jnp.ndarray, mask_padding: jnp.ndarray, 
                 key: jax.random.PRNGKey, ) -> Tuple[jnp.ndarray, dict]:
        """
        Forward pass for training.
        
        Args:
            shared_obs: (batch, seq_length + steps_condition, state_dim)
            act: (batch, seq_length + steps_condition, n_agents, act_dim)
            mask_padding: (batch, seq_length + steps_condition)
            key: JAX random key
            
        Returns:
            loss: scalar loss
            logs: dict with loss info
        """
        batch_size = shared_obs.shape[0]
        seq_length = shared_obs.shape[1] - self.steps_condition
        all_obs = shared_obs  # (b, T, state_dim)

        def step_fn(carry, i):
            """Single training step for one timestep."""
            all_obs_carry, key_carry = carry
            key_carry, key_sigma, key_noise, key_mask, key_model = jax.random.split(key_carry, 5)

            # Extract obs, next_obs, act for this timestep
            obs = jax.lax.dynamic_slice(
                all_obs_carry,
                (0, i, 0),
                (batch_size, self.steps_condition, all_obs_carry.shape[2]),
            )  # (b, steps_condition, state_dim)
            
            next_obs = all_obs_carry[:, self.steps_condition + i, :]  # (b, state_dim)
            next_obs = next_obs[:, None, :]  # (b, 1, state_dim)
            
            act_slice = jax.lax.dynamic_slice(
                act,
                (0, i, 0, 0),
                (batch_size, self.steps_condition, self.num_agents, act.shape[3]),
            )  # (b, steps_condition, n_agents, act_dim)
            
            mask = mask_padding[:, self.steps_condition + i]  # (b,)

            # Handle discrete actions
            act_input = act_slice if self.is_continuous_act else jnp.argmax(act_slice, axis=-1)

            # Sample sigma and apply noise
            sigma = self.sample_sigma(batch_size, key_sigma)
            noisy_next_obs = self.apply_noise(next_obs, sigma, key_noise)

            # Compute conditioners
            cs = self.compute_conditioners(sigma)

            # Create action mask: all ones except last timestep has one-hot for random agent
            activate_indices = jax.random.randint(
                key_mask, (batch_size,), 0, self.num_agents
            )
            act_mask_prefix = jnp.ones(
                (batch_size, self.steps_condition - 1, self.num_agents), dtype=jnp.int32
            )
            act_mask_last = jax.nn.one_hot(
                activate_indices, self.num_agents, dtype=jnp.int32
            )[:, None, :]
            act_mask = jnp.concatenate([act_mask_prefix, act_mask_last], axis=1)

            # Compute model output
            model_output = self.compute_model_output(
                noisy_next_obs, obs, act_input, cs, act_mask, key_model
            )

            # Compute loss
            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
            
            # MSE loss with mask
            sq_diff = (model_output - target) ** 2
            # Reduce over non-batch dims, then apply mask
            sq_diff_reduced = jnp.mean(sq_diff, axis=(1, 2))  # (b,)
            loss = jnp.sum(sq_diff_reduced * mask) / jnp.maximum(jnp.sum(mask), 1.0)

            # Update all_obs with denoised prediction
            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            all_obs_updated = all_obs_carry.at[:, self.steps_condition + i, :].set(
                denoised[:, 0, :]
            )

            return (all_obs_updated, key_carry), loss

        # Run over sequence
        (final_obs, _), losses = jax.lax.scan( step_fn, (all_obs, key), jnp.arange(seq_length), )

        total_loss = jnp.mean(losses)
        return total_loss, {"loss_denoising": total_loss}


# JIT-compiled version for inference
@eqx.filter_jit
def denoise_jit( denoiser: Denoiser, noisy_next_obs: jnp.ndarray, sigma: jnp.ndarray, obs: jnp.ndarray, 
                act: jnp.ndarray, act_mask: jnp.ndarray, key: jax.random.PRNGKey, ) -> jnp.ndarray:
    """JIT-compiled denoise function."""
    return denoiser.denoise(noisy_next_obs, sigma, obs, act, act_mask, key)
