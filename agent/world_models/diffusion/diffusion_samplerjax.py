from dataclasses import dataclass
from typing import List, Tuple, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import repeat

from .denoiserjax import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    agent_order: str = ""

    loc: float = -0.4
    scale: float = 1.2


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int) -> jnp.ndarray:
    """Build sigma schedule for denoising."""
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = jnp.linspace(0, 1, num_steps)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return jnp.concatenate([sigmas, jnp.zeros(1)])


class DiffusionSampler(eqx.Module):
    """Diffusion sampler for inference, implemented in JAX/Equinox."""
    
    denoiser: Denoiser
    sigmas: jnp.ndarray  # Not static - JAX arrays cannot be compared with __eq__
    num_steps_denoising: int = eqx.static_field()
    sigma_min: float = eqx.static_field()
    sigma_max: float = eqx.static_field()
    rho: int = eqx.static_field()
    order: int = eqx.static_field()
    s_churn: float = eqx.static_field()
    s_tmin: float = eqx.static_field()
    s_tmax: float = eqx.static_field()
    s_noise: float = eqx.static_field()
    agent_order: str = eqx.static_field()

    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.num_steps_denoising = cfg.num_steps_denoising
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.order = cfg.order
        self.s_churn = cfg.s_churn
        self.s_tmin = cfg.s_tmin
        self.s_tmax = cfg.s_tmax
        self.s_noise = cfg.s_noise
        self.agent_order = cfg.agent_order
        
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho)

    def encode(self, state: jnp.ndarray) -> jnp.ndarray:
        """Encode state."""
        return self.denoiser.encode(state)

    def decode(self, state: jnp.ndarray) -> jnp.ndarray:
        """Decode state."""
        return self.denoiser.decode(state)

    def sample_agent_order(
        self, 
        num_agents: int, 
        order: str = "default",
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Sample agent order for sequential denoising."""
        if order == 'default':
            agent_order = jnp.flip(jnp.arange(num_agents))
        elif order == 'reverse':
            agent_order = jnp.arange(num_agents)
        elif order == 'random':
            assert key is not None, "Key required for random order"
            agent_order = jax.random.permutation(key, num_agents)
        else:
            raise NotImplementedError('Please specify the agent order for denoising.')
        
        return agent_order

    def sample(
        self,
        prev_state: jnp.ndarray,
        prev_act: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Sample next state given previous states and actions.
        
        Args:
            prev_state: (batch, seq_length, state_dim) or (batch, seq_length, num_agents, state_dim)
            prev_act: (batch, seq_length, num_agents, act_dim)
            key: JAX random key
            
        Returns:
            x: Sampled next state (batch, 1, state_dim)
            trajectory: List of intermediate states
        """
        # Handle 4D state input (mean over agents)
        if prev_state.ndim == 4:
            prev_state = jnp.mean(prev_state, axis=2)

        # Handle discrete actions
        if not self.denoiser.is_continuous_act and prev_act.ndim == 4:
            prev_act = jnp.argmax(prev_act, axis=-1)

        b, t, d = prev_state.shape
        
        gamma_ = jnp.minimum(self.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        
        # Initialize with noise
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (b, 1, d)) * self.sigmas[0]
        
        # Get agent order
        num_agents = self.denoiser.num_agents
        key, subkey = jax.random.split(key)
        agent_order = self.sample_agent_order(num_agents, self.agent_order, subkey)
        
        # Repeat agent order if needed
        if self.num_steps_denoising != len(agent_order):
            agent_order = jnp.repeat(agent_order, 2)

        trajectory = [x]

        # Denoising loop
        for idx in range(len(self.sigmas) - 1):
            sigma = self.sigmas[idx]
            next_sigma = self.sigmas[idx + 1]
            
            # Compute gamma
            in_range = (self.s_tmin <= sigma) & (sigma <= self.s_tmax)
            gamma = jnp.where(in_range, gamma_, 0.0)
            sigma_hat = sigma * (gamma + 1)
            
            # Add noise if gamma > 0
            key, subkey = jax.random.split(key)
            eps = jax.random.normal(subkey, x.shape) * self.s_noise
            x = jnp.where(gamma > 0, x + eps * jnp.sqrt(sigma_hat**2 - sigma**2), x)

            # Create action mask
            act_mask = jnp.ones((*prev_act.shape[:3],), dtype=jnp.int32)
            one_hot_agent = jax.nn.one_hot(agent_order[idx], num_agents, dtype=jnp.int32)
            act_mask = act_mask.at[:, -1].set(jnp.broadcast_to(one_hot_agent, (b, num_agents)))

            # Denoise
            key, subkey = jax.random.split(key)
            denoised = self.denoiser.denoise(x, sigma_hat, prev_state, prev_act, act_mask, subkey)
            
            # Compute derivative
            d_cur = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            
            if self.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d_cur * dt
            else:
                # Heun's method
                x_2 = x + d_cur * dt
                key, subkey = jax.random.split(key)
                denoised_2 = self.denoiser.denoise(x_2, next_sigma, prev_state, prev_act, act_mask, subkey)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d_cur + d_2) / 2
                x = x + d_prime * dt
            
            trajectory.append(x)

        return x, trajectory

    def ensemble_sample(
        self,
        prev_obs: jnp.ndarray,
        prev_act: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[List[jnp.ndarray], List[List[jnp.ndarray]]]:
        """
        Ensemble sampling with different agent orders.
        
        Returns:
            xs: List of sampled states
            trajs: List of trajectories
        """
        xs = []
        trajs = []
        
        # Default order
        key, subkey = jax.random.split(key)
        x, trajectory = self.sample(prev_obs, prev_act, subkey)
        xs.append(x)
        trajs.append(trajectory)
        
        # Reverse order - create new sampler with reverse order
        reverse_sampler = self._with_agent_order("reverse")
        key, subkey = jax.random.split(key)
        x, trajectory = reverse_sampler.sample(prev_obs, prev_act, subkey)
        xs.append(x)
        trajs.append(trajectory)
        
        return xs, trajs

    def _with_agent_order(self, order: str) -> "DiffusionSampler":
        """Create a new sampler with different agent order."""
        new_sampler = object.__new__(DiffusionSampler)
        object.__setattr__(new_sampler, 'denoiser', self.denoiser)
        object.__setattr__(new_sampler, 'sigmas', self.sigmas)
        object.__setattr__(new_sampler, 'num_steps_denoising', self.num_steps_denoising)
        object.__setattr__(new_sampler, 'sigma_min', self.sigma_min)
        object.__setattr__(new_sampler, 'sigma_max', self.sigma_max)
        object.__setattr__(new_sampler, 'rho', self.rho)
        object.__setattr__(new_sampler, 'order', self.order)
        object.__setattr__(new_sampler, 's_churn', self.s_churn)
        object.__setattr__(new_sampler, 's_tmin', self.s_tmin)
        object.__setattr__(new_sampler, 's_tmax', self.s_tmax)
        object.__setattr__(new_sampler, 's_noise', self.s_noise)
        object.__setattr__(new_sampler, 'agent_order', order)
        return new_sampler


# JIT-compiled sample function
@eqx.filter_jit
def sample_jit( sampler: DiffusionSampler, prev_state: jnp.ndarray, prev_act: jnp.ndarray, 
               key: jax.random.PRNGKey, ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled sample function. Returns final state only (no trajectory for JIT)."""
    # Simplified version for JIT - returns only final state
    if prev_state.ndim == 4:
        prev_state = jnp.mean(prev_state, axis=2)

    if not sampler.denoiser.is_continuous_act and prev_act.ndim == 4:
        prev_act = jnp.argmax(prev_act, axis=-1)

    b, t, d = prev_state.shape
    
    gamma_ = jnp.minimum(sampler.s_churn / (len(sampler.sigmas) - 1), 2**0.5 - 1)
    
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (b, 1, d)) * sampler.sigmas[0]
    
    num_agents = sampler.denoiser.num_agents
    key, subkey = jax.random.split(key)
    agent_order = sampler.sample_agent_order(num_agents, sampler.agent_order, subkey)
    
    if sampler.num_steps_denoising != len(agent_order):
        agent_order = jnp.repeat(agent_order, 2)

    def step_fn(carry, idx):
        x, key = carry
        sigma = sampler.sigmas[idx]
        next_sigma = sampler.sigmas[idx + 1]
        
        in_range = (sampler.s_tmin <= sigma) & (sigma <= sampler.s_tmax)
        gamma = jnp.where(in_range, gamma_, 0.0)
        sigma_hat = sigma * (gamma + 1)
        
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, x.shape) * sampler.s_noise
        x = jnp.where(gamma > 0, x + eps * jnp.sqrt(sigma_hat**2 - sigma**2), x)

        act_mask = jnp.ones((*prev_act.shape[:3],), dtype=jnp.int32)
        one_hot_agent = jax.nn.one_hot(agent_order[idx], num_agents, dtype=jnp.int32)
        act_mask = act_mask.at[:, -1].set(jnp.broadcast_to(one_hot_agent, (b, num_agents)))

        key, subkey = jax.random.split(key)
        denoised = sampler.denoiser.denoise(x, sigma_hat, prev_state, prev_act, act_mask, subkey)
        
        d_cur = (x - denoised) / sigma_hat
        dt = next_sigma - sigma_hat
        x = x + d_cur * dt
        
        return (x, key), x

    (x_final, _), _ = jax.lax.scan( step_fn, (x, key), jnp.arange(len(sampler.sigmas) - 1), )

    return x_final
