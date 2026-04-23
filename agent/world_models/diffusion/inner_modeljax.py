from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange

# Suppress the warning about JAX arrays being set as static (intentional for FourierFeatures buffer)
warnings.filterwarnings('ignore', message='A JAX array is being set as static')

from ..blocksjax import  FourierFeatures, GroupNorm
from ..temporalunetjax import UNet1D
from ..perceiverjax import SequentialActionEmb
@dataclass
class StateInnerModelConfig:
    state_dim: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    action_dim: Optional[int] = None
    dim: int = 128
    dim_mults: Tuple[int, ...] = (1, 4, 8)
class InnerModel(eqx.Module):
    # --- submodules ---
    noise_emb: eqx.Module
    act_emb: eqx.Module
    cond_proj1: eqx.nn.Linear
    cond_proj2: eqx.nn.Linear

    conv_in: eqx.Module
    unet: eqx.Module
    norm_out: eqx.Module
    conv_out: eqx.Module

    is_continuous_act: bool = eqx.field(static=True)

    def __init__( self, cfg: StateInnerModelConfig, perceiver_cfg, num_agents: int, is_continuous_act: bool = False, *, key: jax.random.PRNGKey, ):
        k = jax.random.split(key, 10)
        # jax.debug.print("InnerModel: cond_channels={}", cfg.cond_channels)
        self.is_continuous_act = is_continuous_act

        # noise embedding
        self.noise_emb = FourierFeatures(cfg.cond_channels, key=k[0])
        
        # action embedding (Perceiver)
        self.act_emb = SequentialActionEmb(
            num_agents=num_agents, num_steps_conditioning=cfg.num_steps_conditioning, action_dim=cfg.action_dim,
            is_continuous_act=is_continuous_act, perceiver_cfg=perceiver_cfg, key=k[1], )

        # cond projection
        self.cond_proj1 = eqx.nn.Linear(cfg.cond_channels, cfg.cond_channels, key=k[2])
        self.cond_proj2 = eqx.nn.Linear(cfg.cond_channels, cfg.cond_channels, key=k[3])

        # input conv: (B, T, C) -> (B, T, hidden)
        self.conv_in = eqx.nn.Conv1d(cfg.state_dim, cfg.channels[0], 3, stride=1, padding=1, key=k[4])

        # UNet1D
        self.unet = UNet1D( cond_channels=cfg.cond_channels, depths=cfg.depths, channels=cfg.channels, attn_depths=cfg.attn_depths, key=k[5], )

        # output
        self.norm_out = GroupNorm(cfg.channels[0], key=k[6])
        self.conv_out = eqx.nn.Conv1d(cfg.channels[0], cfg.state_dim, 3, stride=1, padding=1, key=k[7])
        

        # zero-init output conv (重要)
        self.conv_out = eqx.tree_at(
            lambda m: m.weight,
            self.conv_out,
            jnp.zeros_like(self.conv_out.weight),
        )
    def compute_action_cond( self, act: jnp.ndarray, mask: jnp.ndarray, return_cross_attn: bool = False, key: Optional[jax.random.PRNGKey] = None, ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        return self.act_emb(act, mask, return_cross_attn,key)
    def __call__(
        self,
        noisy_next_obs: jnp.ndarray,   # (B, H, T)
        c_noise: jnp.ndarray,          # (B,)
        obs: jnp.ndarray,              # (B, H, T)
        act: jnp.ndarray,              # (B, T, N, act_dim)
        act_mask: jnp.ndarray,         # (B, T, N)
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:

        # --- action condition ---
        act_cond, _ = self.compute_action_cond(act, act_mask, False,key)

        # --- cond embedding ---
        cond = self.noise_emb(c_noise) + act_cond
        cond = jax.vmap(self.cond_proj2)(jax.nn.silu(jax.vmap(self.cond_proj1)(cond)))

        # --- input concat ---
        x = jnp.concatenate([obs, noisy_next_obs], axis=1)   # (B, 2H, T)
        x = rearrange(x, "b h t -> b t h")

        # --- UNet ---
        x = jax.vmap(self.conv_in)(x)
        # vmap unet over batch dimension (x: [B, T, C], cond: [B, cond_channels])
        def unet_call(x, cond):
            return self.unet(x, cond)
        x, _, _ = jax.vmap(unet_call)(x, cond)

        # --- output ---
        x = jax.vmap(self.conv_out)(jax.nn.silu(jax.vmap(self.norm_out)(x)))
        x = rearrange(x, "b t h -> b h t")

        # only predict last horizon step
        return x[:, -1:, :]
