from dataclasses import dataclass
from typing import Optional, Tuple
import math
import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange, repeat


# =========================
# Config
# =========================

@dataclass
class PerceiverConfig:
    dim: int
    latent_dim: int
    num_latents: int
    depth: int

    cross_heads: int
    cross_dim_head: int
    latent_heads: int
    latent_dim_head: int
    attn_dropout: float
    ff_dropout: float

    output_dim: int
    final_proj_head: bool


# =========================
# Utils
# =========================

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d


# =========================
# PreNorm
# =========================

class PreNorm(eqx.Module):
    fn: eqx.Module
    norm: eqx.nn.LayerNorm
    norm_context: Optional[eqx.nn.LayerNorm]

    def __init__(self, dim, fn, context_dim=None):
        self.fn = fn
        self.norm = eqx.nn.LayerNorm(dim)
        self.norm_context = ( eqx.nn.LayerNorm(context_dim) if exists(context_dim) else None )

    def __call__(self, x, **kwargs):
        # Handle both 2D (num_agents, dim) and 3D (batch, num_agents, dim) inputs
        if x.ndim == 2:
            x = jax.vmap(self.norm)(x)
            if exists(self.norm_context):
                context = jax.vmap(self.norm_context)(kwargs["context"])
                kwargs["context"] = context
        else:
            x = jax.vmap(jax.vmap(self.norm))(x)
            if exists(self.norm_context):
                context = jax.vmap(jax.vmap(self.norm_context))(kwargs["context"])
                kwargs["context"] = context
        
        return self.fn(x, **kwargs)


# =========================
# FeedForward (GEGLU)
# =========================
class GEGLU(eqx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (..., 2 * hidden_dim)
        """
        x, gates = jnp.split(x, 2, axis=-1)
        return x * jax.nn.gelu(gates)
    
class FeedForward(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__( self, dim: int, mult: int = 4, dropout: float = 0.0, key: jax.random.PRNGKey=None ):
        k1, k2 = jax.random.split(key, 2)

        self.fc1 = eqx.nn.Linear( dim, dim * mult * 2, key=k1, )
        self.fc2 = eqx.nn.Linear( dim * mult, dim, key=k2, )
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__( self, x: jnp.ndarray, *, key: Optional[jax.random.PRNGKey] = None, **kwargs) -> jnp.ndarray:
        """
        x: (b, n, dim) or (n, dim)
        """
        # Handle both 2D (n, dim) and 3D (b, n, dim) inputs
        if x.ndim == 2:
            x = jax.vmap(self.fc1)(x)
            x = GEGLU()(x)
            x = jax.vmap(self.fc2)(x)
        else:
            x = jax.vmap(jax.vmap(self.fc1))(x)
            x = GEGLU()(x)
            x = jax.vmap(jax.vmap(self.fc2))(x)
        x = self.dropout(x, key=key)
        return x
    
# class FeedForward(eqx.Module):
#     l1: eqx.nn.Linear
#     l2: eqx.nn.Linear
#     dropout: eqx.nn.Dropout

#     def __init__(self, dim, mult=4, dropout=0.0, *, key):
#         k1, k2 = jax.random.split(key)
#         jax.debug.print("FeedForward dim: {}, mult: {}", dim, mult)#512,4
#         self.l1 = eqx.nn.Linear(dim, dim * mult * 2, key=k1)#512,4096
#         self.l2 = eqx.nn.Linear(dim * mult, dim, key=k2)
#         self.dropout = eqx.nn.Dropout(dropout)

#     def __call__(self, x, *, key=None):
#         jax.debug.print("FeedForward input x shape: {}", x.shape)#64,32,512
        
#         x, gates = jnp.split(jax.vmap(self.l1)(x), 2, axis=-1)
        
#         x = x * jax.nn.gelu(gates)
#         x = self.l2(x)
#         return self.dropout(x, key=key)


# =========================
# Attention
# =========================

class PerAttention(eqx.Module):
    heads: int
    scale: float
    to_q: eqx.nn.Linear
    to_kv: eqx.nn.Linear
    to_out: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__( self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, *, key, ):
        inner = heads * dim_head
        context_dim = default(context_dim, query_dim)
        k1, k2, k3 = jax.random.split(key, 3)

        self.heads = heads
        self.scale = dim_head ** -0.5
        # jax.debug.print("PerAttention query_dim: {}, context_dim: {}, inner: {}", query_dim, context_dim, inner)
        self.to_q = eqx.nn.Linear(query_dim, inner, key=k1)#512,64
        self.to_kv = eqx.nn.Linear(context_dim, inner * 2, key=k2)
        self.to_out = eqx.nn.Linear(inner, query_dim, key=k3)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, *, context=None, mask=None, return_attn=False, key=None):
        h = self.heads
        context = default(context, x)
        
        # Handle both 2D (n, d) and 3D (b, n, d) inputs
        is_2d = x.ndim == 2
        if is_2d:
            # Add batch dimension for consistent processing
            x = x[None, :, :]  # (1, n, d)
            context = context[None, :, :]
            if mask is not None:
                mask = mask[None, :]
        
        # vmap Linear layers over batch dimensions (b, n)
        q = jax.vmap(jax.vmap(self.to_q))(x)
        kv = jax.vmap(jax.vmap(self.to_kv))(context)
        k, v = jnp.split(kv, 2, axis=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )

        sim = jnp.einsum("bid,bjd->bij", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -jnp.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim = jnp.where(~mask, max_neg_value, sim)

        attn = jax.nn.softmax(sim, axis=-1)

        attn_out = attn
        attn = self.dropout(attn, key=key)

        out = jnp.einsum("bij,bjd->bid", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = jax.vmap(jax.vmap(self.to_out))(out)

        if is_2d:
            out = out[0]  # Remove batch dimension
            if return_attn:
                attn_out = rearrange(attn_out, "(b h) n d -> b h n d", h=h)[0]

        if return_attn:
            if not is_2d:
                attn_out = rearrange(attn_out, "(b h) n d -> b h n d", h=h)
            return out, attn_out
        return out


# =========================
# Perceiver
# =========================

class Perceiver(eqx.Module):
    latents: jnp.ndarray
    cross_attn: PreNorm
    cross_ff: PreNorm
    layers: Tuple[Tuple[PreNorm, PreNorm], ...]
    _proj_ln: Optional[eqx.nn.LayerNorm]
    _proj_linear: Optional[eqx.nn.Linear]
    _final_proj_head: bool

    def __init__( self, dim, latent_dim, output_dim, num_latents, depth,
        cross_heads=1, cross_dim_head=64, latent_heads=8, latent_dim_head=64, attn_dropout=0.0, ff_dropout=0.0, final_proj_head=False, *, key, ):
        keys = jax.random.split(key, depth + 5)
        self.latents = jax.random.normal(keys[0], (num_latents, latent_dim))

        self.cross_attn = PreNorm( latent_dim,
            PerAttention( latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout, key=keys[1], ),
            context_dim=dim, )
        self.cross_ff = PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout, key=keys[2]),
        )

        self.layers = tuple(
            (
                PreNorm(
                    latent_dim,
                    PerAttention(
                        latent_dim,
                        heads=latent_heads,
                        dim_head=latent_dim_head,
                        dropout=attn_dropout,
                        key=keys[i + 3],
                    ),
                ),
                PreNorm(
                    latent_dim,
                    FeedForward(latent_dim, dropout=ff_dropout, key=keys[i + 4]),
                ),
            )
            for i in range(depth)
        )

        self._proj_ln = eqx.nn.LayerNorm(latent_dim) if final_proj_head else None
        self._proj_linear = eqx.nn.Linear(latent_dim, output_dim, key=keys[-1]) if final_proj_head else None
        self._final_proj_head = final_proj_head

    def _apply_proj_head(self, x):
        if not self._final_proj_head:
            return x
        x = jnp.mean(x, axis=1)  # (b, n, d) -> (b, d)
        x = jax.vmap(self._proj_ln)(x)  # vmap LayerNorm over batch
        x = jax.vmap(self._proj_linear)(x)  # vmap Linear over batch
        return x

    def __call__(self, data, mask=None, return_cross_attn=False, *, key=None):
        b = data.shape[0]
        x = repeat(self.latents, "n d -> b n d", b=b)

        if return_cross_attn:
            x_attn, attn = self.cross_attn(
                x, context=data, mask=mask, return_attn=True
            )
            x = x + x_attn
        else:
            x = x + self.cross_attn(x, context=data, mask=mask)

        x = x + self.cross_ff(x)

        for self_attn, self_ff in self.layers:
            x = x + self_attn(x)
            x = x + self_ff(x)

        if return_cross_attn:
            return self._apply_proj_head(x), attn

        return self._apply_proj_head(x)


# =========================
# SequentialActionEmb
# =========================

class SequentialActionEmb(eqx.Module):
    token_emb: eqx.Module
    timestep_emb: eqx.nn.Embedding
    agent_emb: eqx.nn.Embedding
    perceiver: Perceiver

    def __init__( self, num_agents, num_steps_conditioning, action_dim, is_continuous_act, perceiver_cfg: PerceiverConfig, *, key, ):
        k = jax.random.split(key, 6)

        if is_continuous_act:
            self.token_emb = eqx.nn.Sequential(
                [
                    eqx.nn.Linear(action_dim, perceiver_cfg.dim, key=k[0]),
                    eqx.nn.Lambda(jax.nn.silu),
                    eqx.nn.Linear(perceiver_cfg.dim, perceiver_cfg.dim, key=k[1]),
                ]
            )
        else:
            self.token_emb = eqx.nn.Embedding(action_dim, perceiver_cfg.dim, key=k[3])

        self.timestep_emb = eqx.nn.Embedding( num_steps_conditioning, perceiver_cfg.dim, key=k[4] )
        self.agent_emb = eqx.nn.Embedding(num_agents, perceiver_cfg.dim, key=k[5])

        self.perceiver = Perceiver(**perceiver_cfg.__dict__, key=k[2])

    def __call__(self, x, mask=None, return_cross_attn=False,  key=None):
        b, t, n, d = x.shape
        # jax.debug.print("SequentialActionEmb input x shape: {}", x.shape)#torch.Size([64, 3, 2, 3])
        # x = jax.vmap(jax.vmap(jax.vmap(self.token_emb)))(x)
        x_flat = x.reshape(-1, d)
        key,keys = jax.random.split(key,2)
        keys = jax.random.split(keys, x_flat.shape[0])
        x_flat = jax.vmap(self.token_emb)(x_flat)
        x = x_flat.reshape(b, t, n, -1)
        # x = self.token_emb(x)
        agent = jax.vmap(self.agent_emb)(jnp.arange(n))
        # agent = self.agent_emb(jnp.arange(n))
        x = rearrange(x, "b t n d -> (b t) n d") + agent[None]
        x = rearrange(x, "(b t) n d -> b t n d", b=b)

        # time = self.timestep_emb(jnp.arange(t))
        time = jax.vmap(self.timestep_emb)(jnp.arange(t))
        x = rearrange(x, "b t n d -> (b n) t d") + time[None]
        x = rearrange(x, "(b n) t d -> b t n d", b=b)

        x = rearrange(x, "b t n d -> b (t n) d")
        mask = rearrange(mask, "b t n -> b (t n)") if exists(mask) else None

        if return_cross_attn:
            out, attn = self.perceiver(
                x, mask=mask, return_cross_attn=True
            )
            return out, attn

        return self.perceiver(x, mask=mask), None


# =========================
# SimpleActionEncoder
# =========================

class SimpleActionEncoder(eqx.Module):
    token_emb: eqx.Module
    pos_emb: eqx.nn.Embedding
    layers: Tuple[Tuple[PreNorm, PreNorm], ...]
    to_out: eqx.Module

    def __init__(
        self,
        num_agents,
        action_dim,
        is_continuous_act,
        embed_dim,
        output_dim,
        num_heads,
        attn_dropout,
        ff_dropout,
        depth,
        *,
        key,
    ):
        k = jax.random.split(key, depth * 2 + 5)

        if is_continuous_act:
            self.token_emb = eqx.nn.Sequential(
                [
                    eqx.nn.Linear(action_dim, embed_dim, key=k[0]),
                    eqx.nn.Lambda(jax.nn.silu),
                    eqx.nn.Linear(embed_dim, embed_dim, key=k[1]),
                ]
            )
        else:
            self.token_emb = eqx.nn.Embedding(action_dim, embed_dim, key=k[2])

        self.pos_emb = eqx.nn.Embedding(num_agents, embed_dim, key=k[3])

        self.layers = tuple(
            (
                PreNorm(
                    embed_dim,
                    PerAttention(
                        embed_dim,
                        heads=num_heads,
                        dim_head=embed_dim // num_heads,
                        dropout=attn_dropout,
                        key=k[4 + i * 2],
                    ),
                ),
                PreNorm(
                    embed_dim,
                    FeedForward(embed_dim, dropout=ff_dropout, key=k[4 + i * 2 + 1]),
                ),
            )
            for i in range(depth)
        )

        self.to_out = eqx.nn.Sequential(
            [
                lambda x: jnp.mean(x, axis=1),
                eqx.nn.LayerNorm(embed_dim),
                eqx.nn.Linear(embed_dim, output_dim, key=k[-1]),
            ]
        )

    def __call__(self, x):
        # x shape: (num_agents, action_dim) when called via vmap
        # or (batch, num_agents, action_dim) otherwise
        is_2d = x.ndim == 2
        if is_2d:
            # Called via vmap - x is (num_agents, action_dim)
            n = x.shape[0]  # num_agents
            # Apply token_emb to each agent's action
            x = jax.vmap(self.token_emb)(x)  # (num_agents, embed_dim)
        else:
            # x is (batch, num_agents, action_dim)
            n = x.shape[1]
            # Apply token_emb to each (batch, agent) pair
            x = jax.vmap(jax.vmap(self.token_emb))(x)  # (batch, num_agents, embed_dim)
        
        x = x + jax.vmap(self.pos_emb)(jnp.arange(n))

        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)

        # Handle 2D output case - to_out expects axis=1 for num_agents
        if is_2d:
            # x is (num_agents, embed_dim), mean over axis=0 (agents)
            x = jnp.mean(x, axis=0)  # (embed_dim,)
            x = self.to_out.layers[1](x)  # LayerNorm
            x = self.to_out.layers[2](x)  # Linear
            return x
        else:
            return self.to_out(x)
