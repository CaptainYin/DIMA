import equinox as eqx
import jax
import jax.numpy as jnp
import math
from typing import Optional
from typing import List, Tuple

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
class SinusoidalPosEmb(eqx.Module):
    dim: int = eqx.static_field()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half = self.dim // 2
        freqs = jnp.exp( -math.log(10000) * jnp.arange(half) / (half - 1) )
        emb = x[:, None] * freqs[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

class AdaGroupNorm(eqx.Module):
    num_groups: int = eqx.static_field()
    channels: int = eqx.static_field()
    linear: eqx.nn.Linear
    norm: eqx.nn.GroupNorm = eqx.static_field()
    def __init__(self, channels, cond_dim, *, key):
        self.num_groups = max(1, channels // GN_GROUP_SIZE)
        self.channels = channels
        self.linear = eqx.nn.Linear(cond_dim, channels * 2, key=key)
        self.norm = eqx.nn.GroupNorm(groups=self.num_groups, channels=self.channels, eps=GN_EPS,channelwise_affine=False)

    def __call__(self, x, cond):
        h = self.norm(x)
        scale, shift = jnp.split(self.linear(cond), 2, axis=-1)
        return h * (1 + scale[..., None]) + shift[..., None]

class Downsample1d(eqx.Module):
    conv: eqx.nn.Conv1d

    def __init__(self, dim, *, key):
        self.conv = eqx.nn.Conv1d(dim, dim, 3, stride=2, padding=1, key=key)

    def __call__(self, x):
        return self.conv(x)
class Upsample1d(eqx.Module):
    conv: eqx.nn.ConvTranspose1d

    def __init__(self, dim, *, key):
        self.conv = eqx.nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1, key=key)

    def __call__(self, x):
        return self.conv(x)
    
class ResBlock(eqx.Module):
    proj: Optional[eqx.nn.Conv1d]
    norm1: AdaGroupNorm
    norm2: AdaGroupNorm
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    def __init__(self, in_channels, out_channels, cond_channels, attn: bool, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.proj = None if in_channels == out_channels else eqx.nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, key=k1)
        self.norm1 = AdaGroupNorm(in_channels, cond_channels, key=k2)
        
        self.conv1 = eqx.nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1, key=k3)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels, key=k4)
        self.conv2 = eqx.nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1, key=k5)

    def __call__(self, x, cond):
        r = x if self.proj is None else self.proj(x)
        x = self.conv1(jax.nn.silu(self.norm1(x, cond)))
        x = self.conv2(jax.nn.silu(self.norm2(x, cond)))
        return x + r


class ResBlocks(eqx.Module):
    resblocks: Tuple[ResBlock, ...]
    in_channels: int

    def __init__(
        self,
        *,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
        key: jax.random.PRNGKey,
    ):
        assert len(list_in_channels) == len(list_out_channels)

        self.in_channels = list_in_channels[0]

        keys = jax.random.split(key, len(list_in_channels))

        self.resblocks = tuple(
            ResBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                cond_channels=cond_channels,
                attn=attn,
                key=k,
            )
            for (in_ch, out_ch, k) in zip(
                list_in_channels,
                list_out_channels,
                keys,
            )
        )

    # -------------------------------------
    # forward
    # -------------------------------------
    def __call__(
        self,
        x: jnp.ndarray,
        cond: jnp.ndarray,
        to_cat: Optional[List[jnp.ndarray]] = None,
    ):
        """
        x:      (B, T, C)
        cond:   (B, cond_channels)
        to_cat: list of tensors for skip concat
        """
        outputs = []

        for i, block in enumerate(self.resblocks):
            if to_cat is not None:
                x = jnp.concatenate([x, to_cat[i]], axis=0)  # channel dim (C, L)

            x = block(x, cond)
            outputs.append(x)

        return x, outputs



class UNet1D(eqx.Module):
    d_blocks: Tuple
    u_blocks: Tuple
    mid_blocks: eqx.Module
    downsamples: Tuple
    upsamples: Tuple
    _num_down: int

    def __init__(
        self,
        *,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[bool],
        key: jax.random.PRNGKey,
    ):
        assert len(depths) == len(channels) == len(attn_depths)

        self._num_down = len(channels) - 1

        keys = jax.random.split(key, 5)
        d_key, u_key, mid_key, down_key, up_key = keys

        # -------------------------
        # Down / Up ResBlocks
        # -------------------------
        d_blocks = []
        u_blocks = []

        d_keys = jax.random.split(d_key, len(depths))
        u_keys = jax.random.split(u_key, len(depths))

        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]

            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                    key=d_keys[i],
                )
            )

            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                    key=u_keys[i],
                )
            )

        self.d_blocks = tuple(d_blocks)
        self.u_blocks = tuple(reversed(u_blocks))

        # -------------------------
        # Middle blocks
        # -------------------------
        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=False,
            key=mid_key,
        )

        # -------------------------
        # Down / Up sampling
        # -------------------------
        self.downsamples = tuple(
            [eqx.nn.Identity()]
            + [
                Downsample1d(c, key=k)
                for c, k in zip(channels[:-1], jax.random.split(down_key, len(channels) - 1))
            ]
        )

        self.upsamples = tuple(
            [eqx.nn.Identity()]
            + [
                Upsample1d(c, key=k)
                for c, k in zip(
                    reversed(channels[:-1]),
                    jax.random.split(up_key, len(channels) - 1),
                )
            ]
        )

    # ======================================================
    # Forward
    # ======================================================
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray):
        """
        x:    (B, T, C)
        cond: (B, cond_channels)
        """

        d_outputs = []

        # -------------------------
        # Down path
        # -------------------------
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        # -------------------------
        # Middle
        # -------------------------
        x, _ = self.mid_blocks(x, cond)

        # -------------------------
        # Up path
        # -------------------------
        u_outputs = []
        for block, up, skip in zip(
            self.u_blocks,
            self.upsamples,
            reversed(d_outputs),
        ):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, to_cat=skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        return x, d_outputs, u_outputs
