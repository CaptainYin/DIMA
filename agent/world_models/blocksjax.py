from functools import partial
import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


# =====================
# Constants
# =====================

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8


# =====================
# Convs (NCHW)
# =====================

def Conv1x1(in_ch, out_ch, *, key):
    return eqx.nn.Conv2d( in_ch, out_ch, kernel_size=1, stride=1, padding=0, key=key )


def Conv3x3(in_ch, out_ch, *, key):
    return eqx.nn.Conv2d( in_ch, out_ch, kernel_size=3, stride=1, padding=1, key=key )


# =====================
# GroupNorm
# =====================

class GroupNorm(eqx.Module):
    norm: eqx.nn.GroupNorm

    def __init__(self, in_channels: int, *, key=None):
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = eqx.nn.GroupNorm(groups=num_groups, channels=in_channels, eps=GN_EPS, )

    def __call__(self, x):
        return self.norm(x)


class AdaGroupNorm(eqx.Module):
    in_channels: int
    num_groups: int
    linear: eqx.nn.Linear

    def __init__(self, in_channels: int, cond_channels: int, *, key):
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = eqx.nn.Linear( cond_channels, in_channels * 2, key=key )
        self.norm = eqx.nn.GroupNorm(groups=self.num_groups, channels=self.in_channels, eps=GN_EPS,channelwise_affine=False)

    def __call__(self, x, cond):
        # x = eqx.nn.group_norm( x, self.num_groups, eps=GN_EPS )
        x = self.norm(x)
        scale_shift = self.linear(cond)
        scale, shift = jnp.split(scale_shift, 2, axis=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1.0 + scale) + shift


# =====================
# Self-Attention
# =====================

class SelfAttention2d(eqx.Module):
    n_head: int
    norm: GroupNorm
    qkv_proj: eqx.nn.Conv2d
    out_proj: eqx.nn.Conv2d

    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM, *, key):
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0

        k1, k2 = jax.random.split(key, 2)

        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3, key=k1)
        self.out_proj = Conv1x1(in_channels, in_channels, key=k2)

        self.out_proj = eqx.tree_at(
            lambda m: m.weight,
            self.out_proj,
            jnp.zeros_like(self.out_proj.weight),
        )
        self.out_proj = eqx.tree_at(
            lambda m: m.bias,
            self.out_proj,
            jnp.zeros_like(self.out_proj.bias),
        )

    def __call__(self, x):
        n, c, h, w = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(
            n, self.n_head * 3, c // self.n_head, h * w
        ).transpose(0, 1, 3, 2)

        q, k, v = jnp.split(qkv, 3, axis=1)

        att = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
        att = att / math.sqrt(k.shape[-1])
        att = jax.nn.softmax(att, axis=-1)

        y = jnp.matmul(att, v)
        y = y.transpose(0, 1, 3, 2).reshape(n, c, h, w)

        return x + self.out_proj(y)


# =====================
# Fourier Features
# =====================

class FourierFeatures(eqx.Module):
    # Frozen random features (not trainable), like PyTorch's register_buffer
    weight: jnp.ndarray = eqx.field(static=True)

    def __init__(self, cond_channels: int, *, key):
        assert cond_channels % 2 == 0
        object.__setattr__(self, 'weight', jax.random.normal(key, (1, cond_channels // 2)))

    def __call__(self, x):
        f = 2 * math.pi * x[:, None] @ self.weight
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


# =====================
# Down / Up sample
# =====================

class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, in_channels: int, *, key):
        self.conv = eqx.nn.Conv2d( in_channels, in_channels, kernel_size=3, stride=2, padding=1, key=key, )

    def __call__(self, x):
        return self.conv(x)


class Upsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, in_channels: int, *, key):
        self.conv = Conv3x3(in_channels, in_channels, key=key)

    def __call__(self, x):
        x = jax.image.resize( x, (x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2), method="nearest", )
        return self.conv(x)


# =====================
# SmallResBlock
# =====================

class SmallResBlock(eqx.Module):
    norm: GroupNorm
    conv: eqx.nn.Conv2d
    skip: eqx.Module

    def __init__(self, in_ch: int, out_ch: int, *, key):
        k1, k2 = jax.random.split(key)
        self.norm = GroupNorm(in_ch)
        self.conv = Conv3x3(in_ch, out_ch, key=k1)
        self.skip = (
            Conv1x1(in_ch, out_ch, key=k2)
            if in_ch != out_ch
            else eqx.nn.Identity()
        )

    def __call__(self, x):
        return self.skip(x) + self.conv(jax.nn.silu(self.norm(x)))


# =====================
# ResBlock
# =====================

class ResBlock(eqx.Module):
    proj: eqx.Module
    norm1: AdaGroupNorm
    conv1: eqx.nn.Conv2d
    norm2: AdaGroupNorm
    conv2: eqx.nn.Conv2d
    attn: eqx.Module

    def __init__(self, in_ch, out_ch, cond_ch, attn, *, key):
        k = jax.random.split(key, 6)
        self.proj = (
            Conv1x1(in_ch, out_ch, key=k[0])
            if in_ch != out_ch
            else eqx.nn.Identity()
        )
        self.norm1 = AdaGroupNorm(in_ch, cond_ch, key=k[1])
        self.conv1 = Conv3x3(in_ch, out_ch, key=k[2])
        self.norm2 = AdaGroupNorm(out_ch, cond_ch, key=k[3])
        self.conv2 = Conv3x3(out_ch, out_ch, key=k[4])
        self.attn = SelfAttention2d(out_ch, key=k[5]) if attn else eqx.nn.Identity()

        self.conv2 = eqx.tree_at(
            lambda m: m.weight,
            self.conv2,
            jnp.zeros_like(self.conv2.weight),
        )

    def __call__(self, x, cond):
        r = self.proj(x)
        x = self.conv1(jax.nn.silu(self.norm1(x, cond)))
        x = self.conv2(jax.nn.silu(self.norm2(x, cond)))
        x = x + r
        return self.attn(x)


# =====================
# ResBlocks
# =====================

class ResBlocks(eqx.Module):
    resblocks: Tuple[ResBlock, ...]

    def __init__(
        self,
        list_in: List[int],
        list_out: List[int],
        cond_ch: int,
        attn: bool,
        *,
        key,
    ):
        keys = jax.random.split(key, len(list_in))
        self.resblocks = tuple(
            ResBlock(i, o, cond_ch, attn, key=k)
            for i, o, k in zip(list_in, list_out, keys)
        )

    def __call__(self, x, cond, to_cat=None):
        outputs = []
        for i, block in enumerate(self.resblocks):
            if to_cat is not None:
                x = jnp.concatenate([x, to_cat[i]], axis=1)
            x = block(x, cond)
            outputs.append(x)
        return x, outputs


# =====================
# UNet
# =====================

class UNet(eqx.Module):
    d_blocks: Tuple[ResBlocks, ...]
    u_blocks: Tuple[ResBlocks, ...]
    mid_blocks: ResBlocks
    downsamples: Tuple[eqx.Module, ...]
    upsamples: Tuple[eqx.Module, ...]
    _num_down: int

    def __init__(self, cond_channels, depths, channels, attn_depths, *, key):
        self._num_down = len(channels) - 1
        keys = jax.random.split(key, 100)

        d_blocks, u_blocks = [], []
        ki = 0
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    [c1] + [c2] * (n - 1),
                    [c2] * n,
                    cond_channels,
                    attn_depths[i],
                    key=keys[ki],
                )
            )
            ki += 1
            u_blocks.append(
                ResBlocks(
                    [2 * c2] * n + [c1 + c2],
                    [c2] * n + [c1],
                    cond_channels,
                    attn_depths[i],
                    key=keys[ki],
                )
            )
            ki += 1

        self.d_blocks = tuple(d_blocks)
        self.u_blocks = tuple(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            [channels[-1]] * 2,
            [channels[-1]] * 2,
            cond_channels,
            True,
            key=keys[ki],
        )
        ki += 1

        self.downsamples = tuple(
            [eqx.nn.Identity()]
            + [Downsample(c, key=keys[ki + i]) for i, c in enumerate(channels[:-1])]
        )
        ki += len(channels)

        self.upsamples = tuple(
            [eqx.nn.Identity()]
            + [Upsample(c, key=keys[ki + i]) for i, c in enumerate(reversed(channels[:-1]))]
        )

    def __call__(self, x, cond):
        _, _, h, w = x.shape
        n = self._num_down
        pad_h = math.ceil(h / 2**n) * 2**n - h
        pad_w = math.ceil(w / 2**n) * 2**n - w
        x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, outs = block(x_down, cond)
            d_outputs.append((x_down, *outs))

        x, _ = self.mid_blocks(x, cond)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, outs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *outs))

        x = x[..., :h, :w]
        return x, d_outputs, u_outputs
