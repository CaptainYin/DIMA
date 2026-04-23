"""
JAX/Equinox implementation of Reward and End Prediction Models.
Compatible interface with PyTorch version for drop-in replacement.
"""

from typing import Optional, Tuple, NamedTuple, Dict, Any, List
from dataclasses import dataclass
import math
import numpy as np

import torch
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from jaxtyping import Array, Float, Bool
from einops import rearrange

# Import from JAX transformer modules
from .transformerjax import Transformer, get_sinusoid_encoding_table, Perceiver
from .perceiverjax import SimpleActionEncoder
from .perceiverjax import SimpleActionEncoder
from networks.toolsjax import HLGaussLoss, TwoHotCategoricalLoss, LOSSES_DICT


class TransRewEndModelOutput(NamedTuple):
    """Output container for TransRewEndModel."""
    output_sequence: Float[Array, "batch seq_len embed_dim"]
    pred_rewards: Float[Array, "batch seq_len 1"]
    logits_ends: Float[Array, "batch seq_len 1_or_2"]
    pred_avail_action: Optional[Float[Array, "batch seq_len num_agents action_dim"]]
    attn_output: Optional[List] = None


class GroupNorm1d(eqx.Module):
    """Group normalization for 1D sequences."""
    num_groups: int = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    weight: Float[Array, "num_channels"]
    bias: Float[Array, "num_channels"]

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = jnp.ones(num_channels)
        self.bias = jnp.zeros(num_channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        original_shape = x.shape
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])
        
        B, T, C = x.shape
        x = x.reshape(B, T, self.num_groups, C // self.num_groups)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(B, T, C)
        x = x * self.weight + self.bias
        
        if len(original_shape) == 2:
            x = x.reshape(original_shape)
        return x


class ChiResidualBlock(eqx.Module):
    """Residual block with SiLU activation."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm1: GroupNorm1d
    norm2: GroupNorm1d
    skip: Optional[eqx.nn.Linear]

    def __init__(self, in_dim: int, out_dim: int, num_groups: int = 8, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(in_dim, out_dim, key=keys[0])
        self.linear2 = eqx.nn.Linear(out_dim, out_dim, key=keys[1])
        self.norm1 = GroupNorm1d(min(num_groups, out_dim), out_dim)
        self.norm2 = GroupNorm1d(min(num_groups, out_dim), out_dim)
        self.skip = eqx.nn.Linear(in_dim, out_dim, key=keys[2]) if in_dim != out_dim else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = self.skip(x) if self.skip is not None else x
        x = self.linear1(x)
        x = self.norm1(x)
        x = jax.nn.silu(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = jax.nn.silu(x)
        return x + residual



class RewEndEncoder1D_Chi(eqx.Module):
    """1D encoder for reward/end prediction with residual blocks."""
    blocks: list
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_blocks: int = 3,
        *,
        key: jax.random.PRNGKey
    ):
        keys = jax.random.split(key, num_blocks + 1)
        
        blocks = []
        curr_dim = in_dim
        for i in range(num_blocks):
            blocks.append(ChiResidualBlock(curr_dim, hidden_dim, key=keys[i]))
            curr_dim = hidden_dim
        self.blocks = blocks
        self.out_proj = eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


@dataclass
class TransformerConfigJax:
    """Config for TransRewEndModel JAX version."""
    tokens_per_block: int = 2
    max_blocks: int = 64
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embed_dropout: float = 0.1
    
    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks

@dataclass
class StateRewEndModelConfig:
    lstm_dim: 512
    cond_channels: int
    depths: List[int]
    dim: int
    dim_mults: List[int]
    attn_depths: List[bool]
    mlp_dim: int
class RewEndModel(eqx.Module):
    """Reward and End prediction model using GRU-style encoder.
    Compatible interface with PyTorch version."""
    encoder: RewEndEncoder1D_Chi
    act_emb: SimpleActionEncoder
    gru_cell: eqx.nn.GRUCell
    reward_head: eqx.nn.Sequential
    con_head: eqx.nn.Sequential
    av_action_head: Optional[eqx.nn.Sequential]
    
    num_agents: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    is_continuous_act: bool = eqx.field(static=True)
    pred_shared_reward: bool = eqx.field(static=True)
    pred_shared_continuation: bool = eqx.field(static=True)
    pred_av_action: bool = eqx.field(static=True)
    use_ce_for_cont: bool = eqx.field(static=True)
    cond_channels: int = eqx.field(static=True)
    lstm_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(
        self,
        cfg:StateRewEndModelConfig, 
        num_agents: int,
        state_dim: int,
        action_dim: int,
        is_continuous_act: bool,
        pred_shared_reward: bool = True,
        pred_shared_continuation: bool = True,
        pred_av_action: bool = False,
        use_ce_for_cont: bool = False,
        *,
        key: jax.random.PRNGKey = None,
        **kwargs,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
            
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.is_continuous_act = is_continuous_act
        self.pred_shared_reward = pred_shared_reward
        self.pred_shared_continuation = pred_shared_continuation
        self.pred_av_action = pred_av_action
        self.use_ce_for_cont = use_ce_for_cont
        self.cond_channels = getattr(cfg, 'cond_channels', 128)
        self.lstm_dim = getattr(cfg, 'lstm_dim', 512)
        self.latent_dim = 256
        
        keys = jax.random.split(key, 10)
        
        # Encoder for state (matching RewEndEncoder1D_Chi from PyTorch)
        self.encoder = RewEndEncoder1D_Chi(
            in_dim=state_dim, 
            hidden_dim=128, 
            out_dim=self.latent_dim, 
            num_blocks=2, 
            key=keys[0]
        )
        
        # Action embedding (matching SimpleActionEncoder from PyTorch)
        self.act_emb = SimpleActionEncoder(
            num_agents=num_agents, 
            action_dim=action_dim, 
            is_continuous_act=is_continuous_act,
            output_dim=self.cond_channels, 
            embed_dim=256, 
            depth=3, 
            key=keys[1]
        )
        
        # GRU cell (matching nn.GRUCell from PyTorch)
        self.gru_cell = eqx.nn.GRUCell(self.latent_dim, self.lstm_dim, key=keys[2])
        
        # Reward head (matching PyTorch structure)
        if pred_shared_reward:
            self.reward_head = eqx.nn.Sequential([
                eqx.nn.Linear(self.lstm_dim, self.lstm_dim, key=keys[3]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, self.lstm_dim, key=jax.random.split(keys[3])[0]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, 1, use_bias=False, key=jax.random.split(keys[3])[1]),
            ])
        else:
            self.reward_head = eqx.nn.Sequential([
                eqx.nn.Linear(self.lstm_dim + num_agents, self.lstm_dim, key=keys[3]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, self.lstm_dim, key=jax.random.split(keys[3])[0]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, 1, use_bias=False, key=jax.random.split(keys[3])[1]),
            ])
        
        # Continuation head (matching PyTorch structure)
        con_head_output_dim = 2 if use_ce_for_cont else 1
        if pred_shared_continuation:
            self.con_head = eqx.nn.Sequential([
                eqx.nn.Linear(self.lstm_dim, self.lstm_dim, key=keys[4]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, self.lstm_dim, key=jax.random.split(keys[4])[0]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, con_head_output_dim, use_bias=False, key=jax.random.split(keys[4])[1]),
            ])
        else:
            self.con_head = eqx.nn.Sequential([
                eqx.nn.Linear(self.lstm_dim + num_agents, self.lstm_dim, key=keys[4]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, self.lstm_dim, key=jax.random.split(keys[4])[0]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, con_head_output_dim, use_bias=False, key=jax.random.split(keys[4])[1]),
            ])
        
        # Available action head (optional, matching PyTorch structure)
        if pred_av_action:
            self.av_action_head = eqx.nn.Sequential([
                eqx.nn.Linear(self.lstm_dim + num_agents, self.lstm_dim, key=keys[5]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, self.lstm_dim, key=jax.random.split(keys[5])[0]),
                eqx.nn.Lambda(lambda x: jax.nn.silu(x)),
                eqx.nn.Linear(self.lstm_dim, 2 * action_dim, use_bias=False, key=jax.random.split(keys[5])[1]),
            ])
        else:
            self.av_action_head = None

    def predict_rew_end(
        self, 
        state: jnp.ndarray, 
        act: jnp.ndarray, 
        next_state: jnp.ndarray,
        hx: Optional[jnp.ndarray] = None,
        done: jnp.ndarray = None
    ):
        """Predict reward, end logits, and available actions.
        
        Args:
            state: (B, T, state_dim)
            act: (B, T, num_agents, action_dim)
            next_state: (B, T, state_dim)
            hx: Optional hidden state (B, lstm_dim)
            done: Optional done flags (B, T)
        """
        B, T, N = act.shape[:3]
        
        # Reshape for processing
        state_flat = state.reshape(B * T, -1)
        act_flat = act.reshape(B * T, N, -1)
        next_state_flat = next_state.reshape(B * T, -1)
        
        if not self.is_continuous_act:
            act_flat = jnp.argmax(act_flat, axis=-1)
        
        # Get action conditioning
        act_cond = jax.vmap(self.act_emb)(act_flat)  # (B*T, cond_channels)
        
        # Stack state and next_state, then encode
        # In PyTorch: x = torch.stack((state, next_state), dim=1)
        # Then rearrange to (B*T, 2, state_dim) -> encoder expects this
        x = jnp.stack([state_flat, next_state_flat], axis=1)  # (B*T, 2, state_dim)
        
        # Encode (encoder processes the stacked states)
        x = jax.vmap(self.encoder)(x)  # (B*T, latent_dim)
        x = x.reshape(B, T, -1)  # (B, T, latent_dim)
        
        # Initialize hidden state if not provided
        if hx is None:
            hx = jnp.zeros((B, self.lstm_dim))
        
        # Process through GRU sequentially
        outputs = []
        for i in range(T):
            hx = self.gru_cell(x[:, i], hx)
            outputs.append(hx)
            if done is not None:
                # Reset hidden state on done (with detach equivalent)
                hx = hx * (1 - done[:, i].astype(jnp.float32))[:, None]
                hx = jax.lax.stop_gradient(hx)
        
        x = jnp.stack(outputs, axis=1)  # (B, T, lstm_dim)
        
        # Create agent-wise features with agent IDs
        agent_id = jnp.eye(N, dtype=jnp.float32)  # (N, N)
        agent_id = jnp.broadcast_to(agent_id[None, None, :, :], (B, T, N, N))  # (B, T, N, N)
        agent_x = jnp.concatenate([
            jnp.broadcast_to(x[:, :, None, :], (B, T, N, self.lstm_dim)),
            agent_id
        ], axis=-1)  # (B, T, N, lstm_dim + N)
        
        # Predict reward
        if self.pred_shared_reward:
            pred_r = jax.vmap(jax.vmap(self.reward_head))(x)  # (B, T, 1)
        else:
            pred_r = jax.vmap(jax.vmap(jax.vmap(self.reward_head)))(agent_x)  # (B, T, N, 1)
        
        # Predict continuation/end logits
        if self.pred_shared_continuation:
            logits_e = jax.vmap(jax.vmap(self.con_head))(x)  # (B, T, 1 or 2)
        else:
            logits_e = jax.vmap(jax.vmap(jax.vmap(self.con_head)))(agent_x)  # (B, T, N, 1 or 2)
        
        # Predict available actions
        if self.pred_av_action and self.av_action_head is not None:
            logits_av = jax.vmap(jax.vmap(jax.vmap(self.av_action_head)))(agent_x)  # (B, T, N, 2*action_dim)
            logits_av = logits_av.reshape(B, T, N, self.action_dim, 2)
        else:
            logits_av = None
        
        return pred_r, logits_e, logits_av, hx

    def compute_av_action(self, state: jnp.ndarray, action: jnp.ndarray, next_state: jnp.ndarray = None):
        """Compute available action predictions."""
        if next_state is None:
            next_state = state
            
        if state.ndim == 2:
            state = state[:, None, :]
            action = action[:, None, :, :]
            next_state = next_state[:, None, :]
            
        _, _, logits_av, _ = self.predict_rew_end(state, action, next_state)
        if logits_av is not None:
            return jax.nn.softmax(logits_av.squeeze(1), axis=-1)[..., 1]  # Return prob of available
        return None

    def __call__(self, batch, gamma=0.997, contdisc=True, **kwargs):
        """Compute loss matching PyTorch interface."""
        state = batch['shared_obs']
        act = batch['action']
        next_state = batch['next_shared_obs']
        rew = batch['reward']
        end = batch['done']
        
        B, T = state.shape[:2]
        
        pred_r, logits_e, logits_av_action, _ = self.predict_rew_end(
            state, act, next_state, 
            done=end.all(axis=-2).squeeze(-1) if end.ndim == 4 else end
        )
        
        # Reward loss
        if self.pred_shared_reward:
            loss_rew = jnp.mean(jnp.abs(pred_r - rew.mean(axis=2)))
        else:
            loss_rew = jnp.mean(jnp.abs(pred_r - rew))
        
        # End/continuation loss
        if self.use_ce_for_cont:
            if self.pred_shared_continuation:
                target_end = end.all(axis=2).astype(jnp.int32).reshape(-1)
                logits_flat = logits_e.reshape(-1, 2)
            else:
                target_end = end.astype(jnp.int32).reshape(-1)
                logits_flat = logits_e.reshape(-1, 2)
            loss_end = -jnp.mean(jnp.sum(jax.nn.log_softmax(logits_flat) * jax.nn.one_hot(target_end, 2), axis=-1))
        else:
            # Compute continuation target
            if self.pred_shared_continuation:
                is_terminal = end.all(axis=2)
            else:
                is_terminal = end.astype(jnp.bool_)
            target_con = (~is_terminal).astype(jnp.float32)
            if contdisc:
                target_con = target_con * gamma
            loss_end = jnp.mean(jax.nn.sigmoid_cross_entropy(logits_e, target_con))
        
        # Available action loss
        if self.pred_av_action and logits_av_action is not None and 'av_action' in batch:
            # Roll done flags to mask first timestep after done
            tmp = jnp.roll(batch['done'], 1, axis=1).squeeze(-1)
            labels_av_actions = batch['av_action']
            # Set labels to 1 (available) for timesteps after done
            labels_av_actions = jnp.where(
                tmp.all(axis=-1, keepdims=True)[..., None],
                jnp.ones_like(labels_av_actions),
                labels_av_actions
            )
            
            logits_flat = rearrange(logits_av_action[:, :-1], 'b l n a e -> (b l n a) e')
            labels_flat = rearrange(labels_av_actions[:, 1:], 'b l n a -> (b l n a)').astype(jnp.int32)
            loss_av_action = -jnp.mean(jnp.sum(
                jax.nn.log_softmax(logits_flat) * jax.nn.one_hot(labels_flat, 2), axis=-1
            ))
        else:
            loss_av_action = jnp.array(0.0)
        
        loss = loss_rew + loss_end + loss_av_action
        
        metrics = {
            "loss_rew": float(loss_rew),
            "loss_end": float(loss_end),
            "loss_av_action": float(loss_av_action) if self.pred_av_action else 0.0,
            "loss_total": float(loss),
        }
        
        return loss, metrics


class TransRewEndModel(eqx.Module):
    """Transformer-based Reward and End prediction model.
    Compatible interface with PyTorch version."""
    
    # Embeddings
    embedder: eqx.nn.Sequential
    act_emb: SimpleActionEncoder
    pos_emb: eqx.nn.Embedding
    
    # Transformer
    transformer: Transformer
    
    # Heads
    head_rewards: eqx.nn.Sequential
    head_ends: eqx.nn.Sequential
    heads_avail_actions: Optional[eqx.Module]
    
    # Reward loss for CE mode
    reward_loss: Optional[Any] = None
    
    # Config
    config: TransformerConfigJax = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    act_vocab_size: int = eqx.field(static=True)
    num_agents: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    is_discrete_action: bool = eqx.field(static=True)
    use_symlog: bool = eqx.field(static=True)
    use_ce_for_end: bool = eqx.field(static=True)
    use_ce_for_reward: bool = eqx.field(static=True)
    use_ce_for_av_action: bool = eqx.field(static=True)
    enable_av_pred: bool = eqx.field(static=True)
    pred_av_action: bool = eqx.field(static=True)
    num_action_tokens: int = eqx.field(static=True)
    num_obs_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        act_vocab_size: int,
        num_agents: int,
        config,  # TransformerConfig object
        action_dim: int,
        is_discrete_action: bool,
        use_symlog: bool = False,
        use_ce_for_end: bool = False,
        use_ce_for_av_action: bool = True,
        enable_av_pred: bool = False,
        use_ce_for_reward: bool = False,
        rewards_prediction_config: dict = None,
        *,
        key: jax.random.PRNGKey = None,
        **kwargs,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Handle config - could be dataclass or object
        if hasattr(config, 'embed_dim'):
            embed_dim = config.embed_dim
            tokens_per_block = config.tokens_per_block
            max_blocks = config.max_blocks
            max_tokens = config.max_tokens
            num_heads = getattr(config, 'num_heads', 8)
            num_layers = getattr(config, 'num_layers', 4)
            attention_dropout = getattr(config, 'attention_dropout', 0.1)
            residual_dropout = getattr(config, 'residual_dropout', 0.1)
            embed_dropout = getattr(config, 'embed_dropout', 0.1)
        else:
            embed_dim = 256
            tokens_per_block = 2
            max_blocks = 64
            max_tokens = tokens_per_block * max_blocks
            num_heads = 8
            num_layers = 4
            attention_dropout = 0.1
            residual_dropout = 0.1
            embed_dropout = 0.1
        
        # Store as JAX config
        self.config = TransformerConfigJax(
            tokens_per_block=tokens_per_block,
            max_blocks=max_blocks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embed_dropout=embed_dropout,
        )
        
        self.state_dim = state_dim
        self.act_vocab_size = act_vocab_size
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.is_discrete_action = is_discrete_action
        self.use_symlog = use_symlog or use_ce_for_reward  # symlog enabled when using CE for reward
        self.use_ce_for_end = use_ce_for_end
        self.use_ce_for_reward = use_ce_for_reward
        self.use_ce_for_av_action = use_ce_for_av_action
        self.enable_av_pred = enable_av_pred
        self.pred_av_action = enable_av_pred
        
        self.num_action_tokens = 1
        self.num_obs_tokens = tokens_per_block - self.num_action_tokens
        
        keys = jax.random.split(key, 15)
        
        # State embedder (matching PyTorch: GeneralEmbedder)
        self.embedder = eqx.nn.Sequential([
            eqx.nn.Linear(state_dim, embed_dim, key=keys[0]),
            eqx.nn.LayerNorm(embed_dim),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(embed_dim, embed_dim, key=keys[1]),
            eqx.nn.LayerNorm(embed_dim),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(embed_dim, embed_dim, key=keys[2]),
        ])
        
        # Action embedder (matching PyTorch: SimpleActionEncoder with perceiver attention)
        self.act_emb = SimpleActionEncoder(
            num_agents=num_agents, 
            action_dim=action_dim, 
            is_continuous_act=not is_discrete_action,
            embed_dim=256,
            output_dim=embed_dim, 
            num_heads=8, 
            attn_dropout=0., 
            ff_dropout=0., 
            depth=3, 
            key=keys[3]
        )
        
        # Position embedding
        self.pos_emb = eqx.nn.Embedding(max_tokens, embed_dim, key=keys[4])
        
        # Transformer (matching PyTorch Transformer)
        self.transformer = Transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=embed_dim * 4,
            max_seq_len=max_tokens,
            dropout=attention_dropout,
            key=keys[5]
        )
        
        # Reward head (matching PyTorch structure with Head)
        if use_ce_for_reward:
            assert rewards_prediction_config is not None
            bin_width = (rewards_prediction_config["max_v"] - rewards_prediction_config["min_v"]) / rewards_prediction_config["bins"]
            self.reward_loss = LOSSES_DICT[rewards_prediction_config["loss_type"]](
                min_value=rewards_prediction_config["min_v"],
                max_value=rewards_prediction_config["max_v"],
                num_bins=rewards_prediction_config["bins"],
                sigma=bin_width * 0.75
            )
            reward_out_dim = self.reward_loss.output_dim
        else:
            self.reward_loss = None
            reward_out_dim = 1
            
        self.head_rewards = eqx.nn.Sequential([
            eqx.nn.Linear(embed_dim, embed_dim, key=keys[6]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(embed_dim, embed_dim, key=keys[7]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(embed_dim, reward_out_dim, key=keys[8]),
        ])
        
        # End head (matching PyTorch structure)
        end_out_dim = 2 if use_ce_for_end else 1
        self.head_ends = eqx.nn.Sequential([
            eqx.nn.Linear(embed_dim, embed_dim, key=keys[9]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(embed_dim, embed_dim, key=keys[10]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(embed_dim, end_out_dim, key=keys[11]),
        ])
        
        # Available action head (matching PyTorch AgentWiseHead with DiscreteDist)
        if enable_av_pred:
            if use_ce_for_av_action:
                # Match PyTorch: AgentWiseHead with DiscreteDist
                self.heads_avail_actions = eqx.nn.Sequential([
                    eqx.nn.Linear(embed_dim + num_agents, embed_dim, key=keys[12]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(embed_dim, embed_dim, key=keys[13]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(embed_dim, action_dim * 2, key=keys[14]),  # 2 classes per action
                ])
            else:
                self.heads_avail_actions = eqx.nn.Sequential([
                    eqx.nn.Linear(embed_dim, embed_dim, key=keys[12]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(embed_dim, embed_dim, key=keys[13]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(embed_dim, action_dim, key=keys[14]),
                ])
        else:
            self.heads_avail_actions = None

    def forward(
        self, 
        tokens: jnp.ndarray, 
        perattn_out: jnp.ndarray = None,
        past_keys_values = None,
        return_attn: bool = False,
        attention_mask: jnp.ndarray = None
    ) -> TransRewEndModelOutput:
        """Forward pass matching PyTorch TransRewEndModel.forward().
        
        Args:
            tokens: (B, L*2, state_dim) - interleaved obs tokens
            perattn_out: (B, L, embed_dim) - action embeddings to insert
            past_keys_values: Optional KV cache dict with 'keys' and 'values'
            return_attn: Whether to return attention weights
            attention_mask: Optional attention mask
            
        Returns:
            TransRewEndModelOutput with predictions
        """
        # Handle torch tensor inputs - convert to JAX
        input_device = None
        is_torch_input = isinstance(tokens, torch.Tensor)
        if is_torch_input:
            input_device = tokens.device
            tokens = jnp.array(tokens.detach().cpu().numpy())
            if perattn_out is not None:
                perattn_out = jnp.array(perattn_out.detach().cpu().numpy())
            if attention_mask is not None:
                attention_mask = jnp.array(attention_mask.detach().cpu().numpy())
            # Ignore PyTorch KV cache and attention mask - not compatible with JAX transformer
            past_keys_values = None
            attention_mask = None  # Also ignore attention mask for torch inputs
        elif attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                attention_mask = jnp.array(attention_mask.detach().cpu().numpy())
        
        B = tokens.shape[0]
        num_steps = tokens.shape[1]
        
        # Determine previous steps from KV cache
        prev_steps = 0
        if past_keys_values is not None and 'keys' in past_keys_values:
            # past_keys_values['keys'] shape: (num_layers, B, num_heads, prev_seq_len, head_dim)
            prev_steps = past_keys_values['keys'].shape[3] if past_keys_values['keys'] is not None else 0
        
        assert num_steps + prev_steps <= self.config.max_tokens, \
            f"Total sequence length {num_steps + prev_steps} exceeds max_tokens {self.config.max_tokens}"
        
        # Embed observation tokens (every other position starting from 0)
        # tokens shape: (B, L*2, state_dim) where even indices are obs, odd are placeholders
        obs_indices = jnp.arange(0, num_steps, 2)
        act_indices = jnp.arange(1, num_steps, 2)
        
        # Embed observations at even positions
        obs_tokens = tokens[:, obs_indices, :]  # (B, L, state_dim)
        sequences = jnp.zeros((B, num_steps, self.config.embed_dim))
        
        # Apply embedder to observation tokens
        obs_emb = jax.vmap(jax.vmap(self.embedder))(obs_tokens)  # (B, L, embed_dim)
        sequences = sequences.at[:, obs_indices].set(obs_emb)
        
        # Insert action embeddings at odd positions
        if perattn_out is not None:
            assert len(act_indices) == perattn_out.shape[1], \
                f"perattn_out length {perattn_out.shape[1]} doesn't match act_indices length {len(act_indices)}"
            sequences = sequences.at[:, act_indices].set(perattn_out)
        
        # Add position embeddings (accounting for previous steps from KV cache)
        positions = prev_steps + jnp.arange(num_steps)
        pos_emb = jax.vmap(self.pos_emb)(positions)  # (num_steps, embed_dim)
        sequences = sequences + pos_emb[None, :, :]  # broadcast to (B, num_steps, embed_dim)
        
        # Apply transformer with KV cache support
        x, attn_output = self.transformer(
            sequences, 
            kv_state=past_keys_values,
            attn_mask=attention_mask
        )
        
        # Extract outputs at action token positions (odd indices - last token of each block)
        # In PyTorch: all_but_last_pattern selects the last token of each block
        out_tokens = x[:, act_indices]  # (B, L, embed_dim)
        
        # Predict rewards
        pred_rewards = jax.vmap(jax.vmap(self.head_rewards))(out_tokens)  # (B, L, reward_out_dim)
        
        # Predict ends
        logits_ends = jax.vmap(jax.vmap(self.head_ends))(out_tokens)  # (B, L, 1 or 2)
        
        # Predict available actions (agent-wise)
        if self.enable_av_pred and self.heads_avail_actions is not None:
            L = out_tokens.shape[1]
            # Create agent-wise features with agent IDs
            agent_id = jnp.eye(self.num_agents, dtype=jnp.float32)  # (N, N)
            agent_id = jnp.broadcast_to(agent_id[None, None, :, :], (B, L, self.num_agents, self.num_agents))
            
            # Expand out_tokens for each agent
            out_expanded = jnp.broadcast_to(
                out_tokens[:, :, None, :], 
                (B, L, self.num_agents, self.config.embed_dim)
            )  # (B, L, N, embed_dim)
            
            # Concatenate with agent IDs
            agent_x = jnp.concatenate([out_expanded, agent_id], axis=-1)  # (B, L, N, embed_dim + N)
            
            # Apply head
            logits_av = jax.vmap(jax.vmap(jax.vmap(self.heads_avail_actions)))(agent_x)  # (B, L, N, action_dim*2)
            logits_av = logits_av.reshape(B, L, self.num_agents, self.action_dim, 2)
        else:
            logits_av = None
        
        # Convert outputs back to torch if inputs were torch
        if is_torch_input and input_device is not None:
            x = torch.from_numpy(np.array(x)).to(input_device)
            pred_rewards = torch.from_numpy(np.array(pred_rewards)).to(input_device)
            logits_ends = torch.from_numpy(np.array(logits_ends)).to(input_device)
            if logits_av is not None:
                logits_av = torch.from_numpy(np.array(logits_av)).to(input_device)
        
        return TransRewEndModelOutput(
            output_sequence=x,
            pred_rewards=pred_rewards,
            logits_ends=logits_ends,
            pred_avail_action=logits_av,
            attn_output=attn_output if return_attn else None
        )

    def __call__(
        self, 
        tokens: jnp.ndarray, 
        perattn_out: jnp.ndarray = None,
        past_keys_values = None,
        return_attn: bool = False,
        attention_mask: jnp.ndarray = None
    ) -> TransRewEndModelOutput:
        """Alias for forward() to match PyTorch calling convention."""
        return self.forward(tokens, perattn_out, past_keys_values, return_attn, attention_mask)

    def get_act_emb(self, action: jnp.ndarray) -> jnp.ndarray:
        """Get action embeddings matching PyTorch version."""
        if self.is_discrete_action and action.ndim == 3:
            action = jnp.argmax(action, axis=-1)
        return jax.vmap(self.act_emb)(action)

    # def __call__(
    #     self, 
    #     state_or_tokens: jnp.ndarray, 
    #     action_or_perattn: jnp.ndarray = None,
    #     next_state_or_kv = None,
    #     hx_or_return_attn = None,
    #     done_or_attn_mask = None
    # ):
    #     """Forward pass - auto-detects calling convention.
        
    #     Two calling conventions:
    #     1. Transformer forward: (tokens, perattn_out, past_keys_values, return_attn, attention_mask)
    #        - tokens has shape [B, L*2, state_dim] (interleaved)
    #     2. Environment step: (state, action) or (state, action, next_state, hx, done)
    #        - state has shape [B, 1, state_dim] or [B, state_dim]
    #        - action has shape [B, num_agents, action_dim] or [B, 1, num_agents, action_dim]
    #        - Returns (rew, pcont, end)
    #     """
    #     # Detect calling convention based on second argument type
    #     # If action_or_perattn has shape [..., num_agents, action_dim], it's env step mode
    #     if action_or_perattn is not None:
    #         # Check if this looks like an action tensor (has num_agents dimension)
    #         if action_or_perattn.ndim >= 2 and action_or_perattn.shape[-2] == self.num_agents:
    #             # Environment step mode
    #             return self.predict_rew_end(
    #                 state_or_tokens, 
    #                 action_or_perattn,
    #                 next_state=next_state_or_kv if isinstance(next_state_or_kv, jnp.ndarray) else None,
    #                 hx=hx_or_return_attn if isinstance(hx_or_return_attn, jnp.ndarray) else None,
    #                 done=done_or_attn_mask if isinstance(done_or_attn_mask, jnp.ndarray) else None
    #             )
        
    #     # Transformer forward mode
    #     return self.forward(
    #         state_or_tokens, 
    #         action_or_perattn, 
    #         next_state_or_kv, 
    #         hx_or_return_attn if isinstance(hx_or_return_attn, bool) else False, 
    #         done_or_attn_mask
    #     )

    # def predict_rew_end(
    #     self, 
    #     state, 
    #     act, 
    #     next_state = None,
    #     hx = None,
    #     done = None
    # ):
    #     """Predict reward, continuation, and end for environment stepping.
        
    #     Args:
    #         state: (B, state_dim) or (B, 1, state_dim) - can be torch.Tensor or jnp.ndarray
    #         act: (B, num_agents, action_dim) or (B, 1, num_agents, action_dim) - can be torch.Tensor or jnp.ndarray
    #         next_state: Optional, not used in transformer version
    #         hx: Optional hidden state (not used, for compatibility)
    #         done: Optional done flags (not used)
            
    #     Returns:
    #         Tuple of (pred_rewards, pred_pcont, logits_ends)
    #     """
    #     # Convert torch tensors to jax arrays
    #     is_torch = isinstance(state, torch.Tensor)
    #     input_device = None
    #     if is_torch:
    #         input_device = state.device
    #         state = jnp.array(state.detach().cpu().numpy())
    #         act = jnp.array(act.detach().cpu().numpy())
        
    #     # Ensure proper shapes
    #     if state.ndim == 2:
    #         state = state[:, None, :]  # (B, 1, state_dim)
    #     if act.ndim == 3:
    #         act = act[:, None, :, :]  # (B, 1, num_agents, action_dim)
        
    #     B, L = state.shape[:2]
        
    #     # Get action embeddings
    #     action_flat = act.reshape(B * L, self.num_agents, -1)
    #     if self.is_discrete_action:
    #         action_flat = jnp.argmax(action_flat, axis=-1)
    #     act_cond = jax.vmap(self.act_emb)(action_flat)
    #     act_cond = act_cond.reshape(B, L, self.config.embed_dim)
        
    #     # Create interleaved tokens (obs at even positions, placeholder at odd)
    #     tokens = jnp.zeros((B, L * 2, self.state_dim))
    #     tokens = tokens.at[:, 0::2].set(state)
        
    #     # Forward pass
    #     output = self.forward(tokens, perattn_out=act_cond)
        
    #     # Extract predictions
    #     pred_rewards = output.pred_rewards  # (B, L, reward_dim)
    #     logits_ends = output.logits_ends  # (B, L, 1 or 2)
        
    #     # Compute continuation probability (pcont)
    #     if self.use_ce_for_end:
    #         # logits_ends is (B, L, 2), apply softmax and take prob of class 0 (continue)
    #         pcont = jax.nn.softmax(logits_ends, axis=-1)[..., 0:1]  # (B, L, 1)
    #     else:
    #         # logits_ends is (B, L, 1), apply sigmoid for continuation prob
    #         pcont = jax.nn.sigmoid(logits_ends)  # (B, L, 1)
        
    #     # Handle reward output
    #     if self.use_ce_for_reward and self.reward_loss is not None:
    #         # Decode reward from logits using the loss function's mode
    #         pred_rewards = self.reward_loss.mode(pred_rewards)
    #         if self.use_symlog:
    #             pred_rewards = symexp(pred_rewards)
    #     elif self.use_symlog:
    #         pred_rewards = symexp(pred_rewards)
        
    #     # Squeeze time dimension if L=1, and squeeze the last dim to get (B,)
    #     if L == 1:
    #         pred_rewards = pred_rewards.squeeze(1)  # (B, reward_dim)
    #         pcont = pcont.squeeze(1)  # (B, 1)
    #         logits_ends = logits_ends.squeeze(1)  # (B, 1 or 2)
        
    #     # Squeeze reward to (B,) if it has trailing 1 dim
    #     if pred_rewards.ndim > 1 and pred_rewards.shape[-1] == 1:
    #         pred_rewards = pred_rewards.squeeze(-1)  # (B,)
    #     # Squeeze pcont to (B,) if it has trailing 1 dim
    #     if pcont.ndim > 1 and pcont.shape[-1] == 1:
    #         pcont = pcont.squeeze(-1)  # (B,)
        
    #     # Convert back to torch if input was torch
    #     if is_torch:
    #         pred_rewards = torch.from_numpy(np.asarray(pred_rewards)).to(input_device)
    #         pcont = torch.from_numpy(np.asarray(pcont)).to(input_device)
    #         logits_ends = torch.from_numpy(np.asarray(logits_ends)).to(input_device)
        
    #     return pred_rewards, pcont, logits_ends

    def compute_av_action(self, state, action, next_state = None):
        """Compute available action predictions."""
        # Convert torch tensors to jax arrays
        is_torch = isinstance(state, torch.Tensor)
        if is_torch:
            device = state.device
            state = jnp.array(state.detach().cpu().numpy())
            action = jnp.array(action.detach().cpu().numpy())
        
        if state.ndim == 2:
            state = state[:, None, :]
            action = action[:, None, :, :]
        
        B, L = state.shape[:2]
        
        # Reshape action for embedding
        action_flat = action.reshape(B * L, self.num_agents, -1)
        if self.is_discrete_action:
            action_flat = jnp.argmax(action_flat, axis=-1)
        act_cond = jax.vmap(self.act_emb)(action_flat)
        act_cond = act_cond.reshape(B, L, self.config.embed_dim)
        
        # Create tokens
        tokens = jnp.zeros((B, L * 2, self.state_dim))
        tokens = tokens.at[:, 0::2].set(state)
        
        output = self.forward(tokens, perattn_out=act_cond)
        
        if output.pred_avail_action is not None:
            # Return probability of available (class 1)
            result = jax.nn.softmax(output.pred_avail_action[:, -1], axis=-1)[..., 1]
            if is_torch:
                result = torch.from_numpy(np.asarray(result)).to(device)
            return result
        return None

    def compute_loss(self, batch: Dict[str, jnp.ndarray], attention_mask: jnp.ndarray = None, **kwargs):
        """Compute training loss. Matches PyTorch compute_loss() interface."""
        shared_obs = batch['shared_obs']  # (B, L, state_dim)
        action = batch['action']  # (B, L, num_agents, action_dim)
        reward = batch['reward']  # (B, L, num_agents, 1) or (B, L, 1)
        done = batch['done']  # (B, L, num_agents, 1)
        
        B, L = shared_obs.shape[:2]
        
        # Get action embeddings for each timestep
        action_flat = action.reshape(B * L, self.num_agents, -1)
        if self.is_discrete_action:
            action_flat = jnp.argmax(action_flat, axis=-1)
        act_cond = jax.vmap(self.act_emb)(action_flat)
        act_cond = act_cond.reshape(B, L, self.config.embed_dim)
        
        # Create interleaved tokens (obs, placeholder, obs, placeholder, ...)
        tokens = jnp.zeros((B, L * 2, self.state_dim))
        tokens = tokens.at[:, 0::2].set(shared_obs)
        
        # Forward pass
        output = self.forward(tokens, perattn_out=act_cond, attention_mask=attention_mask)
        
        ### Compute discount/end loss
        if not self.use_ce_for_end:
            # Using log probability (Dreamer V3 style)
            target_con = 1.0 - done.all(axis=2).astype(jnp.float32)  # (B, L, 1)
            loss_ends = jnp.mean(optax.sigmoid_binary_cross_entropy(output.logits_ends, target_con))
        else:
            # Using cross-entropy
            labels_ends = done.all(axis=2).astype(jnp.int32).reshape(-1)  # (B*L,)
            logits_ends = output.logits_ends.reshape(-1, 2)
            loss_ends = -jnp.mean(jnp.sum(
                jax.nn.log_softmax(logits_ends) * jax.nn.one_hot(labels_ends, 2), axis=-1
            ))
        
        ### Compute reward loss
        labels_rewards = reward.mean(axis=2) if reward.ndim == 4 else reward  # (B, L, 1)
        if self.use_symlog:
            labels_rewards = jnp.sign(labels_rewards) * jnp.log(jnp.abs(labels_rewards) + 1)
        
        if self.use_ce_for_reward and self.reward_loss is not None:
            labels_flat = labels_rewards.reshape(-1)
            logits_flat = output.pred_rewards.reshape(-1, output.pred_rewards.shape[-1])
            loss_rewards = self.reward_loss(logits_flat, labels_flat)
        else:
            loss_rewards = jnp.abs(output.pred_rewards - labels_rewards).mean()
        
        ### Compute available action loss
        if self.enable_av_pred and 'av_action' in batch and batch['av_action'] is not None:
            # Roll done flags to mask first timestep after done
            tmp = jnp.roll(done.all(axis=2), 1, axis=1).squeeze(-1)  # (B, L)
            labels_av_actions = batch['av_action']  # (B, L, N, action_dim)
            
            # Set labels to 1 (available) for timesteps after done
            labels_av_actions = jnp.where(
                tmp[:, :, None, None],
                jnp.ones_like(labels_av_actions),
                labels_av_actions
            )
            
            # Match PyTorch: use [:-1] for logits and [1:] for labels
            logits_flat = rearrange(output.pred_avail_action[:, :-1], 'b l n a e -> (b l n a) e')
            labels_flat = rearrange(labels_av_actions[:, 1:], 'b l n a -> (b l n a)').astype(jnp.int32)
            loss_av_actions = -jnp.mean(jnp.sum(
                jax.nn.log_softmax(logits_flat) * jax.nn.one_hot(labels_flat, 2), axis=-1
            ))
        else:
            loss_av_actions = jnp.array(0.0)
        
        total_loss = loss_ends + loss_rewards + loss_av_actions
        
        loss_dict = {
            'loss_rew': loss_rewards,
            'loss_end': loss_ends,
            'loss_av_action': loss_av_actions if self.enable_av_pred else jnp.array(0.0),
            'info_loss': jnp.array(0.0),
            'loss_total': total_loss,
        }
        
        return total_loss, loss_dict

    def train(self, mode=True):
        """Compatibility method - no-op for JAX."""
        return self
    
    def eval(self):
        """Compatibility method - no-op for JAX."""
        return self
    
    def parameters(self, recurse=True):
        """Return parameters as list for PyTorch compatibility."""
        import torch
        params = eqx.filter(self, eqx.is_array)
        flat_params = jax.tree_util.tree_leaves(params)
        torch_params = [torch.nn.Parameter(torch.from_numpy(np.asarray(p))) for p in flat_params]
        return torch_params

    def named_parameters(self, prefix='', recurse=True):
        """Return named parameters for PyTorch compatibility."""
        import torch
        params = eqx.filter(self, eqx.is_array)
        flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params)
        for path, val in flat_params:
            name = '.'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            if prefix:
                name = f"{prefix}.{name}"
            yield name, torch.nn.Parameter(torch.from_numpy(np.asarray(val)))

    def named_modules(self, prefix=''):
        """Compatibility method - yields (name, module) pairs."""
        yield prefix, self
    
    def to(self, device):
        """Compatibility method - returns self."""
        return self
    
    def state_dict(self):
        """Return state dict for PyTorch compatibility."""
        import torch
        params = eqx.filter(self, eqx.is_array)
        flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params)
        state_dict = {}
        for path, val in flat_params:
            name = '.'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            state_dict[name] = torch.from_numpy(np.asarray(val))
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict - placeholder for JAX."""
        pass
@eqx.filter_jit
def get_act_emb(act_emb: SimpleActionEncoder ,is_discrete_action: Bool, action: jnp.ndarray) -> jnp.ndarray:
    """Get action embeddings matching PyTorch version."""
    if is_discrete_action and action.ndim == 3:
        action = jnp.argmax(action, axis=-1)
    return jax.vmap(act_emb)(action)

# Helper function for symlog transform
def symlog(x):
    """Symmetric log transform."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x):
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def create_rew_end_model(
    config: Dict[str, Any],
    obs_dim: int,
    action_dim: int,
    *,
    key: jax.random.PRNGKey
) -> eqx.Module:
    """Factory function to create reward/end models from config."""
    model_type = config.get("model_type", "transformer")
    
    if model_type == "rnn":
        return RewEndModel(
            cfg=config,
            num_agents=config.get("num_agents", 2),
            state_dim=obs_dim,
            action_dim=action_dim,
            is_continuous_act=config.get("is_continuous_act", True),
            key=key
        )
    else:
        return TransRewEndModel(
            state_dim=obs_dim,
            act_vocab_size=action_dim,
            num_agents=config.get("num_agents", 2),
            config=config,
            action_dim=action_dim,
            is_discrete_action=not config.get("is_continuous_act", True),
            key=key
        )
