
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
from agent.world_models.diffusion.inner_modeljax import InnerModel as InnerModelJax
from agent.world_models.diffusion.inner_model import InnerModel
import pickle
import jax.numpy as jnp
import jax
import timeit
import equinox as eqx

from agent.world_models.rew_end_model import RewEndModel, TransRewEndModel
# JAX versions of rew_end_model (for future use)
from agent.world_models.rew_end_modeljax_eqx import RewEndModel as RewEndModelJax, TransRewEndModel as TransRewEndModelJax
config,enable_av_pred = pickle.load(open('rewendmodel_config.pkl', 'rb'))

key= jax.random.PRNGKey(0)
key, subkey = jax.random.split(key,2)
rew_end_model_jax = TransRewEndModelJax(
                state_dim=config.STATE_DIM,
                act_vocab_size=config.ACTION_SIZE,
                num_agents=config.NUM_AGENTS,
                config=config.trans_config,
                action_dim=config.ACTION_SIZE,
                is_discrete_action=not config.CONTINUOUS_ACTION,
                use_ce_for_end=config.use_ce_for_cont,
                use_ce_for_av_action=True,
                enable_av_pred=enable_av_pred,
                key=subkey
            )
rew_end_model = TransRewEndModel(
                state_dim=config.STATE_DIM,
                act_vocab_size=config.ACTION_SIZE,
                num_agents=config.NUM_AGENTS,
                config=config.trans_config,
                action_dim=config.ACTION_SIZE,
                is_discrete_action=not config.CONTINUOUS_ACTION,
                use_ce_for_end=config.use_ce_for_cont,
                use_ce_for_av_action=True,
                enable_av_pred=enable_av_pred,
            )


def count_parameters(model):
    """Count the total number of parameters in an Equinox model."""
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


pt_params = sum(p.numel() for p in rew_end_model.parameters())
jax_params = count_parameters(rew_end_model_jax)

print(f"inner_model (PyTorch) parameter count: {pt_params}")
print(f"inner_model_jax (Equinox/JAX) parameter count: {jax_params}")
