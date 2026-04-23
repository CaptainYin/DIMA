
from agent.world_models.diffusion.inner_modeljax import InnerModel as InnerModelJax
from agent.world_models.diffusion.inner_model import InnerModel
import pickle
import jax.numpy as jnp
import jax
cfg,num_agents,is_continuous_act = pickle.load(open('denoiser_cfg.pkl', 'rb'))
# print(cfg)
# exit()
innermodel_input, c_noise, obs, act, act_mask = pickle.load(open('denoiser_inner_model_input.pkl', 'rb'))
key= jax.random.PRNGKey(0)
key, subkey = jax.random.split(key,2)
inner_model_jax = InnerModelJax(cfg.inner_model, cfg.perceiver, num_agents=num_agents, is_continuous_act=is_continuous_act,key=key)
inner_model = InnerModel(cfg.inner_model, cfg.perceiver, num_agents=num_agents, is_continuous_act=is_continuous_act)

innermodel_input_jax = jnp.array(innermodel_input.cpu().numpy())
c_noise_jax = jnp.array(c_noise.cpu().numpy())
obs_jax = jnp.array(obs.cpu().numpy())
act_jax = jnp.array(act.cpu().numpy())
act_mask_jax = jnp.array(act_mask.cpu().numpy())

# print(inner_model_jax(innermodel_input_jax,c_noise_jax,obs_jax,act_jax,act_mask_jax,subkey).shape)
# print(jnp.array(innermodel_input), jnp.array(c_noise), jnp.array(obs), jnp.array(act), jnp.array(act_mask))
# print(inner_model(innermodel_input, c_noise, obs, act, act_mask).shape)
import timeit
import equinox as eqx
def count_parameters(model):
    """Count the total number of parameters in an Equinox model."""
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
# JIT compile the JAX model
@eqx.filter_jit
def run_jax_jit(model, noisy_next_obs, c_noise, obs, act, act_mask, key):
    return model(noisy_next_obs, c_noise, obs, act, act_mask, key)

# Warmup JIT compilation
print("Warming up JIT...")
_ = run_jax_jit(inner_model_jax, innermodel_input_jax, c_noise_jax, obs_jax, act_jax, act_mask_jax, subkey)
_ = run_jax_jit(inner_model_jax, innermodel_input_jax, c_noise_jax, obs_jax, act_jax, act_mask_jax, subkey)
print("JIT warmup done")

pt_params = sum(p.numel() for p in inner_model.parameters())
jax_params = count_parameters(inner_model_jax)

print(f"inner_model (PyTorch) parameter count: {pt_params}")
print(f"inner_model_jax (Equinox/JAX) parameter count: {jax_params}")
            
def run():
    inner_model(innermodel_input, c_noise, obs, act, act_mask)

def run_jax():
    run_jax_jit(inner_model_jax, innermodel_input_jax, c_noise_jax, obs_jax, act_jax, act_mask_jax, subkey).block_until_ready()

print("PyTorch time (10 runs), JAX JIT time (10 runs):")
print(timeit.timeit(run, number=10), timeit.timeit(run_jax, number=10))