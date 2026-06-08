from __future__ import absolute_import, division, print_function

import os
import time
from os import replace

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from absl import logging

logging.set_verbosity(logging.DEBUG)
import os.path as osp
from pathlib import Path
import yaml

from gym.spaces import Box, Discrete


class SMACv2Env:
    def __init__(self, args):
        self.map_config = self.load_map_config(args["map_name"])
        self.seed(args["seed"])

    def step(self, actions):
        # processed_actions = np.squeeze(actions, axis=1).tolist()
        reward, terminated, info = self.env.step(actions)
        obs = self.env.get_obs()
        state = self.repeat(self.env.get_state())
        rewards = [[reward]] * self.n_agents
        dones = [terminated] * self.n_agents
        if terminated:
            if self.env.env.timeouts > self.timeouts:
                assert (
                    self.env.env.timeouts - self.timeouts == 1
                ), "Change of timeouts unexpected."
                info["bad_transition"] = True
                self.timeouts = self.env.env.timeouts
        # infos = [info] * self.n_agents
        avail_actions = self.env.get_avail_actions()
        return self.wrap(obs), self.wrap(state), self.wrap(rewards), \
            self.wrap(dones), info, self.wrap(avail_actions)

    def reset(self):
        self.env.reset()
        obs = self.env.get_obs()
        state = self.repeat(self.env.get_state())
        avail_actions = self.env.get_avail_actions()
        return self.wrap(obs), self.wrap(state), self.wrap(avail_actions)
    
    def wrap(self, l):
        d = {}
        for i in range(self.n_agents):
            d[i] = l[i]
        return d

    def seed(self, seed):
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

        self.env = StarCraftCapabilityEnvWrapper(seed=seed, **self.map_config)
        env_info = self.env.get_env_info()
        n_actions = env_info["n_actions"]
        state_shape = env_info["state_shape"]
        obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.timeouts = self.env.env.timeouts
        self.discrete = True

        self.n_obs = env_info["obs_shape"]
        self.n_actions = env_info["n_actions"]
        self.state_dim = state_shape

        self.share_observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(state_shape,))
        )
        self.observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        )
        self.action_space = self.repeat(Discrete(n_actions))

    def close(self):
        self.env.close()

    def load_map_config(self, map_name):
        repo_root = Path(__file__).resolve().parents[2]
        workspace_root = repo_root.parent
        candidate_dirs = [
            os.getenv("SMACV2_MAP_CONFIG_DIR"),
            repo_root / "configs" / "envs_cfgs" / "smacv2_map_config",
            workspace_root / "HARL-main" / "harl" / "configs" / "envs_cfgs" / "smacv2_map_config",
            Path.home() / "3rdApps" / "smacv2_map_config",
        ]
        map_config_path = None
        for candidate_dir in candidate_dirs:
            if candidate_dir is None:
                continue
            candidate_path = Path(candidate_dir) / f"{map_name}.yaml"
            if candidate_path.exists():
                map_config_path = candidate_path
                break
        if map_config_path is None:
            searched = ", ".join(str(path) for path in candidate_dirs if path is not None)
            raise FileNotFoundError(
                f"SMACv2 map config '{map_name}.yaml' was not found. Searched: {searched}"
            )
        with open(str(map_config_path), "r", encoding="utf-8") as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)
        return map_config

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)
    
    def get_avail_actions(self):
        return self.env.get_avail_actions()
