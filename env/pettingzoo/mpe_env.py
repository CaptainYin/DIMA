import copy
from contextlib import contextmanager
import importlib
import logging
import os


def _headless_pygame_enabled():
    return os.environ.get("DIMA_PETTINGZOO_RENDER", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configure_headless_pygame():
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    if _headless_pygame_enabled():
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"


_configure_headless_pygame()

import numpy as np
import pygame.freetype
import supersuit as ss
from configs.mpe_scenarios import normalize_mpe_scenario

_configure_headless_pygame()


@contextmanager
def _skip_full_pygame_init_when_headless():
    if not _headless_pygame_enabled():
        yield
        return

    original_init = pygame.init

    def _headless_init():
        pygame.freetype.init()
        return (1, 0)

    pygame.init = _headless_init
    try:
        yield
    finally:
        pygame.init = original_init

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class PettingZooMPEEnv:
    def __init__(self, scenario, seed, continuous_actions, **args):
        _configure_headless_pygame()
        self.args = copy.deepcopy(args)
        self.scenario = normalize_mpe_scenario(scenario)
        # del self.args["scenario"]
        self.discrete = not continuous_actions
        self.args["continuous_actions"] = continuous_actions
        # if (
        #     "continuous_actions" in self.args
        #     and self.args["continuous_actions"] == True
        # ):
        #     self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26
        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)
        with _skip_full_pygame_init_when_headless():
            self.env = ss.pad_action_space_v0(
                ss.pad_observations_v0(self.module.parallel_env(**self.args))  # Initialize with self.args
            )
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        
        # compatiable with MARIE
        # import ipdb; ipdb.set_trace()
        self.n_actions = self.action_space[0].shape[0] if continuous_actions else self.action_space[0].n
        self.n_obs = self.observation_space[0].shape[0]
        self.individual_action_space = self.action_space[0]
        self.state_dim = self.share_observation_space[0].shape[0]

        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.discrete:
            env_actions = self.wrap_discrete_actions(actions)
        else:
            env_actions = self.wrap(actions)
        obs, rew, term, trunc, info = self.env.step(env_actions)
        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        s_obs = self.repeat(self.env.state())
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.wrap_with_id(self.unwrap(obs)),
            self.wrap_with_id(s_obs),
            self.wrap_with_id(rewards),
            self.wrap_with_id(self.unwrap(dones)),
            self.wrap_with_id(self.unwrap(info)),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(seed=self._seed))
        s_obs = self.repeat(self.env.state())
        return self.wrap_with_id(obs), self.wrap_with_id(s_obs), self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def wrap_discrete_actions(self, actions):
        action_ids = self._discrete_action_ids(actions)
        return {agent: int(action_ids[i]) for i, agent in enumerate(self.agents)}

    def _discrete_action_ids(self, actions):
        action_array = self._to_numpy(actions)

        while action_array.ndim > 2 and action_array.shape[0] == 1:
            action_array = action_array[0]

        if action_array.ndim == 2:
            if action_array.shape[0] != self.n_agents:
                raise ValueError(
                    "Discrete MPE actions must have one row per agent; "
                    f"got shape {action_array.shape} for {self.n_agents} agents."
                )
            if action_array.shape[1] == 1:
                action_ids = action_array.reshape(self.n_agents)
            else:
                action_ids = action_array.argmax(axis=-1)
        elif action_array.ndim == 1:
            if action_array.size == self.n_agents:
                action_ids = action_array
            elif action_array.size == self.n_agents * self.n_actions:
                action_ids = action_array.reshape(self.n_agents, self.n_actions).argmax(axis=-1)
            else:
                raise ValueError(
                    "Discrete MPE actions must be integer IDs or one-hot rows; "
                    f"got shape {action_array.shape} for {self.n_agents} agents and "
                    f"{self.n_actions} actions."
                )
        else:
            raise ValueError(
                "Discrete MPE actions must be integer IDs or one-hot rows; "
                f"got shape {action_array.shape}."
            )

        rounded = np.rint(action_ids)
        if not np.allclose(action_ids, rounded):
            raise ValueError(f"Discrete MPE action IDs must be integers; got {action_ids}.")

        action_ids = rounded.astype(np.int64)
        if np.any(action_ids < 0) or np.any(action_ids >= self.n_actions):
            raise ValueError(
                f"Discrete MPE action IDs must be in [0, {self.n_actions}); "
                f"got {action_ids}."
            )
        return action_ids

    def _to_numpy(self, actions):
        if hasattr(actions, "detach"):
            return actions.detach().cpu().numpy()
        if isinstance(actions, (list, tuple)):
            return np.asarray([self._to_numpy(action) for action in actions])
        return np.asarray(actions)
    
    def wrap_with_id(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[i] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
