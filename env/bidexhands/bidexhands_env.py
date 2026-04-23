from __future__ import annotations

import ctypes
import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from gym.spaces import Box


def _to_numpy(value: Any) -> np.ndarray:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and isinstance(value, torch_mod.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@contextmanager
def _pushd(path: Path):
    cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


class BiDexHandsEnv:
    def __init__(
        self,
        task_name: str,
        seed: int,
        backend: str = "isaacgym",
        num_envs: int = 1,
        episode_limit: int = 200,
        headless: bool = True,
        rl_device: str = "cuda:0",
        sim_device: str = "cuda:0",
        pipeline: str = "gpu",
    ) -> None:
        if num_envs != 1:
            raise ValueError(
                "BiDexHandsEnv for DIMA expects num_envs=1 in the current runner."
            )
        if backend != "isaacgym":
            raise ValueError(
                "bidexhands uses IsaacGym backend; only backend='isaacgym' is supported."
            )
        if not headless:
            raise ValueError("Current DIMA bidexhands adapter supports headless=True only.")

        self.task_name = task_name
        self.seed_value = int(seed)
        self.backend = backend
        self.num_envs = int(num_envs)
        self.rl_device = rl_device
        self.sim_device = sim_device
        self.pipeline = pipeline
        self.episode_limit = int(episode_limit)

        self._repo_root = Path(__file__).resolve().parents[3] / "DexterousHands"
        self._pkg_root = self._repo_root / "bidexhands"
        if not self._pkg_root.exists():
            raise ImportError(
                f"bidexhands source directory not found at {self._pkg_root}. "
                "Expected DexterousHands/bidexhands in the project parent."
            )
        if str(self._repo_root) not in sys.path:
            sys.path.insert(0, str(self._repo_root))

        self.env = self._build_env()
        self._init_spaces()

    @staticmethod
    def _patch_numpy_deprecated_aliases() -> None:
        if "float" not in np.__dict__:
            np.float = float  # type: ignore[attr-defined]
        if "int" not in np.__dict__:
            np.int = int  # type: ignore[attr-defined]
        if "bool" not in np.__dict__:
            np.bool = np.bool_  # type: ignore[attr-defined]

    @staticmethod
    def _preload_libpython_for_isaacgym() -> None:
        major = sys.version_info.major
        minor = sys.version_info.minor
        candidates = [
            Path(sys.prefix) / "lib" / f"libpython{major}.{minor}.so.1.0",
            Path(sys.prefix) / "lib" / f"libpython{major}.{minor}.so",
            Path(sys.exec_prefix) / "lib" / f"libpython{major}.{minor}.so.1.0",
            Path(sys.exec_prefix) / "lib" / f"libpython{major}.{minor}.so",
        ]
        for lib_path in candidates:
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            break

    @staticmethod
    def _import_bidexhands_with_isaacgym_guard():
        BiDexHandsEnv._preload_libpython_for_isaacgym()
        return importlib.import_module("bidexhands")

    def _build_env(self):
        self._patch_numpy_deprecated_aliases()
        bidexhands = self._import_bidexhands_with_isaacgym_guard()
        config_mod = importlib.import_module("bidexhands.utils.config")
        package_utils_mod = importlib.import_module("bidexhands.utils.package_utils")

        original_get_args = config_mod.get_args
        original_package_get_args = package_utils_mod.get_args

        def patched_get_args(benchmark=False, use_rlg_config=False, task_name="", algo=""):
            original_argv = sys.argv
            sys.argv = [original_argv[0]]
            try:
                args = original_get_args(
                    benchmark=benchmark,
                    use_rlg_config=use_rlg_config,
                    task_name=task_name,
                    algo=algo,
                )
            finally:
                sys.argv = original_argv

            args.num_envs = self.num_envs
            args.seed = self.seed_value
            args.episode_length = self.episode_limit
            args.headless = True
            args.rl_device = self.rl_device
            args.sim_device = self.sim_device
            args.pipeline = self.pipeline

            if ":" in args.sim_device:
                sim_device_type, compute_device_id = args.sim_device.split(":", 1)
                args.sim_device_type = sim_device_type
                args.compute_device_id = int(compute_device_id)
            else:
                args.sim_device_type = args.sim_device
                args.compute_device_id = 0

            args.device_id = args.compute_device_id
            args.device = args.sim_device_type if args.pipeline in ("gpu", "cuda") else "cpu"
            args.use_gpu = args.sim_device_type == "cuda"
            args.use_gpu_pipeline = args.pipeline in ("gpu", "cuda")
            args.graphics_device_id = args.compute_device_id if args.use_gpu_pipeline else -1
            return args

        config_mod.get_args = patched_get_args
        package_utils_mod.get_args = patched_get_args
        try:
            with _pushd(self._pkg_root):
                return bidexhands.make(self.task_name, "mappo")
        finally:
            config_mod.get_args = original_get_args
            package_utils_mod.get_args = original_package_get_args

    def _init_spaces(self) -> None:
        self.n_agents = int(self.env.num_agents)
        self.n = self.n_agents
        self.discrete = False

        self.observation_space = list(self.env.observation_space)
        self.share_observation_space = list(self.env.share_observation_space)
        self.true_action_space = tuple(self.env.action_space)
        self.n_obs = int(self.observation_space[0].shape[0])
        self.state_dim = int(self.share_observation_space[0].shape[0])
        self.max_time_steps = int(
            getattr(self.env.task, "max_episode_length", self.episode_limit)
        )

        self.n_actions = max(int(space.shape[0]) for space in self.true_action_space)
        self.action_space = tuple(
            Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)
            for _ in range(self.n_agents)
        )
        self.individual_action_space = self.action_space[0]

    def _wrap_agent_dict(self, values: np.ndarray) -> Dict[int, np.ndarray]:
        return {
            agent_id: np.asarray(values[agent_id], dtype=np.float32)
            for agent_id in range(self.n_agents)
        }

    def reset(self, **kwargs):
        del kwargs
        obs_all, state_all, _ = self.env.reset()
        obs = _to_numpy(obs_all)[0]
        state = _to_numpy(state_all)[0]
        return self._wrap_agent_dict(obs), self._wrap_agent_dict(state), None

    def step(self, actions):
        import torch

        action_array = _to_numpy(actions).astype(np.float32)
        if action_array.ndim > 2:
            action_array = np.squeeze(action_array, axis=0)

        per_agent_actions = []
        for agent_id in range(self.n_agents):
            true_dim = int(self.true_action_space[agent_id].shape[0])
            per_agent_actions.append(
                torch.as_tensor(
                    action_array[agent_id, :true_dim],
                    dtype=torch.float32,
                ).view(1, true_dim)
            )

        obs_all, state_all, reward_all, done_all, _, _ = self.env.step(per_agent_actions)

        obs = _to_numpy(obs_all)[0]
        state = _to_numpy(state_all)[0]
        reward = _to_numpy(reward_all)[0]
        done = _to_numpy(done_all)[0]

        info = {agent_id: {} for agent_id in range(self.n_agents)}
        reward_dict = {
            agent_id: [float(np.asarray(reward[agent_id]).reshape(-1)[0])]
            for agent_id in range(self.n_agents)
        }
        done_dict = {
            agent_id: bool(np.asarray(done[agent_id]).reshape(-1)[0])
            for agent_id in range(self.n_agents)
        }
        return (
            self._wrap_agent_dict(obs),
            self._wrap_agent_dict(state),
            reward_dict,
            done_dict,
            info,
            None,
        )

    def close(self) -> None:
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()

    def seed(self, seed: Optional[int]) -> None:
        self.seed_value = int(seed) if seed is not None else self.seed_value
