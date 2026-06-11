import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib
import native_runtime
native_runtime.prepare_train_process()
import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from environments import Env

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::FutureWarning"
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")

DreamerRunner = None
Experiment = None
EnvCurriculumConfig = None
StarCraftConfig = None
SMACv2Config = None
PettingZooConfig = None
FootballConfig = None
MAMujocoConfig = None
BidexHandsConfig = None
DreamerControllerConfig = None
DreamerLearnerConfig = None
Smacv2DreamerLearnerConfig = None
Smacv2DreamerControllerConfig = None
MPEDreamerLearnerConfig = None
MPEDreamerControllerConfig = None
GRFDreamerLearnerConfig = None
GRFDreamerControllerConfig = None
MAMujocoDreamerLearnerConfig = None
MAMujocoDreamerControllerConfig = None
LOGGER = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="flatland", help="Flatland or SMAC env")
    parser.add_argument("--env_name", type=str, default="5_agents", help="Specific setting")
    parser.add_argument("--policy_class", type=str, default="beta", help="gaussian or beta")

    parser.add_argument("--agent_conf", type=str, default=None)
    parser.add_argument("--enable_mpe_disc", action="store_true")

    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--seed", type=int, default=1, help="Random seed id")
    parser.add_argument("--steps", type=int, default=1e6, help="Max environment steps")
    parser.add_argument("--mode", type=str, default="disabled")
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--sample_temp", type=float, default="inf")
    parser.add_argument("--ce_for_cont", action="store_true")
    parser.add_argument("--state_decoder_type", type=int, default=1)

    parser.add_argument("--load_pretrained", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)

    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument(
        "--rl_device",
        type=str,
        default="cuda:0",
        help="RL device for bidexhands, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--sim_device",
        type=str,
        default="cuda:0",
        help="IsaacGym sim device for bidexhands, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="gpu",
        choices=("gpu", "cpu", "cuda"),
        help="IsaacGym pipeline mode for bidexhands",
    )

    return parser.parse_args()


def _seed_everywhere(seed: int, torch_module) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def _preimport_bidexhands_isaacgym() -> None:
    from env.bidexhands.bidexhands_env import BiDexHandsEnv

    BiDexHandsEnv._patch_numpy_deprecated_aliases()
    BiDexHandsEnv._import_bidexhands_with_isaacgym_guard()


def _lazy_import_training_modules() -> None:
    global DreamerRunner, Experiment, EnvCurriculumConfig
    global StarCraftConfig, SMACv2Config
    global PettingZooConfig, FootballConfig, MAMujocoConfig, BidexHandsConfig
    global DreamerControllerConfig, DreamerLearnerConfig
    global Smacv2DreamerLearnerConfig, Smacv2DreamerControllerConfig
    global MPEDreamerLearnerConfig, MPEDreamerControllerConfig
    global GRFDreamerLearnerConfig, GRFDreamerControllerConfig
    global MAMujocoDreamerLearnerConfig, MAMujocoDreamerControllerConfig
    global LOGGER

    if DreamerRunner is not None:
        return

    from agent.runners.DreamerRunner import DreamerRunner as _DreamerRunner
    from configs import Experiment as _Experiment
    from configs.EnvConfigs import (
        BidexHandsConfig as _BidexHandsConfig,
        EnvCurriculumConfig as _EnvCurriculumConfig,
        FootballConfig as _FootballConfig,
        MAMujocoConfig as _MAMujocoConfig,
        PettingZooConfig as _PettingZooConfig,
        SMACv2Config as _SMACv2Config,
        StarCraftConfig as _StarCraftConfig,
    )
    from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig as _DreamerControllerConfig
    from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig as _DreamerLearnerConfig
    from configs.dreamer.football.GRFControllerConfig import (
        GRFDreamerControllerConfig as _GRFDreamerControllerConfig,
    )
    from configs.dreamer.football.GRFLearnerConfig import GRFDreamerLearnerConfig as _GRFDreamerLearnerConfig
    from configs.dreamer.mamujoco.mamujocoControllerConfig import (
        MAMujocoDreamerControllerConfig as _MAMujocoDreamerControllerConfig,
    )
    from configs.dreamer.mamujoco.mamujocoLearnerConfig import (
        MAMujocoDreamerLearnerConfig as _MAMujocoDreamerLearnerConfig,
    )
    from configs.dreamer.mpe.MpeControllerConfig import MPEDreamerControllerConfig as _MPEDreamerControllerConfig
    from configs.dreamer.mpe.MpeLearnerConfig import MPEDreamerLearnerConfig as _MPEDreamerLearnerConfig
    from configs.dreamer.smacv2.smacv2ControllerConfig import (
        Smacv2DreamerControllerConfig as _Smacv2DreamerControllerConfig,
    )
    from configs.dreamer.smacv2.smacv2LearnerConfig import (
        Smacv2DreamerLearnerConfig as _Smacv2DreamerLearnerConfig,
    )
    from tb_logger import LOGGER as _LOGGER

    DreamerRunner = _DreamerRunner
    Experiment = _Experiment
    EnvCurriculumConfig = _EnvCurriculumConfig
    StarCraftConfig = _StarCraftConfig
    SMACv2Config = _SMACv2Config
    PettingZooConfig = _PettingZooConfig
    FootballConfig = _FootballConfig
    MAMujocoConfig = _MAMujocoConfig
    BidexHandsConfig = _BidexHandsConfig
    DreamerControllerConfig = _DreamerControllerConfig
    DreamerLearnerConfig = _DreamerLearnerConfig
    Smacv2DreamerLearnerConfig = _Smacv2DreamerLearnerConfig
    Smacv2DreamerControllerConfig = _Smacv2DreamerControllerConfig
    MPEDreamerLearnerConfig = _MPEDreamerLearnerConfig
    MPEDreamerControllerConfig = _MPEDreamerControllerConfig
    GRFDreamerLearnerConfig = _GRFDreamerLearnerConfig
    GRFDreamerControllerConfig = _GRFDreamerControllerConfig
    MAMujocoDreamerLearnerConfig = _MAMujocoDreamerLearnerConfig
    MAMujocoDreamerControllerConfig = _MAMujocoDreamerControllerConfig
    LOGGER = _LOGGER


def train_dreamer(exp, n_workers):
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes, save_interval=200000, save_mode="interval")


def get_env_info(configs, env):
    if not env.discrete:
        assert hasattr(env, "individual_action_space")
        individual_action_space = env.individual_action_space
    else:
        individual_action_space = None

    for config in configs:
        config.IN_DIM = env.n_obs
        config.STATE_DIM = env.state_dim
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = individual_action_space

    print(f"Observation dims: {env.n_obs}")
    print(f"Global State dims: {env.state_dim}")
    print(f"Action dims: {env.n_actions}")
    print(f"Num agents: {env.n_agents}")
    print(f"Continuous action for control? -> {not env.discrete}")
    if hasattr(env, "individual_action_space"):
        print(f"Individual action space: {env.individual_action_space}")

    env.close()


def prepare_pettingzoo_configs(env_name, continuous_action=True):
    agent_configs = [MPEDreamerControllerConfig(), MPEDreamerLearnerConfig()]
    env_config = PettingZooConfig(env_name, RANDOM_SEED, continuous_action)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 2000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_smacv2_configs(env_name):
    agent_configs = [Smacv2DreamerControllerConfig(), Smacv2DreamerLearnerConfig()]
    env_config = SMACv2Config(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 2000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_football_configs(env_name):
    agent_configs = [GRFDreamerControllerConfig(), GRFDreamerLearnerConfig()]
    env_config = FootballConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_mamujoco_configs(scenario, agent_config):
    agent_configs = [MAMujocoDreamerControllerConfig(), MAMujocoDreamerLearnerConfig()]
    env_config = MAMujocoConfig(scenario=scenario, seed=RANDOM_SEED, agent_conf=agent_config)

    agent_configs[1].env_name = scenario

    get_env_info(agent_configs, env_config.create_env())
    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


def prepare_bidexhands_configs(task_name, rl_device="cuda:0", sim_device="cuda:0", pipeline="gpu"):
    from gym.spaces import Box

    agent_configs = [MAMujocoDreamerControllerConfig(), MAMujocoDreamerLearnerConfig()]
    env_config = BidexHandsConfig(
        task_name=task_name,
        seed=RANDOM_SEED,
        rl_device=rl_device,
        sim_device=sim_device,
        pipeline=pipeline,
    )

    agent_configs[1].env_name = task_name
    agent_configs[1].DEVICE = rl_device

    if task_name == "ShadowHandBottleCap":
        in_dim, state_dim, action_size = 221, 420, 26
    elif task_name in ("ShadowHandDoorOpenInward", "ShadowHandDoorOpenOutward", "ShadowHandPen"):
        in_dim, state_dim, action_size = 218, 417, 26
    else:
        raise ValueError(
            f"Unsupported bidexhands task '{task_name}'. "
            "Supported tasks: ShadowHandBottleCap, ShadowHandDoorOpenInward, "
            "ShadowHandDoorOpenOutward, ShadowHandPen"
        )

    action_space = Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)
    for config in agent_configs:
        config.IN_DIM = in_dim
        config.STATE_DIM = state_dim
        config.ACTION_SIZE = action_size
        config.NUM_AGENTS = 2
        config.CONTINUOUS_ACTION = True
        config.ACTION_SPACE = action_space

    return {
        "env_config": (env_config, 5000),
        "controller_config": agent_configs[0],
        "learner_config": agent_configs[1],
        "reward_config": None,
        "obs_builder_config": None,
    }


if __name__ == "__main__":
    RANDOM_SEED = 23
    args = parse_args()

    if args.env == "dexhands":
        args.env = Env.BIDEXHANDS.value

    if args.env == Env.BIDEXHANDS.value:
        _preimport_bidexhands_isaacgym()

    _lazy_import_training_modules()

    import torch

    RANDOM_SEED += args.seed * 100

    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.SMACv2:
        configs = prepare_smacv2_configs(args.env_name)
    elif args.env == Env.PETTINGZOO:
        configs = prepare_pettingzoo_configs(args.env_name, continuous_action=not args.enable_mpe_disc)
    elif args.env == Env.GRF:
        configs = prepare_football_configs(args.env_name)
    elif args.env == Env.MAMUJOCO:
        configs = prepare_mamujoco_configs(args.env_name, args.agent_conf)
    elif args.env == Env.BIDEXHANDS:
        configs = prepare_bidexhands_configs(
            args.env_name,
            rl_device=args.rl_device,
            sim_device=args.sim_device,
            pipeline=args.pipeline,
        )
    else:
        raise Exception("Unknown environment")

    _seed_everywhere(RANDOM_SEED, torch)
    torch.autograd.set_detect_anomaly(True)

    assert args.state_decoder_type in [1, 2]

    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)
    configs["learner_config"].seed = RANDOM_SEED

    configs["learner_config"].policy_class = args.policy_class
    configs["controller_config"].policy_class = args.policy_class

    if args.policy_class == "gaussian":
        configs["learner_config"].ENTROPY = 0.001
    elif args.policy_class == "beta":
        configs["learner_config"].ENTROPY = 0.01

    configs["learner_config"].use_ce_for_cont = args.ce_for_cont
    configs["learner_config"].compute_end_in_TD = args.ce_for_cont
    configs["learner_config"].diffusion_sampler_cfg.num_steps_denoising = (
        configs["learner_config"].NUM_AGENTS
        if configs["learner_config"].NUM_AGENTS > 2
        else configs["learner_config"].NUM_AGENTS * 2
    )

    configs["learner_config"].load_pretrained = args.load_pretrained
    configs["learner_config"].load_path = args.load_path

    if args.sample_temp == float("inf"):
        configs["learner_config"].sample_temperature = str(args.sample_temp)
    else:
        configs["learner_config"].sample_temperature = args.sample_temp

    configs["learner_config"].state_decoder_type = "s + id" if args.state_decoder_type == 1 else "s + last_obs"
    configs["controller_config"].state_decoder_type = "s + id" if args.state_decoder_type == 1 else "s + last_obs"

    current_date = datetime.now()
    current_date_string = current_date.strftime("%m%d")

    dir_prefix = args.env_name + "-" + args.agent_conf if args.agent_conf is not None else args.env_name
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/{current_date_string}_results") / args.env / dir_prefix

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"

    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        os.makedirs(str(run_dir / "ckpt"))

    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "agent"), dst=run_dir / "agent")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "configs"), dst=run_dir / "configs")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "networks"), dst=run_dir / "networks")
    shutil.copyfile(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "train.py"), dst=run_dir / "train.py")

    print(f"Run files are saved at {str(run_dir)}\n")

    configs["learner_config"].RUN_DIR = str(run_dir)
    configs["learner_config"].map_name = args.env_name

    if args.env == Env.MAMUJOCO:
        group_name = f"raw_trans_branch_{args.env_name}_{args.agent_conf}_H{configs['learner_config'].horizon}"
    else:
        group_name = f"raw_trans_branch_{args.env_name}_H{configs['learner_config'].horizon}"

    if args.ce_for_cont:
        group_name += "_ce_for_cont"

    if args.env in (Env.PETTINGZOO, Env.SMACv2):
        run_name = f"DIMA_s{args.seed}_{args.env_name}"
    elif args.env in (Env.MAMUJOCO, Env.BIDEXHANDS):
        run_name = f"DIMA_s{args.seed}_{args.env_name}"
        if args.agent_conf is not None:
            run_name += f"_{args.agent_conf}"
    else:
        run_name = (
            f"({current_date_string}) raw_H{configs['learner_config'].horizon}"
            f"_t{args.temperature}_s{RANDOM_SEED}_i{configs['learner_config'].N_SAMPLES}"
            f"_{args.policy_class}_gamma{configs['learner_config'].GAMMA}"
        )

    job_type = "DIMA"

    import wandb

    if args.env == Env.MAMUJOCO:
        project_name = "mamujoco"
    elif args.env == Env.PETTINGZOO:
        project_name = "MPE"
    elif args.env == Env.SMACv2:
        project_name = "SMACv2"
    elif args.env == Env.BIDEXHANDS:
        project_name = "dexhands"
    else:
        project_name = "SMAD"

    wandb.init(
        project=project_name,
        mode=args.mode if not args.load_pretrained else "disabled",
        group=group_name,
        job_type=job_type,
        name=run_name,
        config=configs["learner_config"].to_dict(),
        notes="",
    )

    print("group name: ", group_name)
    print("run name: ", run_name)
    print("job type: ", job_type)

    exp_dir = (
        "tb_logs/"
        + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{group_name}"
        + f"_s{RANDOM_SEED}_i{configs['learner_config'].N_SAMPLES}"
        + f"_{args.policy_class}_gamma{configs['learner_config'].GAMMA}"
    )
    if args.use_tensorboard:
        LOGGER.initialize(log_dir=exp_dir)

    exp = Experiment(
        steps=args.steps,
        episodes=500000,
        random_seed=RANDOM_SEED,
        env_config=EnvCurriculumConfig(
            *zip(configs["env_config"]),
            Env(args.env),
            obs_builder_config=configs["obs_builder_config"],
            reward_config=configs["reward_config"],
        ),
        controller_config=configs["controller_config"],
        learner_config=configs["learner_config"],
    )

    train_dreamer(exp, n_workers=args.n_workers)
