import os

import gym
import metaworld.envs.mujoco.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit

import dmc2gym
from rlkit.envs.wrappers import NormalizedBoxEnv


def make_env(cfg, is_eval=False):
    if 'metaworld' in cfg.env:
        env = make_metaworld_env(cfg)
    elif 'robodesk' in cfg.env:
        env = make_robodesk_env(cfg)
    else:
        env = make_dmc_env(cfg)
    print(f"make_env (is_eval={is_eval}): {env.__class__.__name__}, {env.unwrapped.__class__.__name__}")
    return env


def make_dmc_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def make_gym_env(cfg, prefix=""):
    env_name = cfg.env[len(prefix):]
    env = gym.make(env_name)
    return env


def make_metaworld_env(cfg):
    env_name = cfg.env.replace('metaworld_', '')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()

    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(cfg.seed)

    env = TimeLimit(NormalizedBoxEnv(env), env.max_path_length)
    env.metadata["render_modes"] = ["rgb_array",]

    return env


def make_robodesk_env(cfg):

    import robodesk.robodesk.robodesk_state as robodesk_state

    env_name = cfg.env.replace('robodesk_', '')
    env = robodesk_state.RoboDesk(task=env_name, image_size=500)
    env = TimeLimit(NormalizedBoxEnv(env), 500)
    env.metadata["render_modes"] = ["rgb_array",]
    env.metadata["render.modes"] = ["rgb_array",]

    return env
