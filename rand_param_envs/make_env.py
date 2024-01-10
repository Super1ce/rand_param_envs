
import os

import numpy as np
import torch
from gym.spaces.box import Box

from .wrappers import VariBadWrapper
from .hopper_rand_params import HopperRandParamsEnv
from .walker2d_rand_params import Walker2DRandParamsEnv


def make_env(env_id, episodes_per_task, seed=None, **kwargs):
    if env_id == 'walker_params':
        env = Walker2DRandParamsEnv(**kwargs)
    elif env_id == 'hopper_params':
        env = HopperRandParamsEnv(**kwargs)
    if seed is not None:
        env.seed(seed)
    env = VariBadWrapper(env=env,
                         episodes_per_task=episodes_per_task,
                         )
    return env


