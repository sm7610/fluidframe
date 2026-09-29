import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import environments  # noqa: F401  (registers the environments)
from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment
from environments.taylor_green_gym import TaylorGreenContinuousGymEnv

_ENV_ID = "TaylorGreen-v0"


def test_environment_is_registered():
    assert _ENV_ID in gym.registry
