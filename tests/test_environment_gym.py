import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import environments  # noqa: F401  (registers the environments)

_ENV_ID = ["TaylorGreen-v0", "StratifiedShear-v0"]


def test_environment_is_registered():
    for env_id in _ENV_ID:
        assert env_id in gym.registry
