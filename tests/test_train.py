import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import train
from environment_taylor_green import TaylorGreenEnvironment


def test_train(n_episodes=2, n_steps=5, save=False):
    """Test that the training loop runs without errors."""
    train(
        env=TaylorGreenEnvironment(),
        n_episodes=n_episodes,
        n_steps=n_steps,
        save=save,
    )
