import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval import eval


def test_eval(n_episodes=2, n_steps=5, save=False):
    """Test that the evaluation loop runs without errors."""
    eval(
        n_episodes=n_episodes,
        n_steps=n_steps,
        policy=[1, 2, 1, 2, 1, 1, 1, 3, 1, 0, 1, 1],
        swimmer_speed=0.3,
        alignment_timescale=1.0,
    )
