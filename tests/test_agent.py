import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_qlearning import QLearningAgent


def test_action():
    q = np.array([[0.2, 1.2], [1.2, -2.2]])
    agent = QLearningAgent(q)

    action = agent.get_action(observation=0, epsilon=0.0)
    assert action == 1

    action = agent.get_action(observation=1, epsilon=0.0)
    assert action == 0

    actions = []
    for i in range(10):
        action = agent.get_action(observation=1, epsilon=1.0)
        actions.append(action)
    assert len(set(actions)) == 2


def test_setup():
    q = np.array([[0.2, 1.2], [1.2, -2.2]])

    with pytest.raises(AssertionError):
        agent = QLearningAgent(q=q, gamma=-0.1)

    with pytest.raises(AssertionError):
        agent = QLearningAgent(q=q, gamma=1.5)

    agent = QLearningAgent(q=q, gamma=1.0)
