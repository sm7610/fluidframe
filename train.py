# Copyright 2025 Shruti Mishra. All rights reserved
import numpy as np
from tqdm import tqdm
from typing import Optional
from agent_qlearning import QLearningAgent
from environment import Environment


_Q = 1000 * np.ones((12, 4))
_EPSILON = 0.01
_SAVE_FOLDER = "./checkpoints/"


def save_checkpoint(policy: np.ndarray, episode: int) -> None:
    print(f"The policy is {policy}.")
    filename = _SAVE_FOLDER + "policy_" + str(episode) + ".npy"
    np.save(filename, policy)


def train(
    env: Environment,
    n_episodes: int,
    n_steps: int,
    save: bool = False,
    logging: bool = True,
    seed: Optional[int] = None,
) -> None:
    agent = QLearningAgent(q=_Q, seed=seed)  # initialise agent
    episode_returns = []

    for episode in tqdm(range(n_episodes)):
        obs = env.reset()
        episode_return = 0
        for step in range(n_steps):
            action = agent.get_action(
                obs, epsilon=_EPSILON * (1 - episode / n_episodes)
            )
            next_obs, reward = env.step(action)
            agent.update_q(obs, action, reward, next_obs)  # update based on experience
            obs = next_obs
            episode_return += reward

        if save:
            episode_returns.append(episode_return)
            if episode % 1000 == 0:
                save_checkpoint(policy=np.argmax(agent.q, axis=1), episode=episode)
            elif episode == n_episodes - 1:
                save_checkpoint(policy=np.argmax(agent.q, axis=1), episode=episode)
                filename = _SAVE_FOLDER + "episode_returns" + ".npy"
                np.save(filename, episode_returns)

        if logging:
            if episode % 100 == 0:
                print(f"Episode {episode} return: \t {episode_return}")
                print(f"Policy: \t {np.argmax(agent.q, axis=1)}.")
            elif episode == n_episodes - 1:
                save_checkpoint(policy=np.argmax(agent.q, axis=1), episode=episode)
                print(f"Last episode return: \t {episode_return}")
                print(f"Policy: \t {np.argmax(agent.q, axis=1)}.")
