import numpy as np
from typing import Optional


class QLearningAgent:

    def __init__(
        self, q: np.ndarray, gamma: float = 0.999, seed: Optional[int] = None
    ) -> None:
        """
        Args:
            q: Q-table for estimated value
            gamma: discount factor
            seed: Optional int for specifying random number generation
        """
        self.q = q
        self.gamma = gamma
        _, self._n_actions = np.shape(self.q)
        assert self.gamma > 0
        assert self.gamma <= 1.0
        self.rng = np.random.default_rng(seed=seed)

    def update_q(
        self,
        observation: int,
        action_chosen: int,
        reward: float,
        next_observation: int,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Function to update the Q function
        Args:
            observation: current observation
            action_chosen: action chosen
            reward: reward for observation, action pair
            next_observation: the next observation
            learning_rate: learning rate for Q-function update rule
        """

        self.q[observation, action_chosen] = (1 - learning_rate) * self.q[
            observation, action_chosen
        ] + learning_rate * (reward + self.gamma * np.max(self.q[next_observation]))

    def get_action(self, observation: int, epsilon: float) -> int:
        """
        Gets the action based on the current policy and observation.
        Args:
            observation: observation, provided by environment
            epsilon: decides the fraction of times the action is chosen randomly
        Returns:
            action
        """
        q_obs = self.q[observation]
        rand = self.rng.uniform(0.0, 1.0)

        # is_greedy decides whether action is greedy or is chosen randomly
        is_greedy = 1  # choose greedy action

        if rand < epsilon:
            is_greedy = 0  # choose random action

        if is_greedy:
            # greedy algorithm
            action = np.argmax(q_obs)  # choose the action with the highest q value
        else:
            # compute action randomly from all the actions
            actions_rand = self.rng.permutation(
                self._n_actions
            )  # create vector of actions, shuffled
            action = actions_rand[0]  # take the first index

        return action
