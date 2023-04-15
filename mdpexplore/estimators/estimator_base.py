import numpy as np
from abc import ABC, abstractmethod
from typing import List


class Estimator(ABC):
    def __init__(
        self,
        env,
        lambd: float = 1e-3,
        beta: float = 2,
    ) -> None:
        self.env = env
        self.lambd = lambd
        self.beta = beta
        self.estimated_mu = np.ones((self.env.get_states_num(), self.env.get_dim())) / self.env.get_states_num()

        self.observations = np.empty((0, 3), dtype=int)

    def estimate(self):
        if len(self.observations) > 0:
            return self._estimate()
        n_states = self.env.get_states_num()
        n_actions = self.env.get_actions_num()
        return np.zeros((n_states, n_actions, n_states)) / n_states

    def isin_conf(self, mu: np.ndarray) -> List[bool]:
        if len(self.observations) > 0:
            return self._isin_conf(mu)
        return [True]

    def estimate_transition_matrix(self, observations):
        n_states = self.env.get_states_num()
        n_actions = self.env.get_actions_num()
        transition_matrix = np.zeros((n_states, n_actions, n_states))
        for observation in observations:
            transition_matrix[observation[0], observation[1], observation[2]] += 1

        normalizer = transition_matrix.sum(axis=2)

        # If state-action pair is never visited, uses uniform distribution
        transition_matrix += np.where(normalizer == 0, 1, 0)[..., None]
        transition_matrix /= np.sum(transition_matrix, 2)[..., None]
        return transition_matrix, normalizer

    def reset(self):
        self.observations = np.empty((0, 3), dtype=int)

    @abstractmethod
    def _estimate(self):
        ...

    @abstractmethod
    def _isin_conf(self, mu: np.ndarray) -> List[bool]:
        ...
