import autograd.numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy


class DiscreteSolver(ABC):
    def __init__(
        self,
        env: DiscreteEnv,
        reward: np.ndarray,
    ) -> None:

        self.env = env
        self.reward = reward

    @abstractmethod
    def solve(self) -> Policy:
        ...