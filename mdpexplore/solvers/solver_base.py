import autograd.numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy


class DiscreteSolver(ABC):
    def __init__(
        self,
        env: DiscreteEnv,
    ) -> None:

        self.env = env

    @abstractmethod
    def solve(self, reward) -> Policy:
        ...