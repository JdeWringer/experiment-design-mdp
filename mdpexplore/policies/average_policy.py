import autograd.numpy as np
from typing import List

from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import SummarizedPolicy
from mdpexplore.policies.simple_policy import SimplePolicy


class AveragePolicy(SummarizedPolicy):
    def __init__(self, env: DiscreteEnv, ps: List[SimplePolicy], weights: List[float]) -> None:
        super().__init__(env)
        self.ps = ps
        self.p_weights = weights

        p_avg = np.zeros((env.states_num, env.actions_num))
        for policy in self.ps:
            p_avg += policy.p
        p_avg /= len(self.ps)
        self.average_policy = SimplePolicy(env, p_avg)

    def next_action(self, state):
        return self.average_policy.next_action(state)