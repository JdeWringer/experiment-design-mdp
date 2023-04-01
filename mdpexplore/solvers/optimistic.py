from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.solvers.lp import LP


class Optimistic(DiscreteSolver):
    def __init__(self, env, estimator):
        super().__init__(env)
        self.estimator = estimator

    def solve(self, reward) -> Policy:
        return LP(self.env).solve(reward, self.estimator.estimate())
