from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP


class Nominal(DiscreteSolver):
    def __init__(self, env, estimator):
        super().__init__(env)
        self.estimator = estimator

    def solve(self, reward) -> Policy:
        return DP(self.env).solve(reward, self.estimator.estimate())
