import cvxpy as cp
import numpy as np

from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.simple_policy import SimplePolicy


class Optimistic(DiscreteSolver):
    def __init__(self, env, estimator):
        super().__init__(env)
        self.estimator = estimator

    # def solve(self, reward) -> Policy:
    #     ps = [
    #         cp.Variable((self.env.get_states_num(), self.env.get_actions_num()))
    #         for _ in range(self.env.max_episode_length)
    #     ]
    #     mu = cp.Variable((self.env.states_num, self.env.latent_dim))
    #     transition_matrix = np.reshape(self.env.emissions, (-1, self.env.emissions.shape[2])) @ mu.T

    #     objective = 0
    #     prev_visitation_state = self.env.d0
    #     for p in ps:
    #         visitation = cp.multiply(prev_visitation_state[:, None], p)
    #         objective += cp.sum(cp.multiply(visitation, reward))
    #         prev_visitation_state = visitation.flatten() @ transition_matrix
    #     objective = cp.Maximize(objective)
    #     constraints = self.estimator.isin_conf(mu)

    #     problem = cp.Problem(objective, constraints)
    #     problem.solve(solver=cp.MOSEK)

    #     return NonStationaryPolicy(self.env, np.array([p.value for p in ps]))

    def solve(self, reward) -> NonStationaryPolicy:
        emissions = np.reshape(self.env.emissions, (-1, self.env.emissions.shape[2]))
        value_next = np.zeros(self.env.get_states_num(), dtype=float)
        ps = []
        for _ in range(self.env.max_episode_length):
            mu = cp.Variable((self.env.states_num, self.env.latent_dim))
            transition_matrix = emissions @ mu.T
            constraints = self.estimator.isin_conf(mu)
            constraints += [cp.sum(transition_matrix, axis=1) == 1, transition_matrix >= 0]
            objective = cp.Maximize(cp.sum(transition_matrix @ value_next))
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)

            q_function = reward + np.reshape(emissions @ mu.value.T @ value_next, reward.shape)

            p = cp.Variable((self.env.get_states_num(), self.env.get_actions_num()), nonneg=True)
            objective = cp.Maximize(cp.sum(cp.multiply(p, q_function)))
            constraints = [cp.sum(p, axis=1) == 1]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)

            ps.append(p.value)
            value_next = np.sum(p.value * q_function, axis=1)

        return SimplePolicy(self.env, np.mean(np.array(ps), axis=0))
        # return NonStationaryPolicy(self.env, ps[::-1])
