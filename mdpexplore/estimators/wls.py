from typing import List, Any, Tuple

import cvxpy as cp
import mosek
import numpy as np
from mdpexplore.estimators.estimator_base import Estimator


class WLS(Estimator):

    def objective(self, mu, observed_transition_matrix, observed_feature_vectors, observed_normalizer):
        """Calculates the penalized WS objective"""
        inner_products = observed_feature_vectors @ mu.T

        squared_errors = (inner_products - observed_transition_matrix) ** 2

        # Use an upper bound on the variance.
        WS = cp.sum(cp.multiply(squared_errors, observed_normalizer[:, None]) / 4.)

        penalty = cp.norm(mu, "fro") ** 2
        return WS + self.lambd * penalty

    def _estimate(self):
        mu = cp.Variable((self.env.states_num, self.env.latent_dim))
        
        observations = self.observations
        
        empirical_transition_matrix, normalizer = self.estimate_transition_matrix(observations)

        observed_feature_vectors = self.env.emissions[observations[:, 0], observations[:, 1]]
        observed_normalizer = normalizer[observations[:, 0], observations[:, 1]]
        observed_transition_matrix = empirical_transition_matrix[
            observations[:, 0], observations[:, 1]
        ]
        objective = cp.Minimize(
            self.objective(mu, observed_transition_matrix, observed_feature_vectors, observed_normalizer))

        feature_vectors = np.reshape(self.env.emissions, (-1, self.env.get_dim()))
        transition_matrix = feature_vectors @ mu.T
        constraints = [cp.sum(transition_matrix, axis=1) == 1,
                       transition_matrix >= 0]
        
        problem = cp.Problem(objective, constraints)

        # problem.solve(solver=cp.MOSEK, mosek_params={mosek.dparam.intpnt_co_tol_near_rel: 1e6})
        problem.solve(solver=cp.MOSEK)

        return np.tensordot(self.env.emissions, mu.value.T, axes=1)
