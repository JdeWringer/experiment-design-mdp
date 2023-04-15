from typing import List, Any, Tuple

import cvxpy as cp
import numpy as np
from mdpexplore.estimators.estimator_base import Estimator


class WLS(Estimator):
    def objective(self, mu, observed_transition_matrix, observed_feature_vectors, observed_normalizer, counts):
        """Calculates the penalized WS objective"""
        inner_products = observed_feature_vectors @ mu.T

        squared_errors = (cp.multiply(counts[:, None], (inner_products - observed_transition_matrix))) ** 2

        # Use an upper bound on the variance.
        WS = cp.sum(cp.multiply(squared_errors, observed_normalizer[:, None]) / 4.0)

        penalty = cp.norm(mu, "fro") ** 2
        return WS + self.lambd * penalty

    def calc_VT(self, counts):
        feature_vectors = np.reshape(self.env.emissions, (-1, self.env.emissions.shape[2]))
        V0 = self.lambd * np.eye(self.env.get_dim())
        VT = feature_vectors.T @ np.diag(counts.flatten() * 4) @ feature_vectors + V0
        return VT

    def _isin_conf(self, mu: np.ndarray) -> List[bool]:
        counts_indices, counts_num = np.unique(self.observations[:, :2], return_counts=True, axis=0)
        counts = np.zeros((self.env.get_states_num(), self.env.get_actions_num()))
        counts[counts_indices[:, 0], counts_indices[:, 1]] = counts_num
        mat_VT = self.calc_VT(counts)
        cholesky_VT = np.linalg.cholesky(mat_VT)
        return [
            cp.sum((cholesky_VT.T @ (mu[i] - self.estimated_mu[i])) ** 2) <= self.beta**2
            for i in range(self.env.states_num)
        ]

    def _estimate(self):
        mu = cp.Variable((self.env.states_num, self.env.latent_dim))
        empirical_transition_matrix, normalizer = self.estimate_transition_matrix(self.observations)

        observations, counts = np.unique(self.observations[:, :2], return_counts=True, axis=0)

        observed_feature_vectors = self.env.emissions[observations[:, 0], observations[:, 1]]
        observed_normalizer = normalizer[observations[:, 0], observations[:, 1]]
        observed_transition_matrix = empirical_transition_matrix[observations[:, 0], observations[:, 1]]
        objective = cp.Minimize(
            self.objective(mu, observed_transition_matrix, observed_feature_vectors, observed_normalizer, counts)
        )

        feature_vectors = np.reshape(self.env.emissions, (-1, self.env.get_dim()))
        transition_matrix = feature_vectors @ mu.T
        constraints = [cp.sum(transition_matrix, axis=1) == 1, transition_matrix >= 0]

        problem = cp.Problem(objective, constraints)
        problem.solve()
        self.estimated_mu = mu.value
        return np.tensordot(self.env.emissions, mu.value.T, axes=1)
