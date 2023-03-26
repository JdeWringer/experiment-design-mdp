import autograd.numpy as np
from numpy.random import default_rng
from abc import ABC, abstractmethod

from mdpexplore.env.discrete_env import DiscreteEnv

class LinearWorldBase(DiscreteEnv, ABC):
    @abstractmethod
    def __init__(
            self,
            states_num: int,
            actions_num: int,
            latent_dim: int,
            d0,
            max_episode_length: int = 50,
            seed: float = None,
    ) -> None:
        """Simplex class, representing a linear MDP satisfying certain normalization constraints.

        Args:
            states_num (int, optional): number of states. Defaults to 42.
            max_episode_length (int, optional): maximum episode length. Defaults to 50.
            seed (int, optional): random seed. Defaults to None.
        """
        self.d0 = d0
        self.rng = default_rng(seed)

        self.states_num = states_num
        self.actions_num = actions_num
        self.actions = np.arange(actions_num)
        self.latent_dim = latent_dim

        self.max_episode_length = max_episode_length

        self.emissions = self.generate_feature_vectors()
        self.mu = self.generate_mu()

        assert np.isclose(np.sum(self.emissions, axis=2), 1).all()
        assert np.isclose(np.sum(self.mu, axis=0), 1).all()

        self.transition_matrix = None

        self.state = None
        self.reset()

    @abstractmethod
    def generate_feature_vectors(self) -> np.ndarray:
        """Every simplex MDP must have a set of feature vectors \phi(s, a)"""
        ...

    @abstractmethod
    def generate_mu(self) -> np.ndarray:
        """Every simplex MDP must have an unknown matrix that defines the transition matrix \mu(s')"""
        ...

    def available_actions(self, state: int):
        return self.actions

    def next(self, state: int, action: int) -> int:
        """Returns the next state after taking the given action from the given state

        Args:
            state (int): current state ID
            action (int): action ID

        Returns:
            int: state ID after taking the given action from the given state
        """
        P = self.get_transition_matrix()
        next_state = self.rng.choice(len(P), p=P[state, action, :] / P[state, action, :].sum())

        return next_state

    def step(self, action: int):
        """Takes the given action and updates current state.

        Args:
            action (int): ID of action to be taken

        """
        self.visitations[self.state, action] += 1
        current_state = self.state
        self.state = self.next(current_state, action)
        self.observations.append([current_state, action, self.state])
        return self.state

    def get_transition_matrix(self) -> np.ndarray:
        """Returns the transition matrix P(s'|s,a)

        Returns:
            np.ndarray: transition matrix. Index with state, action, next_state.
        """
        if self.transition_matrix is not None:
            return self.transition_matrix

        P = np.tensordot(self.emissions, self.mu.T, axes=1)

        # Check normalization of transition matrix.
        assert np.isclose(np.sum(P, axis=2), 1).all()

        return P
    
    def is_valid_action(self, action, state) -> bool:
        return True

    def reset(self) -> None:
        """Resets the environment to its initial state"""
        self.visitations = np.zeros((self.states_num, self.actions_num))
        self.observations = []
        self.state = np.random.choice(range(len(self.d0)), p=self.d0)
