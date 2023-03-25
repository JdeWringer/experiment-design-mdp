import autograd.numpy as np

from mdpexplore.env.discrete_env import DiscreteEnv

from mdpexplore.policies.simple_policy import SimplePolicy

class PolicyGenerator():
    # TODO: add a constrained policy?
    def __init__(self, env: DiscreteEnv) -> None:
        self.env = env

    def uniform_policy(self):
        '''
        Returns a uniform policy within the environment
        '''
        p = np.ones((self.env.states_num, self.env.actions_num))
        for s in range(self.env.states_num):
            for a in range(self.env.actions_num):
                if not self.env.is_valid_action(a, s):
                    p[s, a] = 0
        p /= np.sum(p, axis=1, keepdims=True)

        return SimplePolicy(self.env, p)
