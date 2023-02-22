import autograd.numpy as np

from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.simple_policy import SimplePolicy


class DP(DiscreteSolver):
    def solve(self) -> Policy:
        '''
        Solves the MDP and returns the value function.
        '''
        transition_matrix = self.env.get_transition_matrix()
        
        actions = np.zeros((self.env.max_episode_length + 1, self.env.states_num), dtype=int)
        values = np.zeros((self.env.max_episode_length + 1, self.env.states_num))

        actions[self.env.max_episode_length, :] = 0 # pointing to the 'wait' action
        if self.env.constrained:
            values[self.env.max_episode_length, :] = -1e10
            values[self.env.max_episode_length, self.env.terminal_state] = \
                self.reward[self.env.terminal_state]
        else:
            values[self.env.max_episode_length, :] = self.reward

        for i in range(self.env.max_episode_length - 1, -1, -1):
            for state in range(self.env.states_num):
                acts = self.env.available_actions(state)
                new_values = np.array(
                    [self.reward[state] + transition_matrix[state, a] @ values[i+1] for a in acts]
                )
                optimal_actions = np.argwhere(new_values == np.max(new_values))
                idx = np.random.choice( 
                    optimal_actions.flatten()
                )
                best_act = acts[idx]
                actions[i, state] = best_act
                values[i, state] = new_values[idx]

        if not self.env.constrained:
            p = np.zeros((self.env.states_num, self.env.actions_num))
            for s in range(self.env.states_num):
                p[s, actions[i, s]] = 1.
            return SimplePolicy(self.env, p)

        ps = np.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))
        for i in range(self.env.max_episode_length):
            for s in range(self.env.states_num):
                ps[i, s, actions[i, s]] = 1.

        return NonStationaryPolicy(self.env, ps)