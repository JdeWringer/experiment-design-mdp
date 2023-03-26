import math
import itertools
import autograd.numpy as np

from mdpexplore.env.linear_world_base import LinearWorldBase

class MachineRepairSimplex(LinearWorldBase):
    def __init__(self, max_episode_length: int = 50, seed: int = None):
        self.rng_mr = np.random.default_rng(2049)
        n_machines = 3
        n_levels = 3
        n_mechanics = 2

        self.n_machines = n_machines
        self.n_levels = n_levels
        self.n_mechanics = n_mechanics

        states_num = n_levels ** n_machines
        actions_num = n_machines ** n_mechanics + 1
        latent_action_dim = math.comb(n_mechanics + n_machines - 1, n_machines - 1) + 1
        latent_dim = states_num * latent_action_dim
        self.latent_action_dim = latent_action_dim

        self.idx2machstatus = list(itertools.product(*[np.arange(n_levels) for _ in range(n_machines)]))
        self.embedidx2mechconfig = list(self.pigeonhole(n_mechanics, n_machines))
        self.idx2mechconfig = np.array(list(itertools.product(*[np.arange(n_machines) for _ in range(n_mechanics)])))

        d0 = np.zeros(states_num)
        d0[self.idx2machstatus.index(tuple([n_levels - 1] * n_machines))] = 1
        super().__init__(
            states_num=states_num,
            actions_num=actions_num,
            latent_dim=latent_dim,
            max_episode_length=max_episode_length,
            d0=d0,
            seed=seed,
        )
        self.constrained = False
        self.discount_factor = 0.99

    @staticmethod
    def pigeonhole(n, m):
        """
        Generate the ways n balls can be placed in m slots
        Taken from https://stackoverflow.com/questions/22939260/every-way-to-organize-n-objects-in-m-list-slots.
        """
        for choice in itertools.combinations(range(n + m - 1), n):
            slot = [c - i for i, c in enumerate(choice)]
            result = [0] * m
            for i in slot:
                result[i] += 1
            yield result

    def generate_feature_vectors(self):

        n_machines = self.n_machines

        phi = np.zeros((self.states_num, self.actions_num, self.latent_dim))
        latent_action_dim = self.latent_action_dim

        for state in range(self.states_num):
            for action in range(self.actions_num):
                if action < self.actions_num - 1:
                    mech_config = self.idx2mechconfig[action]
                    mid_repres = np.zeros(n_machines)
                    for mech_loc in mech_config:
                        mid_repres[mech_loc] += 1
                    action_idx = self.embedidx2mechconfig.index(list(mid_repres))
                    latent_idx = state * latent_action_dim + action_idx
                else:
                    latent_idx = state * latent_action_dim + self.latent_action_dim - 1

                phi[state, action, latent_idx] = 1
        return phi

    def generate_mu(self) -> np.ndarray:
        def get_dist(mech_config, mach_status) -> np.ndarray:
            machine_dists = list()
            for i, machine_level in enumerate(mach_status):
                machine_dist = np.zeros(self.n_levels)
                if mech_config[i] >= 1:
                    repair_prob = 1
                    # Chance of repair
                    machine_dist[self.n_levels - 1] = repair_prob

                    # Chance of staying
                    machine_dist[machine_level] += 1 - repair_prob
                else:
                    if machine_level == 0:
                        machine_dist[machine_level] = 1
                    else:
                        # Probability of degrading.
                        alpha = self.rng_mr.uniform(0.2, 0.8)
                        machine_dist[machine_level - 1] = alpha
                        machine_dist[machine_level] = 1 - alpha
                machine_dists.append(machine_dist)
            machine_dists = np.array(machine_dists)
            result_dist = np.zeros(self.states_num)
            for idx in range(self.states_num):
                new_mach_status = self.idx2machstatus[idx]
                result_dist[idx] = np.prod([machine_dists[m, l] for m, l in enumerate(new_mach_status)])

            return result_dist

        mu = np.zeros((self.states_num, self.latent_dim))

        for d in range(self.latent_dim):
            latent_action_idx = d % self.latent_action_dim
            state_idx = d // self.latent_action_dim

            if latent_action_idx < self.latent_action_dim - 1:
                mech_config = self.embedidx2mechconfig[latent_action_idx]
            else:
                mech_config = np.zeros_like(self.embedidx2mechconfig[0])
            mach_status = self.idx2machstatus[state_idx]
            mu[:, d] = get_dist(mech_config, mach_status)
        return mu
    
    def get_dim(self):
        return self.latent_dim
    
    def get_states_num(self):
        return self.states_num
    
    def get_actions_num(self):
        return self.actions_num