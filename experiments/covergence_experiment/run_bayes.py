import matplotlib.pyplot as plt
import numpy as np
import os

from mdpexplore.utils.argparser import Parser

from mdpexplore.utils.reward_functionals import Design_BayesD, DesignD
from mdpexplore.env.env_builder import EnvBuilder
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.policies.average_policy import AveragePolicy
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy

from tqdm.contrib.concurrent import process_map 
import multiprocessing as mp

config = Parser().parse()
env = EnvBuilder(config).build()

repeats = config.repeats

policies = [DensityPolicy]
num_components_list = [config.num_components]
designs = [design_Bayes]
names = ['Bayesian']
colors = ['green']

results = []
results_std = []
opt = None
for policy, num_comp, design in zip(policies, num_components_list, designs):
    
    vals = []
    
    def run_single(_): 
        me = MdpExplore(
                env,
                objective=design,
                solver=config.solver,
                step=config.step,
                method=config.method,
                verbosity=config.verbosity
            )

        val, opt_val = me.run(
            num_components=num_comp,
            episodes=config.episodes,
            SummarizedPolicyType=policy,
            accuracy=config.accuracy,
            plot=False,
        )
        return val, opt_val
    
    cores = 26

    outputs = process_map(run_single,[repeat for repeat in range(repeats)], max_workers=cores)

    for repeat in range(repeats):
        val, opt_val = outputs[repeat]
        vals.append(val)
        if opt_val is not None:
            opt = opt_val 
    
    results.append(np.mean(np.array(vals), axis = 0))
    results_std.append(np.std(np.array(vals), axis = 0))

plt.figure(figsize=(10, 10))

x_axis = 1 + np.arange(results[0].shape[0])

for index,name in enumerate(names):
    plt.plot(x_axis, opt-results[index], label=name, color=colors[index])
    plt.fill_between(x_axis, opt-results[index] -results_std[index], opt-results[index] + results_std[index], label=name, color=colors[index], alpha = 0.2)

plt.plot(x_axis, 1/(x_axis**2), label='1/x**2', linestyle = '-', color='black')
plt.plot(x_axis, 1/(x_axis), label='1/x', linestyle = '-.', color='black')
plt.plot(x_axis, 1/np.sqrt(x_axis), label='1/sqrt(x)', linestyle = '--', color='black')
plt.legend(loc='lower right')

plt.grid()
plt.xlabel('Number of trajectories rolled out')
plt.ylabel('Objective function value')
plt.xscale('log', base = 2)
plt.yscale('log', base = 2)
plt.savefig(os.path.join('figs', 'objectives.png'))
plt.show()