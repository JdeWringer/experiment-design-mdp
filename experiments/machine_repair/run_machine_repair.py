import numpy as np
from mdpexplore.env.linear_worlds import MachineRepairSimplex
from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP
from mdpexplore.policies.average_policy import AveragePolicy
from mdpexplore.utils.reward_functionals import DesignBayesD
from mdpexplore.mdpexplore import MdpExplore
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gridworlds Problem.')
    parser.add_argument('--seed', default=12, type=int,
                        help='Use this to set the seed for the random number generator')
    parser.add_argument('--save', default="experiment.csv", type=str, help='name of the file')
    parser.add_argument('--cores', default=None, type=int, help='number of cores')
    parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
    parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
    parser.add_argument('--policy', default='average', type=str,
                        help='Summarized policy type (mixed/average/density)')
    parser.add_argument('--num_components', default=1, type=int,
                        help='Number of MaxEnt components (basic policies)')
    parser.add_argument('--episodes', default=4, type=int, help='Number of evaluation policy unrolls')
    parser.add_argument('--repeats', default=1, type=int, help='Number of repeats')
    parser.add_argument('--adaptive', default="Bayes", type=str, help='Number of repeats')
    parser.add_argument('--opt', default="false", type=str, help='Number of repeats')
    parser.add_argument('--linesearch', default=None, type=str, help="type")
    parser.add_argument('--savetrajectory', default=None, type=str, help="type")
    parser.add_argument('--random', default="false", type=str, help="type")
    parser.add_argument('--solver', default="LP", type=str, help="type")

    args = parser.parse_args()

    if args.policy == 'average':
        args.policy = AveragePolicy
    else:
        raise ValueError('Invalid policy type')

    env = MachineRepairSimplex(max_episode_length=30, seed=args.seed)

    design = DesignBayesD(env, scale_reg=False, uniform_alpha=False, lambd=1e-1)

    initial_policy = False

    if args.random == "true":
        initial_policy = True
        args.num_components = 1
        
    if args.solver == 'LP':
        solver = LP
    elif args.solver == 'DP':
        solver = DP
    else:
        raise ValueError('Invalid solver type')

    me = MdpExplore(
        env,
        objective=design,
        solver=solver,
        step=args.linesearch,
        method='frank-wolfe',
        verbosity=args.verbosity,
        initial_policy=initial_policy
    )

    val, opt_val = me.run(
        num_components=args.num_components,
        episodes=args.episodes,
        SummarizedPolicyType=args.policy,
        accuracy=args.accuracy,
        save_trajectory=args.savetrajectory
    )
    vals = np.array(val)
    np.savetxt(args.save, vals)
    if args.opt == "true":
        np.savetxt("results/opt.txt", np.array([opt_val]))
