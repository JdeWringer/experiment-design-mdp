import numpy as np
from mdpexplore.env.linear_worlds import MachineRepairSimplex
from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP
from mdpexplore.policies.average_policy import AveragePolicy
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.utils.reward_functionals import DesignBayesD, DesignD
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.estimators.wls import WLS
from mdpexplore.solvers.nominal import Nominal
from mdpexplore.solvers.optimistic import Optimistic
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gridworlds Problem.")
    parser.add_argument("--seed", default=12, type=int, help="Use this to set the seed for the random number generator")
    parser.add_argument(
        "--save", default="experiments/machine_repair/experiment.csv", type=str, help="name of the file"
    )
    parser.add_argument("--cores", default=None, type=int, help="number of cores")
    parser.add_argument("--verbosity", default=2, type=int, help="Use this to increase debug ouput")
    parser.add_argument("--accuracy", default=None, type=float, help="Termination criterion for optimality gap")
    parser.add_argument("--policy", default="average", type=str, help="Summarized policy type (mixed/average/density)")
    parser.add_argument("--num_components", default=1, type=int, help="Number of MaxEnt components (basic policies)")
    parser.add_argument("--episodes", default=50, type=int, help="Number of evaluation policy unrolls")
    parser.add_argument("--repeats", default=1, type=int, help="Number of repeats")
    parser.add_argument("--design", default="adaptive", type=str, help="Number of repeats")
    parser.add_argument("--opt", default="false", type=str, help="Number of repeats")
    parser.add_argument("--linesearch", default=None, type=str, help="type")
    parser.add_argument("--savetrajectory", default=None, type=str, help="type")
    parser.add_argument("--random", default="false", type=str, help="type")
    parser.add_argument("--solver", default="optimistic", type=str, help="type")
    parser.add_argument("--estimator", default="WLS", type=str, help="type")

    args = parser.parse_args()

    if args.policy == "average":
        args.policy = AveragePolicy
    elif args.policy == "density":
        args.policy = DensityPolicy
    else:
        raise ValueError("Invalid policy type")

    env = MachineRepairSimplex(max_episode_length=30, seed=args.seed)

    if args.design == "adaptive":
        design = DesignBayesD(env, scale_reg=False, uniform_alpha=False, lambd=1e-1)
    elif args.design == "fixed":
        design = DesignD(env, lambd=1e-1)
    else:
        raise ValueError("Invalid design type")

    initial_policy = False

    if args.random == "true":
        initial_policy = True
        args.num_components = 1

    if args.estimator == "WLS":
        estimator = WLS(env)
    elif args.estimator == "None":
        estimator = None
    else:
        raise ValueError("Invalid estimator type")

    if args.solver == "LP":
        solver = LP(env)
    elif args.solver == "DP":
        solver = DP(env)
    elif args.solver == "nominal":
        solver = Nominal(env, estimator)
    elif args.solver == "optimistic":
        solver = Optimistic(env, estimator)
    else:
        raise ValueError("Invalid solver type")

    me = MdpExplore(
        env,
        objective=design,
        solver=solver,
        step=args.linesearch,
        method="frank-wolfe",
        verbosity=args.verbosity,
        initial_policy=initial_policy,
    )

    val, opt_val = me.run(
        num_components=args.num_components,
        episodes=args.episodes,
        SummarizedPolicyType=args.policy,
        accuracy=args.accuracy,
        save_trajectory=args.savetrajectory,
    )
    vals = np.array(val)
    np.savetxt(args.save, vals)
    if args.opt == "true":
        np.savetxt("results/opt.txt", np.array([opt_val]))
