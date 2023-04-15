import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import itertools
import collections


def runner(f):
    def wrapper(file_path: str, run_str: str):
        file_path = pathlib.Path(str(file_path) + ".csv")
        if not pathlib.Path(file_path).is_file():
            f(file_path, run_str)
        return np.loadtxt(file_path, dtype=float)

    return wrapper


def run_experiment(file_path: str, experiment_name: str, params: dict = None, n_runs: int = 1):
    run_str_base = ["python", f"experiments/{experiment_name}/run_{experiment_name}.py"]
    if params is not None:
        file_path_no_seed = file_path
        results = collections.defaultdict(list)
        for config in itertools.product(*params.values()):
            run_str_no_seed = run_str_base
            config_str = "config:"
            for key, param in zip(params.keys(), config):
                config_str += f"__{key}:{param}"
                run_str_no_seed += [f"--{key}", str(param)]
                file_path_no_seed = pathlib.Path(str(file_path_no_seed) + f"_{key}:{param}")

            for i in range(n_runs):
                run_str = run_str_no_seed + ["--seed", str(i)]
                file_path = pathlib.Path(str(file_path_no_seed) + f"_iter:{i}")
                results[config_str].append(_run_experiment(file_path, run_str))
        return results
    else:
        return {"default_config": _run_experiment(file_path, run_str_base)}


@runner
def _run_experiment(file_path, run_str):
    run_str += ["--save", file_path]
    subprocess.run(run_str)


if __name__ == "__main__":
    n_runs = 1
    results_folder = pathlib.Path("experiments/results")
    figs_folder = pathlib.Path("experiments/figs")
    pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(figs_folder).mkdir(parents=True, exist_ok=True)

    experiments = ["machine_repair"]
    for experiment_id, experiment_name in enumerate(experiments):
        results = collections.defaultdict(dict)
        results_folder /= pathlib.Path(experiment_name)

        # Optimal run
        file_path = results_folder / pathlib.Path(experiment_name + "_opt")
        params = {"solver": ["DP"], "episodes": [300], "estimator": ["None"]}
        results[experiment_name] |= run_experiment(file_path, experiment_name, params, n_runs=n_runs)

        # Experiment runs
        params = {"solver": ["nominal", "optimistic"], "episodes": [128]}
        file_path = results_folder / pathlib.Path(experiment_name)
        # Keys are experiment_name, config
        results[experiment_name] |= run_experiment(file_path, experiment_name, params, n_runs=n_runs)

    for experiment_name, results in results.items():
        result_configs = {}
        for config in results:
            params = dict([tuple(e.split(":")) for e in config.split("__")[1:]])
            if "solver" in params and params["solver"] == "DP":
                optimum = np.mean(results[config], axis=0)[-1]
                # result_configs[params["solver"]] = pd.DataFrame(
                #     np.array(results[config])[:, :50]
                # )  # TODO remove this line
            else:
                result_configs[params["solver"]] = pd.DataFrame(np.array(results[config]))
        fig, ax = plt.subplots()
        for label, result_config in result_configs.items():
            result_config = -result_config + optimum
            sns.lineplot(data=pd.melt(result_config), x="variable", y="value", errorbar="sd", ax=ax, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        plt.xlabel("runs")
        plt.ylabel("Sub-optimality")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_folder / pathlib.Path(experiment_name))
    # print(pd.DataFrame.from_dict(results))
