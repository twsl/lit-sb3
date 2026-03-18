"""Argument parser and logic for the ``plot_from_file`` sub-command."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import pickle
from typing import Any
import warnings

import numpy as np

try:
    from matplotlib import pyplot as plt
    import pandas as pd
    import pytablewriter  # ty:ignore[unresolved-import]
    import seaborn  # ty:ignore[unresolved-import]

    _PLOT_DEPS_AVAILABLE = True
except ImportError:
    _PLOT_DEPS_AVAILABLE = False

try:
    from rliable import library as rly  # ty:ignore[unresolved-import]
    from rliable import metrics, plot_utils  # ty:ignore[unresolved-import]

    _RLIABLE_AVAILABLE = True
except ImportError:
    _RLIABLE_AVAILABLE = False
    rly = None

try:
    from lit_sb3.zoo.plots.score_normalization import normalize_score

    _SCORE_NORM_AVAILABLE = True
except ImportError:
    _SCORE_NORM_AVAILABLE = False


# From https://github.com/mwaskom/seaborn/blob/master/seaborn/categorical.py
def restyle_boxplot(artist_dict: dict[str, Any], color: Any, gray: str = "#222222", linewidth: float = 1, fliersize: float = 5) -> None:
    """Style a matplotlib boxplot to look nicer."""
    for box in artist_dict["boxes"]:
        box.update({"facecolor": color, "zorder": 0.9, "edgecolor": gray, "linewidth": linewidth})

    for whisk in artist_dict["whiskers"]:
        whisk.update({"color": gray, "linewidth": linewidth, "linestyle": "-"})

    for cap in artist_dict["caps"]:
        cap.update({"color": gray, "linewidth": linewidth})

    for med in artist_dict["medians"]:
        med.update({"color": gray, "linewidth": linewidth})

    for fly in artist_dict["fliers"]:
        fly.update({"markerfacecolor": gray, "marker": "d", "markeredgecolor": gray, "markersize": fliersize})


def _add_plot_from_file_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", help="Input filename (numpy archive)", type=str)
    parser.add_argument("-skip", "--skip-envs", help="Environments to skip", nargs="+", default=[], type=str)
    parser.add_argument("--keep-envs", help="Envs to keep", nargs="+", default=[], type=str)
    parser.add_argument("--skip-keys", help="Keys to skip", nargs="+", default=[], type=str)
    parser.add_argument("--keep-keys", help="Keys to keep", nargs="+", default=[], type=str)
    parser.add_argument("--no-million", action="store_true", default=False, help="Do not convert x-axis to million")
    parser.add_argument("--skip-timesteps", action="store_true", default=False, help="Do not display learning curves")
    parser.add_argument("-o", "--output", help="Output filename (image)", type=str)
    parser.add_argument("--format", help="Output format", type=str, default="svg")
    parser.add_argument("-loc", "--legend-loc", help="The location of the legend.", type=str, default="best")
    parser.add_argument(
        "--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8]
    )
    parser.add_argument("--fontsize", help="Font size", type=int, default=14)
    parser.add_argument("-l", "--labels", help="Custom labels", type=str, nargs="+")
    parser.add_argument("-b", "--boxplot", help="Enable boxplot", action="store_true", default=False)
    parser.add_argument("-r", "--rliable", help="Enable rliable plots", action="store_true", default=False)
    parser.add_argument(
        "-vs", "--versus", help="Enable probability of improvement plot", action="store_true", default=False
    )
    parser.add_argument("-iqm", "--iqm", help="Enable IQM sample efficiency plot", action="store_true", default=False)
    parser.add_argument("-ci", "--ci-size", help="Confidence interval size (for rliable)", type=float, default=0.95)
    parser.add_argument("-latex", "--latex", help="Enable latex support", action="store_true", default=False)
    parser.add_argument("--merge", help="Merge with other results files", nargs="+", default=[], type=str)


def run_plot_from_file(args: argparse.Namespace) -> None:  # noqa: C901
    """Execute the plot_from_file command with pre-parsed args."""
    if not _PLOT_DEPS_AVAILABLE:
        raise ImportError(
            "matplotlib, seaborn, pandas and pytablewriter are required for plotting. "
            "Install them with: pip install matplotlib seaborn pandas pytablewriter"
        )

    # Activate seaborn
    seaborn.set()
    seaborn.set(style="whitegrid")

    # Enable LaTeX support
    if args.latex:
        plt.rc("text", usetex=True)

    filename = args.input

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    with Path(filename).open("rb") as file_handler:
        results = pickle.load(file_handler)  # noqa: S301

    # Plot table
    writer = pytablewriter.MarkdownTableWriter(max_precision=3)
    writer.table_name = "results_table"
    writer.headers = results["results_table"]["headers"]
    writer.value_matrix = results["results_table"]["value_matrix"]
    writer.write_table()

    del results["results_table"]

    for merge_filename in args.merge:
        # Merge other files
        with Path(merge_filename).open("rb") as file_handler:
            results_2 = pickle.load(file_handler)  # noqa: S301
            del results_2["results_table"]
            for key in results:
                if key in results_2:
                    for new_key in results_2[key]:
                        results[key][new_key] = results_2[key][new_key]

    keys = [key for key in results[next(iter(results.keys()))] if key not in args.skip_keys]
    print(f"keys: {keys}")
    if len(args.keep_keys) > 0:
        keys = [key for key in keys if key in args.keep_keys]
    envs = [env for env in results if env not in args.skip_envs]

    if len(args.keep_envs) > 0:
        envs = [env for env in envs if env in args.keep_envs]

    labels = {key: key for key in keys}
    if args.labels is not None:
        for key, label in zip(keys, args.labels, strict=True):
            labels[key] = label

    if not args.skip_timesteps:
        # Plot learning curves per env
        for env in envs:
            plt.figure(f"Results {env}")
            title = f"{env}"
            plt.title(title, fontsize=args.fontsize)

            x_label_suffix = "" if args.no_million else "(1e6)"
            plt.xlabel(f"Timesteps {x_label_suffix}", fontsize=args.fontsize)
            plt.ylabel("Score", fontsize=args.fontsize)

            for key in keys:
                # x axis in Millions of timesteps
                divider = 1e6
                if args.no_million:
                    divider = 1.0

                timesteps = results[env][key]["timesteps"]
                mean_ = results[env][key]["mean"]
                std_error = results[env][key]["std_error"]

                plt.xticks(fontsize=13)
                plt.plot(timesteps / divider, mean_, label=labels[key], linewidth=3)
                plt.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.5)

            plt.legend(fontsize=args.fontsize)
            plt.tight_layout()

    # Convert to pandas dataframe to use with seaborn
    labels_df, envs_df, scores = [], [], []

    # Post-process to use it with rliable
    # algo: (n_runs, n_envs)
    normalized_score_dict: dict = {}
    # algo: (n_runs, n_envs, n_eval)
    all_eval_normalized_scores_dict: dict = {}
    # Convert env key to env id for normalization
    env_key_to_env_id = {
        "Half": "HalfCheetahBulletEnv-v0",
        "Ant": "AntBulletEnv-v0",
        "Hopper": "HopperBulletEnv-v0",
        "Walker": "Walker2DBulletEnv-v0",
        "LunarLanderContinuous": "LunarLanderContinuous-v3",
        "BipedalWalker": "BipedalWalker-v3",
    }
    # Backward compat
    skip_all_algos_dict = False

    for key in keys:
        algo_scores, all_algo_scores = [], []
        for env in envs:
            if isinstance(results[env][key]["last_evals"], (np.float32, np.float64)):
                # Not enough timesteps
                print(f"Skipping {env}-{key}")
                continue

            for score in results[env][key]["last_evals"]:
                labels_df.append(labels[key])
                envs_df.append(env)
                scores.append(score)

            algo_scores.append(results[env][key]["last_evals"])

            # Backward compat: mean_per_eval key may not be present
            if "mean_per_eval" in results[env][key]:
                all_algo_scores.append(results[env][key]["mean_per_eval"])
            else:
                skip_all_algos_dict = True

            # Normalize score, env key must match env_id
            if env in env_key_to_env_id and _SCORE_NORM_AVAILABLE:
                algo_scores[-1] = normalize_score(algo_scores[-1], env_key_to_env_id[env])
                if not skip_all_algos_dict:
                    all_algo_scores[-1] = normalize_score(all_algo_scores[-1], env_key_to_env_id[env])
            elif env not in env_key_to_env_id and args.rliable:
                warnings.warn(
                    f"{env} not found for normalizing scores, you should update `env_key_to_env_id`", stacklevel=2
                )

        # Truncate to convert to matrix
        min_runs = min(len(algo_score) for algo_score in algo_scores)
        if min_runs > 0:
            algo_scores = [algo_score[:min_runs] for algo_score in algo_scores]
            # shape: (n_envs, n_runs) -> (n_runs, n_envs)
            normalized_score_dict[labels[key]] = np.array(algo_scores).T

        if not skip_all_algos_dict:
            all_algo_scores = [all_algo_score[:, :min_runs] for all_algo_score in all_algo_scores]
            # (n_envs, n_eval, n_runs) -> (n_runs, n_envs, n_eval)
            all_eval_normalized_scores_dict[labels[key]] = np.array(all_algo_scores).transpose((2, 0, 1))

    data_frame = pd.DataFrame(data={"Method": labels_df, "Environment": envs_df, "Score": scores})

    # Rliable plots
    if args.rliable:
        if rly is None or not _RLIABLE_AVAILABLE:
            raise ImportError("You must install rliable package to use this feature: pip install rliable")

        print("Computing bootstrap CI ...")
        algorithms = list(labels.values())

        aggregate_func = lambda x: np.array(  # noqa: E731
            [
                metrics.aggregate_median(x),
                metrics.aggregate_iqm(x),
                metrics.aggregate_mean(x),
                metrics.aggregate_optimality_gap(x),
            ]
        )
        aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
            normalized_score_dict,
            aggregate_func,
            reps=2000,
            confidence_interval_size=args.ci_size,
        )

        fig, _axes = plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_interval_estimates,
            metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
            algorithms=algorithms,
            xlabel="Normalized Score",
            xlabel_y_coordinate=0.02,
            subfigure_width=5,
            row_height=1,
            max_ticks=4,
            interval_height=0.6,
        )
        fig.canvas.manager.set_window_title("Rliable metrics")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # Performance profiles
        normalized_score_thresholds = np.linspace(0.0, 1.5, 50)
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            normalized_score_dict,
            normalized_score_thresholds,
            reps=2000,
            confidence_interval_size=args.ci_size,
        )

        fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
        plot_utils.plot_performance_profiles(
            score_distributions,
            normalized_score_thresholds,
            performance_profile_cis=score_distributions_cis,
            colors=dict(zip(algorithms, seaborn.color_palette("colorblind"), strict=False)),
            xlabel=r"Normalized Score $(\tau)$",
            ax=ax,
        )
        fig.canvas.manager.set_window_title("Performance profiles")  # ty:ignore[unresolved-attribute]
        plt.legend()

        # Probability of improvement
        if args.versus:
            algorithm_pairs_keys = itertools.combinations(algorithms, 2)
            algorithm_pairs = {}
            for algo1, algo2 in algorithm_pairs_keys:
                algorithm_pairs[f"{algo1}, {algo2}"] = (normalized_score_dict[algo1], normalized_score_dict[algo2])

            average_probabilities, average_prob_cis = rly.get_interval_estimates(
                algorithm_pairs,
                metrics.probability_of_improvement,
                reps=1000,
                confidence_interval_size=args.ci_size,
            )
            plot_utils.plot_probability_of_improvement(
                average_probabilities,
                average_prob_cis,
                figsize=(10, 8),
                interval_height=0.6,
            )
            plt.gcf().canvas.manager.set_window_title("Probability of Improvement")  # ty:ignore[unresolved-attribute]
            plt.tight_layout()

        if args.iqm:
            downsample_factor = 4
            n_evals = all_eval_normalized_scores_dict[algorithms[0]].shape[-1]
            eval_indices = np.arange(n_evals - 1)[::downsample_factor]
            eval_indices = np.concatenate((eval_indices, [n_evals - 1]))
            eval_indices_scores_dict = {
                algorithm: score[:, :, eval_indices] for algorithm, score in all_eval_normalized_scores_dict.items()
            }
            iqm = lambda scores: np.array(  # noqa: E731
                [metrics.aggregate_iqm(scores[..., eval_idx]) for eval_idx in range(scores.shape[-1])]
            )
            iqm_scores, iqm_cis = rly.get_interval_estimates(
                eval_indices_scores_dict,
                iqm,
                reps=2000,
                confidence_interval_size=args.ci_size,
            )
            plot_utils.plot_sample_efficiency_curve(
                eval_indices + 1,
                iqm_scores,
                iqm_cis,
                algorithms=algorithms,
                xlabel=r"Number of Evaluations",
                ylabel="IQM Normalized Score",
            )
            plt.gcf().canvas.manager.set_window_title("IQM Normalized Score - Sample Efficiency Curve")  # ty:ignore[unresolved-attribute]
            plt.legend()
            plt.tight_layout()

        plt.show()

    # Plot final results with env as x axis
    plt.figure("Sensitivity plot", figsize=args.figsize)
    plt.title("Sensitivity plot", fontsize=args.fontsize)
    plt.xticks(fontsize=13)
    plt.xlabel("Environment", fontsize=args.fontsize)
    plt.ylabel("Score", fontsize=args.fontsize)

    ax = seaborn.barplot(x="Environment", y="Score", hue="Method", data=data_frame)
    plt.tight_layout()
    if args.output is not None:
        plt.savefig(args.output, format=args.format)

    if args.boxplot:
        # Box plot
        plt.figure("Sensitivity box plot", figsize=args.figsize)
        plt.title("Sensitivity box plot", fontsize=args.fontsize)
        plt.xticks(fontsize=13)
        plt.ylabel("Score", fontsize=args.fontsize)

        data, labels_ = [], []
        for env in envs:
            for key in keys:
                data.append(results[env][key]["last_evals"])
                text = f"{env}-{labels[key]}" if len(envs) > 1 else labels[key]
                labels_.append(text)
        artist_dict = plt.boxplot(data, patch_artist=True)
        color_palette = seaborn.color_palette()
        # orange
        boxplot_color = color_palette[1]
        restyle_boxplot(artist_dict, color=boxplot_color)
        plt.xticks(np.arange(1, len(data) + 1), labels_, rotation=0)
        plt.tight_layout()

    plt.show()


def plot_from_file() -> None:
    """Load results from a pickle file and plot them (parse args then run)."""
    parser = argparse.ArgumentParser("Gather results, plot them and create table")
    _add_plot_from_file_args(parser)
    args = parser.parse_args()
    run_plot_from_file(args)
