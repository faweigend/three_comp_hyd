import logging
import os
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt

import config
import numpy as np

from three_comp_hyd.evolutionary_fitter.pygmo_three_comp_fitter import PyGMOThreeCompFitter
from three_comp_hyd.evolutionary_fitter.three_comp_tools import multi_to_single_objective
from handler.handler_base import HandlerBase


class PyGMOFittingReportCreator(HandlerBase):
    """
    Creates a full study report from athlete data
    """
    dir_path_addition = "PYGMO_FITTING_REPORT_CREATOR"

    def __init__(self):
        """
        """
        super().__init__()
        self._dir_path_addition = PyGMOFittingReportCreator.dir_path_addition

    def __create_and_save_stats(self, root, file, save_path):
        """
        One run with an algorithm configuration results in a couple of tank models - one for each island.
        This function saves stats and properties of these resulting best tank models. Comparison of such saved stat files
        allow to determine which algorithm configuration is most likely to deliver quality results.
        :param root:
        :param file:
        :param save_path:
        :return: resulting stats in a dict
        """
        # load the file and remove json extension
        data = self._load_json(os.path.join(root, file[:-5]))
        fronts = data["fronts"]

        # more details about the results of the last cycle
        last_cycle = list(fronts.values())[-1]

        configs, dists, ttes, recs = [], [], [], []
        for i, isl in enumerate(last_cycle.values()):
            # estimate the one that's closest to 0
            index_min = np.argmin(np.sqrt(np.array(isl["f"])[:, 0] ** 2 + np.array(isl["f"])[:, 1] ** 2))
            # use the multi to single function
            dists.append(multi_to_single_objective(isl["f"][index_min][0], isl["f"][index_min][1]))
            ttes.append(isl["f"][index_min][0])
            recs.append(isl["f"][index_min][1])
            configs.append(isl["x"][index_min])

        dists = np.array(dists)
        ttes = np.array(ttes)
        recs = np.array(recs)
        configs = np.array(configs)

        result = {}

        # migration type was added later.
        # Ensure compatibility with old logs
        if "migration_type" in data:
            result["migration_type"] = data["migration_type"]
        else:
            result["migration_type"] = "p2p" if "p2p" in file else "broadcast"

        # some files are too old to have this info
        if "time_elapsed" in data:
            result["time_elapsed"] = data["time_elapsed"]
        else:
            result["time_elapsed"] = -1

        # add early stopping info if available
        if "early_stopping" in data:
            early_stopping = data["early_stopping"]
            result.update({
                "early_stopping_triggered": early_stopping["activated"],
                "early_stopping_last_cycle": early_stopping["final_cycle"]
            })

        # Name of the pygmo UDP in use
        if "problem" in data:
            result.update({
                "problem": data["problem"],
            })
        else:
            result.update({
                "problem": "undefined"
            })

        # add the problem specific info if available
        if "additional_info" in data:
            result.update({
                "problem_additional_info": data["additional_info"]
            })

        # create the archipelago settings ID
        result["arch"] = "{}_{}_{}_{}".format(data["cycles"],
                                              data["pop_size"],
                                              data["islands"],
                                              result["migration_type"])

        # the final chunk of info to be added
        result.update({
            "root": root,
            "file": file.split(".")[0],
            "algo_dir": root.split("/")[-1],
            "num_islands": len(last_cycle.values()),
            "cycles": data["cycles"],
            "pop_size": data["pop_size"],
            "islands": data["islands"],

            "dists": list(dists),
            "dist_min": np.min(dists),
            "dist_mean": np.mean(dists),
            "dist_min_config": configs[np.argmin(dists)].tolist(),

            "tte_error_scores": list(ttes),
            "rec_error_scores": list(recs),
            "configs": configs.tolist(),
        })

        self._save_json(result, save_path, indent=True)
        return result

    def __plot_stats_overview_list(self, stats: list, save_path: str, min_only: bool = False):
        """
        Used to visualise mean and std variation of coefficients of tank models found with one algorithm run.
        A run with 7 islands results in 7 tank models with slightly varying parameters.
        This function creates a small overview plot for the distribution of result parameters
        :param stats: a list of stat files to be combined. A list with a single item results in an overview for that particular file
        :param save_path:
        :param min_only: whether the config with minimal distance should be the only one to be investigated
        """

        # collect all available info from stat files
        data = defaultdict(list)
        for stat in stats:
            # take only the best solution if min_only is true
            if min_only is True:
                data["dists"].append(stat["dist_min"])
                for k in range(8):
                    data["conf {}".format(k)].append(stat["dist_min_config"][k])
            # else, take all
            else:
                data["dists"] += stat["dists"]
                for k in range(8):
                    # ugly but necessary for the slicing
                    data["conf {}".format(k)] += np.array(stat["configs"])[:, k].tolist()

        fig = plt.figure(figsize=(13, 2))
        axes = []
        for j in range(1, 10):
            # create new axis and apply estimated limits
            axes.append(fig.add_subplot(1, 9, j))

        # plot the distribution for all available data points
        for i, (k, v) in enumerate(data.items()):
            np_v = np.array(v)
            mean = np.mean(np_v)
            std = np.std(np_v)
            axes[i].axvline(mean)
            axes[i].axvspan(mean - std, mean + std, alpha=0.2)
            axes[i].hist(v, bins=10)
            axes[i].yaxis.set_visible(False)
            axes[i].set_title(k)

        # rotate labels for readability
        for ax in axes:
            for label in ax.get_xticklabels():
                label.set_rotation(-45)
                label.set_ha('left')

        plt.suptitle("{}".format(os.path.basename(save_path)))
        plt.tight_layout()
        plt.subplots_adjust(top=0.7)

        # save path for overview plot
        logging.info("saving overview plot to: {}.png".format(save_path))
        plt.savefig("{}.png".format(save_path))
        plt.close(fig)

    def __plot_category_comparison(self, overview_dict: dict, save_path: os.path):
        """
        creates an overview plot that plots each item of the overview dict in one subplot.
        Subplots are stacked vertically and share the x axis
        :param overview_dict:
        :param save_path:
        :return:
        """
        fig = plt.figure(figsize=(10, 10))
        axes = []

        # plot everything with a shared x-axis
        for i, (category, values) in enumerate(overview_dict.items()):
            # create new axis and apply estimated limits
            if len(axes) > 0:
                axes.append(fig.add_subplot(len(overview_dict), 1, i + 1, sharex=axes[0]))
            else:
                axes.append(fig.add_subplot(len(overview_dict), 1, i + 1))

            # transform list of values into an array for easier estimations
            np_v = np.array(values)
            mean = np.mean(np_v)
            std = np.std(np_v)
            # plot mean and std deviation
            axes[-1].axvline(mean)
            axes[-1].axvspan(mean - std, mean + std, alpha=0.2)
            axes[-1].hist(np_v, bins=10)
            axes[-1].yaxis.set_visible(False)
            axes[-1].set_title("{} ({})".format(category, len(np_v)))

        plt.suptitle("{}".format(os.path.basename(save_path)))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # save path for overview plot
        logging.info("saving category overview plot to: {}.png".format(save_path))
        plt.savefig("{}.png".format(save_path))
        plt.close(fig)

    def __plot_best_front(self, root, file, save_path):
        """
        Plots evolving pareto fronts for every single cycle
        :param root:
        :param file:
        :param save_path:
        """
        # load the file and remove json extension
        data = self._load_json(os.path.join(root, file[:-5]))
        fronts = data["fronts"]

        # plot  evolving fronts if requested
        logging.info("Saving best front plot to: {}".format(save_path))

        v = list(fronts.values())[-1]

        fig = plt.figure(figsize=(8, 5))
        axes = fig.add_subplot(1, 1, 1)

        # store the best solution details in here
        found_best = None

        # compare all island non-dominated front lines
        for j, i_f in enumerate(v.values()):
            # sort to prevent the plotting of loops in the step plot
            sorted_by_x = sorted(i_f["f"], key=lambda tup: tup[0])

            # plot the sorted non-dominated front as a step function
            x = [p[0] for p in sorted_by_x]
            y = [p[1] for p in sorted_by_x]

            # find the best model by using the distance to 0
            dist = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
            index = int(np.argmin(dist))

            # keep track of best solution and its details
            if found_best is None:
                found_best = (index, dist[index], x, y)
            else:
                if found_best[1] > dist[index]:
                    found_best = (index, dist[index], x, y)

        x, y = found_best[2], found_best[3]
        axes.step(x, y, where='post', label="Pareto front")
        axes.scatter(x[found_best[0]], y[found_best[0]],
                     label="best configuration",
                     marker="d",
                     s=80)

        axes.legend()
        axes.set_xlabel("expenditure NRMSE")
        axes.set_ylabel("recovery NRMSE")

        plt.savefig(os.path.join(save_path, "best_front_{}".format(len(fronts.values()))))
        plt.close(fig)

    def __plot_evolving_fronts(self, root, file, save_path):
        """
        Plots evolving pareto fronts for every single cycle
        :param root:
        :param file:
        :param save_path:
        """
        # load the file and remove json extension
        data = self._load_json(os.path.join(root, file[:-5]))
        fronts = data["fronts"]

        # plot  evolving fronts if requested
        logging.info("Saving evolving fronts plots to: {}".format(save_path))
        for i, v in enumerate(fronts.values()):

            # skip plots that already exist
            plot_path = os.path.join(save_path, "fronts_{}".format(i))
            if os.path.exists(plot_path):
                continue

            fig = plt.figure(figsize=(10, 6))
            axes = fig.add_subplot(1, 1, 1)

            # plot all island non-dominated front lines
            for j, i_f in enumerate(v.values()):
                # sort to prevent the plotting of loops in the step plot
                sorted_by_x = sorted(i_f["f"], key=lambda tup: tup[0])

                # plot the sorted non-dominated front as a step function
                x = [p[0] for p in sorted_by_x]
                y = [p[1] for p in sorted_by_x]
                axes.step(x, y, where='post', label="island {}".format(j))

                # find the best model by using the distance to 0
                index_min = int(np.argmin(np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)))
                axes.scatter(x[index_min], y[index_min],
                             label="best island {}".format(j),
                             marker="d",
                             s=80)

            axes.set_title("Cycle {}".format(i))
            axes.legend()
            axes.set_xlabel("TTE_nrmse")
            axes.set_ylabel("rec_nrmse")

            plt.savefig(os.path.join(save_path, "fronts_{}".format(i)))
            plt.close(fig)

    def __create_overview_stats_and_plots(self, path: os.path, all_stats: dict, plot_overviews: bool = False):
        """
        :param path: the path to store all overview plot and json files to
        :param all_stats: a dict that's created by the __save_stats function
        :param plot_overviews: whether overview plots should be created
        :return:
        """

        # collect info from all the configs for additional overviews
        all_path = os.path.join(path, "all_configs")

        # collections of stats
        all_best_confs = defaultdict(list)
        all_best_dists = defaultdict(list)
        all_best_mean_dists = defaultdict(list)

        # best performing solutions are determined
        bests = {}
        # categories are merged into overview collection
        merged = []
        best_check_categories = ["dist_min", "dist_mean"]
        # walk over all collected stats
        for k, v in all_stats.items():
            for x in v:
                # keep track of the best
                for cat in best_check_categories:
                    if cat not in bests or bests[cat]["val"] > x[cat]:
                        bests[cat] = {
                            "val": x[cat],
                            "arch": x["arch"],
                            "algo": x["algo_dir"],
                            "problem": x["problem"],
                            "champion_dist": x["dist_min"],
                            "champion": x["dist_min_config"]}

                # store all best configurations for stats categories
                all_best_confs[k].append(x["dist_min_config"])
                # category overview regarding minimal distance and minimal mean distance
                all_best_dists[k].append(x["dist_min"])
                all_best_mean_dists[k].append(x["dist_mean"])

            # all stats merged into one overview
            merged += v

        for k in all_stats.keys():
            np_best_dists = np.array(all_best_dists[k])

            b_config = all_best_confs[k][int(np.argmin(np_best_dists))]  # the configuration of the best
            b_config = [round(x, 2) for x in b_config]
            s_config = ""
            for b in b_config:
                s_config += "{:.2f} & ".format(b)
            s_config = s_config[:-2]

            bests.update({"{}({})".format(k, len(np_best_dists)): {
                "mean_best_dist": np.mean(np_best_dists),
                "std_best_dist": np.std(np_best_dists),
                "min_best_dist": np.min(np_best_dists),
                "max_best_dist": np.max(np_best_dists),
                "latex": "{:.4f} & {:.4f} & {:.4f} & {} \\\\ % {}".format(round(np.min(np_best_dists), 4),
                                                                          round(np.mean(np_best_dists), 4),
                                                                          round(np.max(np_best_dists), 4),
                                                                          s_config,
                                                                          len(np_best_dists))
            }
            })

        # best algorithm results overview
        stats_path = os.path.join(path, "bests")
        # if not os.path.exists("{}.json".format(stats_path)):
        self._save_json(data=bests, file=stats_path)

        # collection of best config of each iteration
        # if not os.path.exists("{}.json".format(all_path)):
        self._save_json(data=all_best_confs, file=all_path)

        if plot_overviews is True:
            # category overview plots for min dist and mean min dist
            dist_path = os.path.join(path, "best_dist")
            # if not os.path.exists("{}.png".format(dist_path)):
            self.__plot_category_comparison(overview_dict=all_best_dists, save_path=dist_path)

            dist_mean_path = os.path.join(path, "best_mean_dist")
            # if not os.path.exists("{}.png".format(dist_mean_path)):
            self.__plot_category_comparison(overview_dict=all_best_dists, save_path=dist_mean_path)

            # combined configs overview plots
            # means and std deviations for dist and every config parameter
            for k, v in all_stats.items():
                overview_plot_path = os.path.join(path, "champ_config_params_{}".format(k))
                if not os.path.exists("{}.png".format(overview_plot_path)):
                    self.__plot_stats_overview_list(stats=v, save_path=overview_plot_path, min_only=True)

        # return the merged stat files
        return merged

    def analysis_all_multi_objective_archipelagos(self,
                                                  clear_all: bool = False,
                                                  plot_overviews: bool = False,
                                                  plot_evolving_fronts: bool = False,
                                                  plot_best_front: bool = False):
        """
        walks through all multi-objective archipelago results and plots some overviews
        :return:
        """

        # get setup directories
        root, setups, _ = next(os.walk(os.path.join(config.paths["data_storage"],
                                                    PyGMOThreeCompFitter.dir_path_addition)))
        # get full path from fitter directory to file
        dirs_ind = root.rfind(PyGMOThreeCompFitter.dir_path_addition) + len(
            PyGMOThreeCompFitter.dir_path_addition) + 1

        # W' and CP configuration
        for setup in setups:
            logging.info("start setup {}".format(setup))
            setup_root, problems, _ = next(os.walk(os.path.join(root, setup)))

            # determine new directory to store results into
            dir_path = os.path.join(config.paths["data_storage"],
                                    PyGMOFittingReportCreator.dir_path_addition,
                                    setup_root[dirs_ind:])

            # clear whole directory if desired
            if os.path.exists(dir_path) and clear_all is True:
                shutil.rmtree(dir_path)

            # keep track of results for overview plots
            setup_all_stats = defaultdict(list)

            # MultiObjective or SingleObjective
            for problem in problems:
                logging.info("  start problem {}".format(problem))
                problem_root, algorithms, _ = next(os.walk(os.path.join(setup_root, problem)))

                # keep track of results for overview plots
                problem_all_stats = defaultdict(list)

                # walk over all algorithm settings
                for algorithm in algorithms:
                    logging.info("      start algorithm {}".format(algorithm))
                    algo_root, _, files = next(os.walk((os.path.join(problem_root, algorithm))))

                    # keep track of best archipelago configuration
                    algorithm_all_stats = defaultdict(list)

                    # walk over all runs with a particular algorithm setting
                    for file in files:
                        if file.split('.')[-1] == "json":
                            logging.info("          start archipelago {}".format(file))

                            # determine new directory to store results into
                            dir_path = os.path.join(config.paths["data_storage"],
                                                    PyGMOFittingReportCreator.dir_path_addition,
                                                    algo_root[dirs_ind:],
                                                    file.split(".")[0])

                            # save stats
                            stats_path = os.path.join(dir_path, "stats")

                            # get values for the plot and estimate/load stats
                            if not os.path.exists("{}.json".format(stats_path)):
                                stats = self.__create_and_save_stats(root=algo_root, file=file, save_path=stats_path)
                            else:
                                stats = self._load_json(stats_path)

                            # now for the overview plot
                            overview_plot_path = os.path.join(dir_path, "overview_plot_{}".format(stats["file"]))
                            if plot_overviews is True and not os.path.exists("{}.png".format(overview_plot_path)):
                                self.__plot_stats_overview_list(stats=[stats], save_path=overview_plot_path)

                            # this plots the evolving fronts and causes a lot of computation
                            if plot_evolving_fronts is True:
                                self.__plot_evolving_fronts(root=algo_root, file=file, save_path=dir_path)
                            if plot_best_front is True:
                                self.__plot_best_front(root=algo_root, file=file, save_path=dir_path)

                            # add the created stats file to the all stats overview
                            algorithm_all_stats[stats["arch"]].append(stats)

                    # determine new directory to store results into
                    path = os.path.join(config.paths["data_storage"],
                                        PyGMOFittingReportCreator.dir_path_addition,
                                        problem_root[dirs_ind:],
                                        algorithm)
                    # make use of overview creation function and obtain merged algorithm stats
                    problem_all_stats[algorithm] += self.__create_overview_stats_and_plots(path=path,
                                                                                           all_stats=algorithm_all_stats,
                                                                                           plot_overviews=plot_overviews)

                # again, determine new directory to store results into
                path = os.path.join(config.paths["data_storage"],
                                    PyGMOFittingReportCreator.dir_path_addition,
                                    setup_root[dirs_ind:],
                                    problem)
                self.__create_overview_stats_and_plots(path, problem_all_stats)
                setup_all_stats[problem] += self.__create_overview_stats_and_plots(path=path,
                                                                                   all_stats=problem_all_stats,
                                                                                   plot_overviews=plot_overviews)

            # and again, determine new directory to store results into
            path = os.path.join(config.paths["data_storage"],
                                PyGMOFittingReportCreator.dir_path_addition,
                                setup_root[dirs_ind:])
            self.__create_overview_stats_and_plots(path=path,
                                                   all_stats=setup_all_stats,
                                                   plot_overviews=plot_overviews)

    def create_latex_table(self,
                           cp: int = 248,
                           w_p: int = 18200,
                           tte_id: str = "setting_0",
                           rec_id: str = "caen"):
        """
        summarizes report findings into the latex table for GECCO publication
        :return:
        """

        # files will be in random order, so set up a dict to sort the results
        latex_table = {}
        gens = [10, 20, 30]
        cycles = [10, 40, 80]
        pops = [32, 64]
        islands = [7, 14, 21]
        for g in gens:
            latex_table.update({g: {}})
            for c in cycles:
                latex_table[g].update({c: {}})
                for p in pops:
                    latex_table[g][c].update({p: {}})
                    for i in islands:
                        latex_table[g][c][p][i] = "& & & & & & & & \\\\"

        # get setup directories
        root, setups, _ = next(os.walk(os.path.join(config.paths["data_storage"],
                                                    PyGMOFittingReportCreator.dir_path_addition)))

        # W' and CP configuration
        for setup in setups:
            if "{}_{}_{}_{}".format(int(w_p), int(cp), tte_id, rec_id) in str(setup):

                problem = "MultiObjectiveThreeComp_False"
                if config.three_comp_phi_constraint is True:
                    problem = "MultiObjectiveThreeComp_True"

                logging.info("start setup {}".format(setup))
                logging.info("start problem {}".format(problem))
                problem_root, algorithms, _ = next(os.walk(os.path.join(root, setup, problem)))

                # MOEAD and settings
                for algorithm in algorithms:
                    bests_json = self._load_json(os.path.join(problem_root, algorithm, "bests"))
                    bests_json.pop("dist_min")
                    bests_json.pop("dist_mean")

                    # get generations from algorithm description
                    start_eg = algorithm.find("_[") + 2
                    stop_eg = algorithm.find(", '")
                    eg = algorithm[start_eg:stop_eg]

                    # now walk through the "10_32_21_p2p(4)" entries
                    for k, v in bests_json.items():
                        # get cycles population and islands
                        ec, ep, ei, _ = k.split("_")
                        try:
                            latex_table[int(eg)][int(ec)][int(ep)][int(ei)] = v["latex"]
                        except KeyError:
                            continue

        # print in the table format
        for g in gens:
            for c in cycles:
                for p in pops:
                    for i in islands:
                        print("{} & {} & {} & {} & {}".format(g, c, p, i, latex_table[g][c][p][i]))

    def write_data_report(self, clear_all: bool = False, plot_best_front: bool = False, plot_overviews: bool = False):
        """
        puts out all what your heart desires regarding pygmo fittings, algorithm performances and
        fitting results into one overview
        :param clear_all: removes everything the report creator wrote before creating new docs
        :param plot_best_front: plots best pareto front for each run
        :param plot_overviews: plots extensive overviews over categories, problems, algorithms, and individual
         configuration distributions
        :return:
        """

        self.analysis_all_multi_objective_archipelagos(clear_all=clear_all,
                                                       plot_evolving_fronts=False,
                                                       plot_best_front=plot_best_front,
                                                       plot_overviews=plot_overviews)
