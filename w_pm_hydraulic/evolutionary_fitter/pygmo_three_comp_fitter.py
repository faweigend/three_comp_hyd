import datetime
import itertools
import json
import logging
import os
import time

import pygmo.core as pygcore
import numpy as np

from w_pm_hydraulic import three_comp_config
from w_pm_hydraulic.data_structure.constant_effort_measures import ConstantEffortMeasures
from w_pm_hydraulic.data_structure.recovery_measures import RecoveryMeasures

from w_pm_hydraulic.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP


class PyGMOThreeCompFitter:
    """
    fits the three component hydraulic model to exercise data
    """

    # this handler's addition to the directory path
    dir_path_addition = "THREE_COMP_PYGMO_FIT"

    def __init__(self, ttes: ConstantEffortMeasures, recovery_measures: RecoveryMeasures, log_level: int = 0):
        """
        constructor with basic parameters
        :param ttes: traditional ttes to fit expenditure to
        :param recovery_measures: recovery trials to estimate fitness with
        :param log_level: determines how much information about the fittings will be stored
        (0 basic, >0 all fronts and migration logs)
        """
        self._dir_path_addition = PyGMOThreeCompFitter.dir_path_addition

        self.__hz = 1
        # (0 basic, >0 all fronts and migration logs)
        self.__log_level = log_level
        # counter for early stopping. If 0, no early stopping is used
        self.__early_stopping = 10

        self.__ttes = ttes
        self.__recovery_measures = recovery_measures

    @property
    def early_stopping(self):
        """simple getter for early stopping"""
        return self.__early_stopping

    @early_stopping.setter
    def early_stopping(self, early_stopping):
        """ simple setter for early stopping"""
        self.__early_stopping = early_stopping

    def __save_fitting_result(self, data, comb, algo, topo):
        """
        standardised storage of grid search results
        :param data:
        :param comb:
        """

        # save algorithm parameters
        data["algorithm"] = {
            "name": algo.get_name(),
            "gen": comb[0],
            "weight_generation": comb[1],
            "decomposition": comb[2],
            "neighbours": comb[3],
            "cr": comb[4],
            "f": comb[5],
            "eta_m": comb[6],
            "realb": comb[7],
            "limit": comb[8],
            "preserve_diversity": comb[9]
        }
        # save topology info
        data["topology"] = {
            "name": topo.get_name()
        }

        # create a folder for algorithm and topology setup
        folder_str = "{}_{}/{}/{}_{}_{}".format(self.__ttes.name, self.__recovery_measures.name,
                                                data["problem"],
                                                data["algorithm"]["name"][:5],
                                                comb,
                                                data["topology"]["name"])
        # create a filename from archipelago setup
        file_str = "{}_{}_{}_{}_{}".format(data["cycles"],
                                           data["pop_size"],
                                           data["islands"],
                                           data["migration_type"],
                                           data["date_time"])
        total_str = "{}/{}".format(folder_str, file_str)

        # write results to file
        self._save_json(data, total_str, indent=False)

        # finish info
        logging.info("Grid search finished comb: {}".format(comb))

    def fit_with_moead(self, gen=1, weight_generation="grid", decomposition="tchebycheff",
                       neighbours=20, cr=1.0, f=0.5, eta_ms=20, realb=0.9, limit=2,
                       preserve_diversity=True, cycles=80, pop_size=64, islands=7,
                       migration_type=0):
        """
        defaults from web:
        gen = 1, weight_generation = 'grid', decomposition = 'tchebycheff', neighbours = 20, CR = 1, F = 0.5,
        eta_m = 20, realb = 0.9, limit = 2, preserve_diversity = true, seed = random

        :param gen:
        :param weight_generation:
        :param decomposition:
        :param neighbours:
        :param cr:
        :param f:
        :param eta_ms:
        :param realb:
        :param limit:
        :param preserve_diversity:
        :param cycles:
        :param pop_size:
        :param islands:
        :param migration_type:
        :return:
        """

        logging.info("Fitting start MOEAD with: {}".format([gen, weight_generation, decomposition, neighbours,
                                                            cr, f, eta_ms, realb, limit, preserve_diversity, cycles,
                                                            pop_size, islands, migration_type]))
        # set up the algorithm
        algo = pygcore.algorithm(pygcore.moead(gen=gen,
                                               weight_generation=weight_generation,
                                               decomposition=decomposition,
                                               neighbours=neighbours,
                                               CR=cr,
                                               F=f,
                                               eta_m=eta_ms,
                                               realb=realb,
                                               limit=limit,
                                               preserve_diversity=preserve_diversity))

        # topology defines how islands are connected
        topo = pygcore.topology(pygcore.fully_connected())

        if migration_type == 0:
            mt = pygcore.migration_type.p2p
        else:
            mt = pygcore.migration_type.broadcast

        # run the fitting using all established parameters
        data = self.fitting_multi_objective_archipelago(cycles=cycles,
                                                        pop_size=pop_size,
                                                        islands=islands,
                                                        algorithm=algo,
                                                        topology=topo,
                                                        migration_type=mt)

        # save what you got
        self.__save_fitting_result(data=data,
                                   comb=[gen, weight_generation, decomposition, neighbours,
                                         cr, f, eta_ms, realb, limit, preserve_diversity],
                                   algo=algo,
                                   topo=topo)

    def grid_search_algorithm_moead(self, islands: int):
        """
        search through possible algorithm parameter variations
        :param islands: number of isolated islands (threads)
        """
        # define migration type p2p(0) or broadcast(1)
        migration_type = [0]

        # number of cycles through number of gen steps
        cycles = [10, 40, 80]

        # population size
        pop_size = [32, 64]

        # gen (int) – number of generations
        gens = [10, 20, 30]

        # weight_generation (str) – method used to generate the weights, one of “grid”, “low discrepancy” or “random”
        weight_gen = ["grid"]  # , "low discrepancy", "random"]

        # decomposition (str) – method used to decompose the objectives, one of “tchebycheff”, “weighted” or “bi”
        decompositions = ["tchebycheff"]  # , "weighted", "bi"]

        # neighbours (int) – size of the weight’s neighborhood
        neighbours = [20]

        # CR (float) – crossover parameter in the Differential Evolution operator
        crs = [1.0]

        # F (float) – parameter for the Differential Evolution operator
        fs = [0.5]

        # eta_m (float) – distribution index used by the polynomial mutation
        eta_ms = [20]

        # realb (float) – chance that the neighbourhood is considered at each generation,
        # rather than the whole population (only if preserve_diversity is true)
        realb = [0.9]

        # limit (int) – maximum number of copies reinserted in the population (only if m_preserve_diversity is true)
        limit = [2]

        # preserve_diversity (bool) – when true activates diversity preservation mechanisms
        preserve_diversity = [True]

        # seed (int) – seed used by the internal random number generator (default is random)

        # create a list of all possible combinations
        combs = list(itertools.product(gens,
                                       weight_gen,
                                       decompositions,
                                       neighbours,
                                       crs,
                                       fs,
                                       eta_ms,
                                       realb,
                                       limit,
                                       preserve_diversity,
                                       migration_type,
                                       cycles,
                                       pop_size))

        # walk through all possible combinations
        for comb in combs:
            logging.info("Grid search start MOEAD with comb: {}".format(comb))

            self.fit_with_moead(gen=comb[0],
                                weight_generation=comb[1],
                                decomposition=comb[2],
                                neighbours=comb[3],
                                cr=comb[4],
                                f=comb[5],
                                eta_ms=comb[6],
                                realb=comb[7],
                                limit=comb[8],
                                preserve_diversity=comb[9],
                                migration_type=comb[10],
                                cycles=comb[11],
                                pop_size=comb[12],
                                islands=islands)

    def fitting_multi_objective_archipelago(self, cycles: int, pop_size: int, islands: int,
                                            migration_type: pygcore.migration_type,
                                            algorithm: pygcore.algorithm,
                                            topology: pygcore.topology):
        """
        Run a fitting that makes use of the parallelised islands.
        It saves the evolved fronts and corresponding configurations
        :param cycles: number of cycles through number of gen steps
        :param pop_size: population size
        :param islands: number of isolated islands (threads)
        :param migration_type: define migration type (broadcast or p2p)
        :param algorithm: define the algorithm in use
        :param topology: topology defines how islands are connected
        """

        # the user defined problem
        udp = MultiObjectiveThreeCompUDP(ttes=self.__ttes,
                                         recovery_measures=self.__recovery_measures)

        # setup data do save
        data = {
            "function": "fitting_multi_objective_archipelago",
            "problem": udp.get_name(),
            "additional_info": udp.get_additional_info(),
            "cycles": cycles,
            "pop_size": pop_size,
            "islands": islands,
            "migration_type": migration_type.name,
            "early_stopping": self.__early_stopping
        }
        logging.info("Start fitting with settings: {}".format(data))

        # islands are threads, an archipelago is a collection of islands
        archi = pygcore.archipelago(t=topology)
        archi.set_migration_type(migration_type)

        # add the desired number of islands
        for i in range(islands):
            # create empty population
            pop = pygcore.population(prob=udp)
            # pushback educated initial guesses until population size is reached
            for _ in range(pop_size):
                pop.push_back(udp.create_educated_initial_guess())
            # create island with population and algorithm
            isl = pygcore.island(algo=algorithm, pop=pop)
            # add island to archipelago
            archi.push_back(isl)

        # non-dominating fronts will be stored in here
        fronts = {}

        def get_parteo_fronts():
            """extracts best non-dominated fronts from
            all islands and returns overview in a dict"""
            isle_fronts = {}
            for j, isle in enumerate(archi):
                # get island population
                pop = isle.get_population()
                # sort to get non-dominated front
                ndf, dl, dc, ndl = pygcore.fast_non_dominated_sorting(pop.get_f())
                # get the first (i.e., best) non-dominated front
                front = ndf[0]
                # append to list of fronts (one for each island)
                isle_fronts[j] = {"f": pop.get_f()[front].tolist(),
                                  "x": pop.get_x()[front].tolist()}

            # keep track of front developments
            return isle_fronts

        # keep track of elapsed time
        time_start = time.time()

        # variables for early stopping
        last_best = None
        early_stop_count = 0
        final_cycle = 0

        # after an evolve of archipelago migration happens
        for i in range(cycles):
            logging.info("Archipelago evolve {} start".format(i))

            # run evolution for n=1
            archi.evolve()
            # wait for all islands to finish
            archi.wait()

            logging.info("Archipelago evolve {} done".format(i))

            # get current island pareto fronts
            int_fronts = get_parteo_fronts()

            # log fronts for every step if the level requires it
            if self.__log_level > 0:
                fronts[i] = int_fronts

            # keep track of best observed fitness
            best = None
            # check all island fronts
            for k, v in int_fronts.items():
                x = [p[0] for p in v["f"]]
                y = [p[1] for p in v["f"]]
                # find the best model by using the distance to 0
                dist = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
                index = int(np.argmin(dist))
                # keep track of best
                if best is None or dist[index] < best:
                    best = dist[index]
            logging.info("best distance {}".format(best))

            # apply early stopping if best distance did not improve
            if self.__early_stopping > 0:
                # keep track of cycles without improvement
                if last_best is None or last_best > best:
                    last_best = best
                    early_stop_count = 0
                elif last_best <= best:
                    early_stop_count += 1
                # halt computation if no improvement was observed for x cycles
                if early_stop_count > self.__early_stopping:
                    logging.info("early stopping triggered")
                    break

                logging.info("early stopping count at {}".format(early_stop_count))

            # update tracking variables
            final_cycle = i + 1

        # store elapsed time
        data["time_elapsed"] = time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))

        # store exact time point
        data["date_time"] = str(datetime.datetime.now())

        # TODO: store the ultimate champion. Early stopping indicates that a better solution was found in the past
        # early stopping params
        data["early_stopping"] = {
            "activated": final_cycle != cycles,
            "final_cycle": final_cycle,
            "early_stopping_threshold": self.__early_stopping
        }

        # add logs and results according to log level
        if self.__log_level > 0:
            data["fronts"] = fronts
            # keep migration log
            data["migration_log"] = str(archi.get_migration_log())
        else:
            data["fronts"] = {"last_cycle": get_parteo_fronts()}

        return data

    def _save_json(self, data: dict, file: str, indent: bool = True):
        """
        saves metadata into standard json format
        :param data: data to store into json
        :param file: filename
        """
        path = os.path.join(self._get_dir_path(), file)
        # create directories if they don't exist
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        try:
            logging.info("Saving json to: {}".format(path))
            with open("{}.json".format(path), 'w') as fp:
                if indent is True:
                    json.dump(data, fp, indent=4)
                else:
                    json.dump(data, fp)
        except TypeError:
            logging.warning(data)
            logging.warning("Metadata is not in a format that can be converted to json: {}".format(path))

    def _load_json(self, file: str):
        """
        load and parse json into dict
        :param file: filename
        :return: dict in structure of read json
        """
        path = os.path.join(self._get_dir_path(), "{}.json".format(file))
        with open(path, 'rb') as fp:
            return json.load(fp)

    def _get_dir_path(self):
        """
        Simple getter for the data storage path
        :return: path
        """
        return os.path.join(three_comp_config.paths["data_storage"], self._dir_path_addition)
