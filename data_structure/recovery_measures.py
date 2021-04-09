class RecoveryMeasures:
    """
    Used to consistently store W' recovery ratios
    """

    def __init__(self, name: str):
        """
        constructor
        """
        self.__name = name
        self.__measures = dict()

    def add_measure(self, p_power: float, r_power: float, r_time: int, recovery_percent: float, p_time: int = None):
        """
        :param p_power:
        :param p_time:
        :param r_power:
        :param r_time:
        :param recovery_percent:
        :return:
        """
        new_test = {
            'p_power': p_power,
            'p_time': p_time,
            'r_power': r_power,
            'r_time': r_time,
            'recovery%': recovery_percent
        }
        self.__measures["t{}".format(len(self) + 1)] = new_test

    def __len__(self):
        """length definition"""
        return len(self.__measures)

    def get_max_r_time(self):
        """
        :return: maximal recovery time in stored trials
        """
        max_t = 0
        for values in list(self.get_all_measures()):
            max_t = max(max_t, values["r_time"])
        return max_t

    def iterate_all_measures(self):
        """
        iterates through all measures and returns the essential values for the objective function
        :return: p_exp, p_rec, t_rec, expected
        """
        for measure in list(self.__measures.values()):
            p_exp = measure['p_power']
            p_rec = measure['r_power']
            t_rec = measure['r_time']
            expected = measure['recovery%'] * 0.01
            yield p_exp, p_rec, t_rec, expected

    def get_all_p_exp_p_rec_combinations(self):
        """
        returns all combinations of power that lead to exhaustion (p_exp) and recovery intensity (p_rec) that
        this recovery measure storage contains
        :return:
        """
        combs = []
        for values in list(self.get_all_measures()):
            comb = (values["p_power"], values["r_power"])
            if comb not in combs:
                combs.append(comb)
        return combs

    def get_all_obs_for_p_exp_p_rec_combination(self, p_exp, p_rec):
        """
        Get all observations for a p_exp and p_rec combination.
        returns times and ratios in two lists.
        :param p_exp: power that lead to exhaustion (p_exp)
        :param p_rec: recovery intensity (p_rec)
        :return: times, ratios
        """
        times, ratios = [], []
        for values in list(self.get_all_measures()):
            if values["p_power"] == p_exp and values["r_power"] == p_rec:
                times.append(values["r_time"])
                ratios.append(values["recovery%"])
        return times, ratios

    def get_all_measures(self):
        """
        :return: all stored tests without keys (descriptors)
        """
        return self.__measures.values()

    def get_measure(self, i: int):
        """
        :param i: index of test/measure to return
        :return: specific test with index i
        """
        return self.__measures["t{}".format(i)]

    @property
    def name(self):
        """return the defined name"""
        return self.__name
