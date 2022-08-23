class SimpleRecMeasures:
    """
    Used to store W' recovery ratios in a consistent way for evolutionary fitting estimations. Parameters are named
    according to the WB1->RB->WB2 protocol
    """

    def __init__(self, name: str):
        """
        constructor
        :param name: name identifier
        """
        self.__name = name
        self.__measures = []

    def add_measure(self, p_work: float, p_rec: float, t_rec: int, recovery_percent: float):
        """
        adds one observationt to internal list
        :param p_work: intensity that lead to exhaustion
        :param p_rec: recovery intensity
        :param t_rec: recovery time
        :param recovery_percent: recovery in percent
        """
        self.__measures.append((p_work, p_rec, t_rec, recovery_percent))

    def __str__(self):
        """
        print function
        :return: stored values as a stringified dict
        """
        return "{} : \n {}".format(self.__name, self.__measures)

    def __len__(self):
        """
        :return: length definition
        """
        return len(self.__measures)

    def iterate_measures(self):
        """
        iterates through all measures and returns the essential values for the objective function
        :return: p_work, p_rec, t_rec, expected
        """
        for p_work, p_rec, t_rec, expected in list(self.__measures):
            yield p_work, p_rec, t_rec, expected

    @property
    def name(self):
        """
        :return: the defined name
        """
        return self.__name

    def get_all_wb_rb_combinations(self):
        """
        returns all combinations of power that lead to exhaustion (WB) and recovery intensity (RB) that
        this recovery measure storage contains
        :return:
        """
        combs = []
        for values in list(self.__measures):
            comb = (values[0], values[1])
            if comb not in combs:
                combs.append(comb)
        return combs

    def get_all_obs_for_wb_rb_combination(self, p_work, p_rec):
        """
        Get all observations for a p_work and p_rec combination.
        returns times and ratios in two lists.
        :param p_work: power that lead to exhaustion (WB)
        :param p_rec: recovery intensity (RB)
        :return: times, ratios
        """
        times, ratios = [], []
        for values in list(self.__measures):
            if values[0] == p_work and values[1] == p_rec:
                times.append(values[2])
                ratios.append(values[3])
        return times, ratios

    def get_max_t_rec(self):
        """
        :return: maximal recovery time in stored trials
        """
        max_t = 0
        for values in list(self.__measures):
            max_t = max(max_t, values[2])
        return max_t
