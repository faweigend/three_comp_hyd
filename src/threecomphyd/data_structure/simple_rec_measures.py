class SimpleRecMeasures:
    """
    Used store W' recovery ratios in a consistent way for evolutionary fitting estimations
    """

    def __init__(self, name: str):
        """
        constructor
        :param name: name identifier
        """
        self.__name = name
        self.__measures = []

    def add_measure(self, p_power: float, r_power: float, r_time: int, recovery_percent: float):
        """
        adds one observationt to internal list
        :param p_power: intensity that lead to exhaustion
        :param r_power: recovery intensity
        :param r_time: recovery time
        :param recovery_percent: recovery in percent
        """
        self.__measures.append((p_power, r_power, r_time, recovery_percent))

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
        for p_exp, p_rec, t_rec, expected in list(self.__measures):
            yield p_exp, p_rec, t_rec, expected

    @property
    def name(self):
        """
        :return: the defined name
        """
        return self.__name
