class SimpleTTEMeasures:
    """
    Collection of TTE and constant power pairs for evolutionary fitting
    """

    def __init__(self, name: str):
        """
        constructor
        :param name: given name identifier
        """
        self.__pairs = []
        self.__name = name

    def __len__(self):
        """
        :return: length definition
        """
        return len(self.__pairs)

    def __str__(self):
        """
        print function
        :return: stored values as a stringified dict
        """
        return "{} : \n {}".format(self.__name, self.__pairs)

    @property
    def name(self):
        """
        :return: name
        """
        return self.__name

    @property
    def pairs(self):
        """
        :return: stored time values as list
        """
        return self.__pairs

    def add_pair(self, t: float, p: float):
        """
        adds a (time,constant power) pair to internal data
        :param t: TTE
        :param p: constant power
        """
        self.__pairs.append((t, p))

    def iterate_pairs(self):
        """
        generator for time/measure pairs
        """
        for t, p in self.__pairs:
            yield t, p
