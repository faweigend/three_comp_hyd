from agents.hydraulic_agent_basis import HydraulicAgentBasis


class TwoCompHydAgent(HydraulicAgentBasis):

    def __init__(self, hz, w_p, m_o, phi):
        """
        :param hz: calculations per second
        :param w_p: cross sectional area of W'
        :param m_o: maximal flow from O to w'
        :param phi: phi (distance bottom -> bottom O)
        """
        super().__init__(hz=hz)

        # and the small volume of
        # the narrow tube B was neglected in this study

        # constants
        self.__phi = phi

        # vessel areas
        self.__w_p = w_p  # area of vessel W'

        # max flows
        self.__m_o = m_o  # max flow from O to P (max aerobic energy consumption i.e. VO2max)

        self.__w_m = 100  # max instantaneous power

        # variable parameters
        self.__h = 0  # state of depletion of vessel W'
        self.__p_o = 0  # flow through R1 (oxygen pipe)

    @property
    def w_p(self):
        """:return cross sectional area of W'"""
        return self.__w_p

    @property
    def phi(self):
        """:return phi (distance bottom -> bottom O)"""
        return self.__phi

    def get_h(self):
        """:return state of depletion of vessel P"""
        return self.__h

    def get_p_o(self):
        """:return flow through R1"""
        return self.__p_o

    def get_m_o(self):
        """:return max flow through R1"""
        return self.__m_o

    def _estimate_possible_power_output(self):
        """
        Update internal capacity estimations by one step.
        :return: the amount of power that the athlete was able to put out
        """
        p = self._pow

        # determine oxygen energy flow (R1)
        # level P above pipe exit. Scale contribution according to h level
        if 0 < self.__h <= (1.0 - self.__phi):
            # contribution through R1 scales with maximal flow capacity
            self.__p_o = self.__m_o * (self.__h / (1.0 - self.__phi))
        # at maximum rate because level h of P is below pipe exit of O
        elif (1.0 - self.__phi) < self.__h <= 1.0:
            # above aerobic threshold (?) max contribution R1 = m_o
            self.__p_o = self.__m_o

        # the change on level in W' is determined by flow p_o
        self.__h += ((p - self.__p_o) / self.__w_p) / self._hz

        # also W' cannot be fuller than full
        if self.__h < 0:
            self.__h = 0
        elif self.__h > 1:
            self.__h = 1

        return self._pow

    def is_exhausted(self):
        """
        exhaustion is reached when level in AnF cannot sustain power demand
        :return: simply returns the exhausted flag
        """
        return self.__h == 1.0

    def is_recovered(self):
        """
        recovery is complete when W' is full again
        :return: simply returns the recovered flag
        """
        return self.__h == 0.0

    def is_equilibrium(self):
        """
        equilibrium is reached when po meets pow
        :return: boolean 
        """
        return abs(self.__p_o - self._pow) < 0.1
