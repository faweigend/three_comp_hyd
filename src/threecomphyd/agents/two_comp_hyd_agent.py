from threecomphyd.agents.hyd_agent_basis import HydAgentBasis


class TwoCompHydAgent(HydAgentBasis):

    def __init__(self, hz, w_p, m_u, phi, psi: float = 0.0):
        """
        :param hz: calculations per second
        :param w_p: cross sectional area of W'
        :param m_u: maximal flow from O to w'
        :param phi: phi (distance Ae to bottom)
        :param phi: psi (distance W' to top)
        """
        super().__init__(hz=hz)

        if psi > 1 - phi:
            raise UserWarning("Top of W\' has to be above or at bottom of Ae (psi > 1 - phi must be False)")

        # constants
        self.__phi = phi
        self.__psi = psi
        self.__w_p = w_p  # area of vessel W'
        self.__m_u = m_u  # max flow from O to P (max aerobic energy consumption i.e. VO2max)

        # variable parameters
        self.__h = self.__psi  # state of depletion of vessel W'
        self.__p_o = 0  # flow through R1 (oxygen pipe)

    @property
    def w_p(self):
        """:return cross sectional area of W'"""
        return self.__w_p

    @property
    def phi(self):
        """:return phi (distance Ae to bottom)"""
        return self.__phi

    @property
    def psi(self):
        """:return psi (distance W' to top)"""
        return self.__psi

    @property
    def m_u(self):
        """:return max flow through R1"""
        return self.__m_u

    def get_h(self):
        """:return state of depletion of vessel P"""
        return self.__h

    def get_p_o(self):
        """:return flow through R1"""
        return self.__p_o

    def _estimate_possible_power_output(self):
        """
        Update internal capacity estimations by one step.
        :return: the amount of power that the athlete was able to put out
        """
        p = self._pow

        # determine oxygen energy flow (R1)
        # level P above pipe exit. Scale contribution according to h level
        if self.__psi < self.__h <= (1.0 - self.__phi):
            # contribution through R1 scales with maximal flow capacity
            self.__p_o = self.__m_u * (self.__h / (1.0 - self.__phi))
        # at maximum rate because level h of P is below pipe exit of O
        elif (1.0 - self.__phi) < self.__h <= 1.0:
            # above aerobic threshold (?) max contribution R1 = m_o
            self.__p_o = self.__m_u

        # the change on level in W' is determined by flow p_o
        self.__h += ((p - self.__p_o) / self.__w_p) / self._hz

        # also W' cannot be fuller than full
        if self.__h < self.__psi:
            self.__h = self.__psi
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
        return self.__h == self.__psi

    def is_equilibrium(self):
        """
        equilibrium is reached when po meets pow
        :return: boolean 
        """
        return abs(self.__p_o - self._pow) < 0.1
