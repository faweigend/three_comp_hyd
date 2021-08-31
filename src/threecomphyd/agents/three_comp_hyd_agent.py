from threecomphyd import config

from threecomphyd.agents.hyd_agent_basis import HydAgentBasis


class ThreeCompHydAgent(HydAgentBasis):
    """
    Implementation of the adjusted version of Morton's generalised three component model of human bioenergetics.
    The numeric procedures by Sundström were used and adapted to some of Morton's procedures.
    Further, slightly adjusted and extended with some extra limitations that are necessary for extreme values.
    See the document for the reasoning behind procedures in _estimate_power_output.
    """

    def __init__(self, hz, a_anf, a_ans, m_ae, m_ans, m_anf, the, gam, phi):
        """
        :param hz: calculations per second
        :param a_anf: cross sectional area of AnF
        :param a_ans: cross sectional area of AnS
        :param m_ae: maximal flow from Ae to AnF
        :param m_ans: maximal flow from AnS to AnF
        :param m_anf: maximal flow from AnF to AnS
        :param the: theta (distance top -> top AnS)
        :param gam: gamma (distance bottom -> bottom AnS)
        :param phi: phi (distance bottom -> bottom Ae)
        """
        super().__init__(hz=hz)

        # constants
        self.__theta, self.__gamma, self.__phi = the, gam, phi
        # height of vessel AnS
        self.__height_ans = 1 - self.__theta - self.__gamma

        # the AnS tank size constraint that corresponds to
        # constraint: theta < 1 - phi
        if self.__height_ans <= 0:
            raise UserWarning("AnS has negative height: Theta {} Gamma {} Phi {}".format(the, gam, phi))

        if config.three_comp_phi_constraint is True:
            # the "hitting the wall" constraint that says glycogen can be depleted below VO2 MAX
            if self.__phi > self.__gamma:
                raise UserWarning("phi not smaller gamma")

        # vessel areas
        self.__a_anf = a_anf  # area of vessel AnF
        self.__a_ans = a_ans  # area of AnS

        # max flows
        self.__m_ae = m_ae  # max flow from Ae to AnF
        self.__m_ans = m_ans  # max flow from AnS to AnF
        self.__m_anf = m_anf  # max flow from AnA to AnS

        # self.__w_m = 100  # max instantaneous power (not in use yet)

        # variable parameters
        self.__h = 0  # state of depletion of vessel AnF
        self.__g = 0  # state of depletion of vessel AnS
        self.__p_ae = 0  # flow from Ae to AnF
        self.__p_an = 0  # flow from AnS to AnF
        self.__m_flow = 0  # maximal flow through pg according to liquid diffs

    def __str__(self):
        """
        print function
        :return: parameter overview
        """
        return "Three Component Hydraulic Agent \n" \
               "AnF, AnS, Mae, Mans, Manf, the, gam, phi \n " \
               "{}".format([self.__a_anf, self.__a_ans, self.__m_ae, self.__m_ans,
                            self.__m_anf, self.__theta, self.__gamma, self.__phi])

    def _estimate_possible_power_output(self):
        """
        Estimates liquid flow to meet set power demands. Exhaustion flag is set and internal tank fill levels and
        pipe flows are updated.
        :return: power output
        """

        def raise_detailed_error_report():
            raise UserWarning("Unhandled tank fill level state \n"
                              "gamma:  {} \n "
                              "theta:  {} \n "
                              "phi:    {} \n "
                              "AnF:    {} \n "
                              "AnS:    {} \n "
                              "g:      {} \n "
                              "h:      {} \n "
                              "m_ae:   {} \n"
                              "m_ans:  {} \n"
                              "m_anf:  {} \n"
                              "p_Ae:   {} \n"
                              "p_An:   {} \n"
                              "pow:    {} \n".format(self.__gamma,
                                                     self.__theta,
                                                     self.__phi,
                                                     self.__a_anf,
                                                     self.__a_ans,
                                                     self.__g,
                                                     self.__h,
                                                     self.__m_ae,
                                                     self.__m_ans,
                                                     self.__m_anf,
                                                     self.__p_ae,
                                                     self.__p_an,
                                                     self._pow))

        # step 1: drop level in AnF according to power demand
        # estimate h_{t+1}: scale to hz (delta t) and drop the level of AnF
        self.__h += self._pow / self.__a_anf / self._hz

        # step 2: determine tank flows to respond to the new change in h_{t+1}

        # step 2 a: determine oxygen energy flow (p_{Ae})
        # level AnF above pipe exit. Scale contribution according to h level
        if 0 <= self.__h < (1 - self.__phi):
            # contribution from Ae scales with maximal flow capacity
            self.__p_ae = self.__m_ae * self.__h / (1 - self.__phi)
        # at maximum rate because level h of AnF is below pipe exit of Ae
        elif (1 - self.__phi) <= self.__h:
            # max contribution R1 = m_ae
            self.__p_ae = self.__m_ae
        else:
            raise_detailed_error_report()

        # step 2 b: determine the slow component energy flow (p_{An})
        # [no change] AnS full and level AnF above level AnS
        if self.__h <= self.__theta and self.__g == 0:
            self.__p_an = 0.0
        # [no change] AnS empty and level AnF below pipe exit
        elif self.__h >= (1 - self.__gamma) and self.__g == self.__height_ans:
            self.__p_an = 0.0
        # [no change] h at equal with g
        elif self.__h == (self.__g + self.__theta):
            self.__p_an = 0.0
        else:
            # [restore] if level AnF above level AnS and AnS is not full
            if self.__h < (self.__g + self.__theta) and self.__g > 0:
                # see EQ (16) in Morton (1986)
                self.__p_an = -self.__m_anf * (self.__g + self.__theta - self.__h) / (1 - self.__gamma)
            # [utilise] if level AnS above level AnF and level AnF above pipe exit of AnS
            elif (self.__g + self.__theta) < self.__h < (1 - self.__gamma):
                # EQ (9) in Morton (1986)
                self.__p_an = self.__m_ans * (self.__h - self.__g - self.__theta) / self.__height_ans
            # [utilise max] if level AnF below or at AnS pipe exit and AnS not empty
            elif (1 - self.__gamma) <= self.__h and self.__g < self.__height_ans:
                # the only thing that affects flow is the amount of remaining liquid (pressure)
                # EQ (20) Morton (1986)
                self.__p_an = self.__m_ans * (self.__height_ans - self.__g) / self.__height_ans
            else:
                raise_detailed_error_report()

            # This check is added to handle cases where the flow causes level height swaps between AnS and AnF
            self.__m_flow = ((self.__h - (self.__g + self.__theta)) / (
                    (1 / self.__a_ans) + (1 / self.__a_anf)))

            # Cap flow according to estimated limits
            if self.__p_an < 0:
                self.__p_an = max(self.__p_an, self.__m_flow)
                # don't refill more than there is capacity
                self.__p_an = max(self.__p_an, -self.__g * self.__a_ans)
            elif self.__p_an > 0:
                self.__p_an = min(self.__p_an, self.__m_flow)
                # don't drain more than is available in AnS
                self.__p_an = min(self.__p_an, (self.__height_ans - self.__g) * self.__a_ans)

        # level AnS is adapted to estimated change
        # g increases as p_An flows out
        self.__g += self.__p_an / self.__a_ans / self._hz
        # refill or deplete AnF according to AnS flow and Power demand
        # h decreases as p_Ae and p_An flow in
        self.__h -= (self.__p_ae + self.__p_an) / self.__a_anf / self._hz

        # step 3: account for rounding errors and set exhaustion flag
        self._exhausted = self.__h >= 1.0
        # apply limits so that tanks cannot be fuller than full or emptier than empty
        self.__g = max(self.__g, 0.0)
        self.__g = min(self.__g, self.__height_ans)
        self.__h = max(self.__h, 0.0)
        self.__h = min(self.__h, 1.0)

        return self._pow

    def is_exhausted(self):
        """
        exhaustion is reached when level in AnF cannot sustain power demand
        :return: simply returns the exhausted boolean
        """
        return bool(self.__h >= 1.0)

    def is_recovered(self):
        """
        recovery is estimated according to w_p ratio
        :return: simply returns the recovered boolean
        """
        return self.get_w_p_ratio() == 1.0

    def is_equilibrium(self):
        """
        equilibrium is reached when ph meets pow and AnS does not contribute or drain
        :return: boolean
        """
        return abs(self.__p_ae - self._pow) < 0.1 and abs(self.__p_an) < 0.1

    def reset(self):
        """power parameters"""
        super().reset()
        # variable parameters
        self.__h = 0  # state of depletion of vessel AnF
        self.__g = 0  # state of depletion of vessel AnS
        self.__p_ae = 0  # flow from Ae to AnF
        self.__p_an = 0  # flow from AnS to AnF

    def get_w_p_ratio(self):
        """
        :return: wp estimation between 0 and 1 for comparison to CP models
        """
        return (1.0 - self.__h) * ((self.__height_ans - self.__g) / self.__height_ans)

    def get_fill_anf(self):
        """
        :return: fill level of AnF between 0 - 1
        """
        return 1 - self.__h

    def get_fill_ans(self):
        """
        :return:fill level of AnS between 0 - 1
        """
        return (self.__height_ans - self.__g) / self.__height_ans

    @property
    def phi_constraint(self):
        """
        getter for phi_constraint flag
        :return boolean ture or false
        """
        return config.three_comp_phi_constraint

    @property
    def a_anf(self):
        """
        :return cross sectional area of AnF
        """
        return self.__a_anf

    @property
    def a_ans(self):
        """
        :return cross sectional area of AnS
        """
        return self.__a_ans

    @property
    def theta(self):
        """
        :return theta (distance top -> top AnS)
        """
        return self.__theta

    @property
    def gamma(self):
        """
        :return gamma (distance bottom -> bottom AnS)
        """
        return self.__gamma

    @property
    def phi(self):
        """
        :return phi (distance bottom -> bottom Ae)
        """
        return self.__phi

    @property
    def height_ans(self):
        """
        :return height of vessel AnS
        """
        return self.__height_ans

    @property
    def m_ae(self):
        """
        :return maximal flow from Ae to AnF
        """
        return self.__m_ae

    @property
    def m_ans(self):
        """
        :return maximal flow from AnS to AnF
        """
        return self.__m_ans

    @property
    def m_anf(self):
        """
        :return maximal flow from AnF to AnS
        """
        return self.__m_anf

    def get_m_flow(self):
        """
        :return maximal flow through pg from liquid height diffs
        """
        return self.__m_flow

    def get_g(self):
        """
        :return state of depletion of vessel AnS
        """
        return self.__g

    def set_g(self, g):
        """
        setter for state of depletion of vessel AnS
        """
        self.__g = g

    def get_h(self):
        """
        :return state of depletion of vessel AnF
        """
        return self.__h

    def set_h(self, h):
        """
        setter for state of depletion of vessel AnF
        """
        self.__h = h

    def get_p_ae(self):
        """
        :return flow from Ae to AnF
        """
        return self.__p_ae

    def get_p_an(self):
        """
        :return flow from AnS to AnF
        """
        return self.__p_an
