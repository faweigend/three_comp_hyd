import logging
import math

import numpy as np
from scipy import optimize


class ODEThreeCompHydSimulator:
    """
    Simulates Three Component Hydraulic Model responses using Ordinary Differential Equations
    """

    @staticmethod
    def tte(p_exp: float, conf: list, max_time: int = 5000) -> (float, float, float):

        # A1
        t1, ht1, gt1 = ODEThreeCompHydSimulator.tte_a1(p=p_exp, conf=conf)
        if t1 == np.inf or t1 > max_time:
            logging.info("EQUILIBRIUM IN A1: t: {} h: {} g: {}".format(t1, ht1, gt1))
            return t1, ht1, gt1

        t2, ht2, gt2 = ODEThreeCompHydSimulator.tte_a2(t1=t1, ht1=ht1, gt1=gt1, p=p_exp, conf=conf)

        # A3
        t3, ht3, gt3 = ODEThreeCompHydSimulator.tte_a3(t2=t2, ht2=ht2, gt2=gt2, p=p_exp, conf=conf)
        if t3 == np.inf or t3 > max_time:
            logging.info("EQUILIBRIUM IN A3: t: {} h: {} g: {}".format(t3, ht3, gt3))
            return t2, ht2, gt2

        # A4
        t4, ht4, gt4 = ODEThreeCompHydSimulator.tte_a4(t3=t3, ht3=ht3, gt3=gt3, p=p_exp, conf=conf)
        if t4 == np.inf or t4 > max_time:
            logging.info("EQUILIBRIUM IN A4: t: {} h: {} g: {}".format(t4, ht4, gt4))
            return t4, ht4, gt4

        # A5
        t5, ht5, gt5 = ODEThreeCompHydSimulator.tte_a5(t4=t4, ht4=ht4, gt4=gt4, p=p_exp, conf=conf)
        if t5 == np.inf or t5 > max_time:
            logging.info("EQUILIBRIUM IN A5: t: {} h: {} g: {}".format(t5, ht5, gt5))
            return t5, ht5, gt5

        # A6
        t6, ht6, gt6 = ODEThreeCompHydSimulator.tte_a6(t5=t5, ht5=ht5, gt5=gt5, p=p_exp, conf=conf)
        if t6 == np.inf or t6 > max_time:
            logging.info("EQUILIBRIUM IN A6: t: {} h: {} g: {}".format(t6, ht6, gt6))
            return t6, ht6, gt6

        return t6, ht6, gt6

    @staticmethod
    def tte_a1(p: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        m_ae = conf[2]
        theta = conf[5]
        phi = conf[7]

        # check if equilibrium in phase A1
        if ((m_ae * min(theta, 1 - phi)) / (p * (1 - phi))) >= 1:
            h = (a_anf * (1 - phi)) / m_ae * (1 - np.exp(-(m_ae * np.inf) / (a_anf * (1 - phi)))) * p / a_anf
            return np.inf, h, 0

        # end of phase A1 -> the time when h(t) = min(theta,1-phi)
        t1 = -a_anf * (1 - phi) / m_ae * np.log(1 - (m_ae * min(theta, 1 - phi) / (p * (1 - phi))))

        # return t1, h(t1), g(t1)
        return t1, min(theta, 1 - phi), 0

    @staticmethod
    def tte_a2(t1: float, ht1: float, gt1: float, p: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        m_ae = conf[2]
        theta = conf[5]
        phi = conf[7]

        # A2 is only applicable if phi is higher than the top of AnS
        if phi <= (1 - theta):
            return t1, ht1, gt1
        # linear utilization -> no equilibrium possible
        t2 = t1 + ((phi - (1 - theta)) * a_anf) / (p - m_ae)
        # return t2, h(t2), g(t2)
        return t2, theta, gt1

    @staticmethod
    def tte_a3(t2: float, ht2: float, gt2: float, p: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # A3 is only applicable if flow from Ae is not at max
        if phi > (1 - theta):
            return t2, ht2, gt2

        # taken from Equation 11 by Morton 1986
        # a = (m_ae * a_ans * (1 - theta - gamma) + m_ans * (a_anf + a_ans) * (1 - phi)) / (
        #         a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # my simplified form
        a = m_ae / (a_anf * (1 - phi)) + \
            m_ans / (a_ans * (1 - theta - gamma)) + \
            m_ans / (a_anf * (1 - theta - gamma))

        b = m_ae * m_ans / \
            (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        c = m_ans * (p * (1 - phi) - m_ae * theta) / \
            (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # wolfram alpha gave these estimations as solutions for l''(t) + a*l'(t) + b*l(t) = c
        r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
        r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

        # c1 and c2 can be determined with g(t2) = g'(t2) = 0
        s_c1 = -c / (b * (1 - r1 / r2)) * np.exp(-r1 * t2)
        s_c2 = s_c1 * np.exp(r1 * t2) * np.exp(-r2 * t2) * -r1 / r2

        def a3_gt(t):
            # the general solution for g(t)
            return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

        # substitute into EQ(9) for h
        def a3_ht(t):
            k1 = a_ans * (1 - theta - gamma) / m_ans * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - theta - gamma) / m_ans * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # if phi > gamma, then phase A3 transitions into phase A4 before AnS is empty
        ht3 = 1 - max(phi, gamma)

        # check if equilibrium in this phase
        if ht2 <= a3_ht(np.inf) <= ht3:
            return np.inf, a3_ht(np.inf), a3_gt(np.inf)
        else:
            t3 = optimize.fsolve(lambda t: ht3 - a3_ht(t), x0=np.array([t2]))[0]
            # use t3 to get g(t3)
            return t3, ht3, a3_gt(t3)

    @staticmethod
    def tte_a4(t3: float, ht3: float, gt3: float, p: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # phase A3 is not applicable if gamma is greater or equal to phi
        if gamma >= phi:
            return t3, ht3, gt3

        # b/a can be simplified as (p-m_ae)/(a_anf + a_ans)
        a = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
        b = (p - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

        # derivative g'(t3) can be calculated manually
        dgt3 = m_ans * (ht3 - gt3 - theta) / (a_ans * (1 - theta - gamma))

        # which then allows to derive c1 and c2
        s_c1 = ((p - m_ae) / (a_anf + a_ans) - dgt3) * np.exp(a * t3)
        s_c2 = (-t3 * b + dgt3) / a - (p - m_ae) / ((a_anf + a_ans) * a) + gt3

        def a4_gt(t):
            # general solution for g(t)
            return t * (p - m_ae) / (a_anf + a_ans) + s_c2 + s_c1 / a * np.exp(-a * t)

        def a4_dgt(t):
            # first derivative g'(t)
            return (p - m_ae) / (a_anf + a_ans) - s_c1 * np.exp(-a * t)

        def a4_ht(t):
            # EQ(9) with constants for g(t) and g'(t)
            return a_ans * (1 - theta - gamma) / m_ans * a4_dgt(t) + a4_gt(t) + theta

        ht4 = 1 - gamma
        # check if equilibrium in this phase
        if ht3 <= a4_ht(np.inf) <= ht4:
            return np.inf, a4_ht(np.inf), a4_gt(np.inf)
        else:
            t4 = optimize.fsolve(lambda t: ht4 - a4_ht(t), x0=np.array([t3]))[0]
            return t4, ht4, a4_gt(t4)

    @staticmethod
    def tte_a5(t4: float, ht4: float, gt4: float, p: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # phase A5 is not applicable if phi is greater or equal to gamma
        if phi >= gamma:
            return t4, ht4, gt4

        # g(t4) = gt4 can be solved for c
        s_cg = (gt4 - (1 - theta - gamma)) * np.exp((m_ans * t4) / ((1 - theta - gamma) * a_ans))

        def a5_gt(t):
            # generalised g(t) for phase A5
            return (1 - theta - gamma) + s_cg * np.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # as defined for EQ(21)
        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = -m_ae / ((1 - phi) * a_anf)
        g = p / a_anf
        b = m_ans * s_cg / ((1 - theta - gamma) * a_anf)

        # find c that matches h(t4) = ht4
        s_ch = (ht4 + b / ((a + k) * np.exp(k) ** t4) + g / a) / np.exp(a) ** t4

        def a5_ht(t):
            return -b / ((a + k) * np.exp(k) ** t) + s_ch * np.exp(a) ** t - g / a

        ht5 = 1 - phi
        # check if equilibrium in this phase
        if ht4 <= a5_ht(np.inf) <= ht5:
            return np.inf, a5_ht(np.inf), a5_gt(np.inf)
        else:
            # solve for time point where phase A5 ends h(t5) = 1-phi
            t5 = optimize.fsolve(lambda t: a5_ht(t) - ht5, x0=np.array([t4]))[0]
            return t5, ht5, a5_gt(t5)

    @staticmethod
    def tte_a6(t5: float, ht5: float, gt5: float, p: float, conf: list) -> (float, float, float):
        """
        Final phase A6 of a time to exhaustion trial. Expects inputs from Phase A5.
        :param t5: time at which A5 ended
        :param ht5: h(t5)
        :param gt5: g(t5)
        :param p: constant power output
        :param conf: configuration of hydraulic model
        :return: [t6: time until h=1, h(t6)=1, g(t6)]
        """
        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]

        # g(t5) = gt5 can be solved for c
        s_cg = (gt5 - (1 - theta - gamma)) / np.exp(-m_ans * t5 / ((1 - theta - gamma) * a_ans))

        def a6_gt(t):
            # generalised g(t) for phase A6
            return (1 - theta - gamma) + s_cg * math.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        k = m_ans / ((1 - theta - gamma) * a_ans)
        # a = -m_ae / a_anf
        b = (m_ans * s_cg) / ((1 - theta - gamma) * a_anf)
        # g = p / a_anf
        ag = (p - m_ae) / a_anf

        # h(t5) = ht5 can be solved for c
        s_ch = -t5 * ag + ((b * math.exp(-k * t5)) / k) + ht5

        def a6_ht(t):
            # generalised h(t) for phase A6
            return t * ag - ((b * math.exp(-k * t)) / k) + s_ch

        ht6 = 1.0
        # estimate an intitial guess that assumes no contribution from g
        initial_guess = (ht6 - s_ch) / ag
        # find end of phase A6. The time point where h(t6)=1
        t6 = optimize.fsolve(lambda t: ht6 - a6_ht(t), x0=np.array([initial_guess]))[0]

        return t6, ht6, a6_gt(t6)

    @staticmethod
    def rec(p_rec: float, conf: list, start_h: float, start_g: float, max_time: int = 5000) -> (float, float, float):

        t5, h5, g5 = ODEThreeCompHydSimulator.rec_a6(t6=0, h6=start_h, g6=start_g, p_rec=p_rec, conf=conf)

    @staticmethod
    def rec_a6(t6: float, h6: float, g6: float, p_rec: float, conf: list):
        """
        recovery from exhaustive exercise.
        :param t6: time in seconds at which recovery starts
        :param h6: depletion state of AnF when recovery starts
        :param g6: depletion state of AnS when recovery starts
        :param p_rec: constant recovery intensity
        :param conf: hydraulic model configuration
        :return: [time at which A6 rec ends, h(rt6), g(rt6)]
        """
        # PHASE A6
        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # g(t6) = g6 can be solved for c
        s_cg = (g6 - (1 - theta - gamma)) / np.exp(-m_ans * t6 / ((1 - theta - gamma) * a_ans))

        def a6_gt(t):
            # generalised g(t) for phase A6
            return (1 - theta - gamma) + s_cg * math.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        k = m_ans / ((1 - theta - gamma) * a_ans)
        # a = -m_ae / a_anf
        b = (m_ans * s_cg) / ((1 - theta - gamma) * a_anf)
        # g = p / a_anf
        ag = (p_rec - m_ae) / a_anf

        # h(t6) = 1 can be solved for c
        s_ch = -t6 * ag + ((b * math.exp(-k * t6)) / k) + h6

        def a6_ht(t):
            # generalised h(t) for recovery phase A6
            return t * ag - ((b * math.exp(-k * t)) / k) + s_ch

        # A6 rec ends either at beginning of A4 or A5
        h_target = max(1 - gamma, 1 - phi)

        # estimate an initial guess that assumes no contribution from g
        initial_guess = 0
        rt6 = optimize.fsolve(lambda t: a6_ht(t) - h_target, x0=np.array([initial_guess]))[0]

        return rt6, h_target, a6_gt(rt6)
