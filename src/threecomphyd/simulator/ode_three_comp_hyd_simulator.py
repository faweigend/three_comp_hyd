import math

import numpy as np
from scipy import optimize


class ODEThreeCompHydSimulator:
    """
    Simulates Three Component Hydraulic Model responses using Ordinary Differential Equations
    """

    @staticmethod
    def phase_a1(p: float, conf: list) -> (float, float, float):

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
    def phase_a2(t1: float, ht1: float, gt1: float, p: float, conf: list) -> (float, float, float):

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
    def phase_a3(t2: float, ht2: float, gt2: float, p: float, conf: list) -> (float, float, float):

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
        a = (m_ae * a_ans * (1 - theta - gamma) + m_ans * (a_anf + a_ans) * (1 - phi)) / (
                a_anf * a_ans * (1 - phi) * (1 - theta - gamma))
        b = m_ae * m_ans / (
                a_anf * a_ans * (1 - phi) * (1 - theta - gamma))
        c = m_ans * (p * (1 - phi) - m_ae * theta) / (
                a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # wolfram alpha gave these estimations as solutions for l''(t) + a*l'(t) + b*l(t) = c
        r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
        r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

        def a3_gt(c1, c2, t):
            # the general solution for g(t)
            return c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t) + c / b

        # c1 and c2 can be determined with g(t2) = g'(t2) = 0
        s_c1 = -c / (b * (1 - r1 / r2)) * np.exp(-r1 * t2)
        s_c2 = s_c1 * np.exp(r1 * t2) * np.exp(-r2 * t2) * -r1 / r2

        # substitute into EQ(9) for h
        def a3_ht(t):
            k1 = a_ans * (1 - theta - gamma) / m_ans * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - theta - gamma) / m_ans * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # if phi > gamma, then phase A3 transitions into phase A4 before AnS is empty
        ht3 = 1 - max(phi, gamma)

        # check if equilibrium in this phase
        if ht2 <= a3_ht(np.inf) <= ht3:
            return np.inf, a3_ht(np.inf), a3_gt(s_c1, s_c2, np.inf)
        else:
            t3 = optimize.newton(lambda t: ht3 - a3_ht(t), x0=np.array([t2]))[0]
            # use t3 to get g(t3)
            return t3, ht3, a3_gt(s_c1, s_c2, t3)

    @staticmethod
    def phase_a4(t3: float, ht3: float, gt3: float, p: float, conf: list) -> (float, float, float):

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

        # b is not needed by simplifying b/a as (p-m_ae)/(a_anf + a_ans)
        a = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
        b = (p - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

        def a4_gt(c1, c2, t):
            # general solution for g(t)
            return t * b / a + c2 + c1 / a * np.exp(-a * t)

        def a4_dgt(c1, t):
            # first derivative g'(t)
            return b / a - c1 * np.exp(-a * t)

        # derivative g'(t3) can be calculated manually
        dgt3 = m_ans * (ht3 - gt3 - theta) / (a_ans * (1 - theta - gamma))

        # which then allows to derive c1 and c2
        s_c1 = (b / a - dgt3) * np.exp(a * t3)
        s_c2 = (-t3 * b + dgt3) / a - b / a ** 2 + gt3

        def a4_ht(t):
            # EQ(9) with constants for g(t) and g'(t)
            return a_ans * (1 - theta - gamma) / m_ans * a4_dgt(s_c1, t) + a4_gt(s_c1, s_c2, t) + theta

        ht4 = 1 - gamma
        # check if equilibrium in this phase
        if ht3 <= a4_ht(np.inf) <= ht4:
            return np.inf, a4_ht(np.inf), a4_gt(s_c1, s_c2, np.inf)
        else:
            t4 = optimize.newton(lambda t: ht4 - a4_ht(t), x0=np.array([t3]))[0]
            return t4, ht4, a4_gt(s_c1, s_c2, t4)

    @staticmethod
    def phase_a5(t4: float, ht4: float, gt4: float, p: float, conf: list) -> (float, float, float):

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

        def a5_gt(c, t):
            # generalised g(t) for phase A5
            return (1 - theta - gamma) + c * math.exp(-m_ans * t / ((1 - theta - gamma) * a_ans))

        # find values for positive and negative signs
        pos = (theta + gamma - 2) / math.exp(-m_ans * t4 / ((1 - theta - gamma) * a_ans))
        neg = (theta + gamma) / math.exp(-m_ans * t4 / ((1 - theta - gamma) * a_ans))
        s_cg = optimize.brentq(lambda c: gt4 - a5_gt(c, t4), a=pos, b=neg)

        # as defined for EQ(21)
        k = m_ans / ((1 - theta - gamma) * a_ans)
        # g not necessary as g/a = p*(1-phi)/m_ae
        a = -m_ae / ((1 - phi) * a_anf)
        b = m_ans * s_cg / ((1 - theta - gamma) * a_anf)

        def a5_ht(c, t):
            return (-b * math.exp(-k * t) / (a + k)) + p * (1 - phi) / m_ae + (c * math.exp(a * t))

        mp = math.exp(a * t4)  # multiplied part of a5_ht(t4) : c * mp
        fp = a5_ht(0, t4)  # first part in a5_ht(t4) : fp + c*mp
        pos = (-fp + 1) / mp
        neg = (-fp - 1) / mp
        # find c that matches h(t4) = ht4
        s_ch = optimize.brentq(lambda c: ht4 - a5_ht(c, t4), a=pos, b=neg)

        ht5 = 1 - phi
        # check if equilibrium in this phase
        if ht4 <= a5_ht(np.inf, s_ch) <= ht5:
            return np.inf, a5_ht(np.inf, s_ch), a5_gt(s_cg, np.inf)
        else:
            # solve for time point where phase A5 ends h(t5)=(1-phi)
            t5 = optimize.newton(lambda t: ht5 - a5_ht(s_ch, t), x0=np.array([t4]))[0]
            # t5 = optimize.fsolve(lambda t: ht5 - a5_ht(s_ch, t), x0=np.array([t4]))[0]
            # t5 = optimize.minimize(lambda t: np.abs(ht5 - a5_ht(s_ch, t)), x0=np.array([t4]))['x'][0]
            return t5, ht5, a5_gt(s_cg, t5)

    @staticmethod
    def phase_a6(t5: float, ht5: float, gt5: float, p: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]

        def a6_gt(c, t):
            # generalised g(t) for phase A6
            return (1 - theta - gamma) + c * math.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # find values for positive and negative signs
        pos = (theta + gamma - 2) / math.exp((-m_ans * t5) / ((1 - theta - gamma) * a_ans))
        neg = (theta + gamma) / math.exp((-m_ans * t5) / ((1 - theta - gamma) * a_ans))
        s_cg = optimize.brentq(lambda c: gt5 - a6_gt(c, t5), a=pos, b=neg)

        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = -m_ae / a_anf
        b = (m_ans * s_cg) / ((1 - theta - gamma) * a_anf)
        g = p / a_anf

        def a6_ht(cx, t):
            # generalised h(t) for phase A6
            return t * (a + g) - ((b * math.exp(-k * t)) / k) + cx

        fp = a6_ht(0, t5)  # get first part of equation for t5 to estimate limits for sign change
        if fp >= 0:
            pos = ht5 + 1
            neg = -fp
        else:
            pos = -fp + ht5 + 1
            neg = 0
        # find ch that matches h(t5) = ht5
        s_ch = optimize.brentq(lambda c: ht5 - a6_ht(c, t5), a=pos, b=neg)

        ht6 = 1.0
        # find end of phase A6. The time point where h(t6)=1
        t6 = optimize.newton(lambda t: ht6 - a6_ht(s_ch, t), x0=np.array([t5]))[0]
        return t6, ht6, a6_gt(s_cg, t6)
