import math

import numpy as np
from scipy import optimize


class ODEThreeCompHydSimulator:
    """
    Simulates Three Component Hydraulic Model responses using Ordinary Differential Equations
    """

    @staticmethod
    def phase_a1(p: float, a_anf: float, m_ae: float, theta: float, phi: float) -> float:
        # end of phase A1 -> the time when h(t) = min(theta,1-phi)
        t1 = (-(a_anf * (1 - phi)) / m_ae) * math.log(1 - ((m_ae * min(theta, 1 - phi)) / (p * (1 - phi))))
        return t1

    @staticmethod
    def phase_a2(t1: float, p: float, a_anf: float, m_ae: float, theta: float, phi: float) -> float:
        # A2 is only applicable if phi is higher than the top of AnS
        if phi <= (1 - theta):
            return t1
        # linear utilization
        return t1 + ((phi - (1 - theta)) * a_anf) / (p - m_ae)

    @staticmethod
    def phase_a3(t2: float, p: float, a_anf: float, a_ans: float, m_ae: float, m_ans: float, theta: float,
                 gamma: float, phi: float) -> (float, float):

        # A3 is only applicable if flow from Ae is not at max
        if phi > (1 - theta):
            return t2, 0

        # taken from Equation 11 by Morton 1986
        a = (m_ae * a_ans * (1 - theta - gamma) + m_ans * (a_anf + a_ans) * (1 - phi)) / (
                a_anf * a_ans * (1 - phi) * (1 - theta - gamma))
        b = (m_ae * m_ans) / (
                a_anf * a_ans * (1 - phi) * (1 - theta - gamma))
        c = (m_ans * (p * (1 - phi) - m_ae * theta)) / (
                a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # use auxiliary equation to derive both negative roots r1 and r2
        coeff = [1, a, b]
        r1, r2 = np.roots(coeff)

        def a3_gt(c1, c2, t):
            # the general solution for g(t)
            return c1 * math.exp(r1 * t) + c2 * math.exp(r2 * t) + c / b

        def a3_dgt(c1, c2, t):
            # derivative of general solution
            return r1 * c1 * math.exp(r1 * t) + r2 * c2 * math.exp(r2 * t)

        # Find c1 and c2 that satisfy g(t2) and dg(t2) = 0
        res = optimize.fsolve(lambda cx: (a3_gt(cx[0], cx[1], t2), a3_dgt(cx[0], cx[1], t2)), x0=np.array([1, 1]))
        s_c1 = float(res[0])
        s_c2 = float(res[1])

        # substitute into EQ(9) for h
        def a2_ht(t):
            return (a_ans * (1 - theta - gamma)) / m_ans * a3_dgt(s_c1, s_c2, t) + a3_gt(s_c1, s_c2, t) + theta

        if phi < gamma:
            # the whole of AnS is utilized in phase A3
            t3 = optimize.fsolve(lambda t: abs((1 - gamma) - a2_ht(t)), x0=np.array([1]))[0]
        else:
            # phase A3 transitions into phase A4 before AnS is empty
            t3 = optimize.fsolve(lambda t: abs((1 - phi) - a2_ht(t)), x0=np.array([1]))[0]

        # use t3 to get g(t3)
        return t3, a3_gt(s_c1, s_c2, t3)

    @staticmethod
    def phase_a4(t3: float, gt3: float, p: float, a_anf: float, a_ans: float, m_ae: float, m_ans: float,
                 theta: float, gamma: float, phi: float) -> (float, float):
        # phase A3 is not applicable if gamma is greater or equal to phi
        if gamma >= phi:
            return t3, gt3

        a = ((a_anf + a_ans) * m_ans) / (a_anf * a_ans * (1 - theta - gamma))
        b = ((p - m_ae) * m_ans) / (a_anf * a_ans * (1 - theta - gamma))

        def a4_gt(c1, c2, t):
            # general solution for g(t)
            return (((b * t) + (c1 * math.exp(-a * t))) / a) + c2

        def a4_dgt(c1, c2, t):
            # first derivative g'(t)
            return (b / a) - c1 * math.exp(-a * t)

        # derivative g'(t3) can be calculated manually
        ht3 = max(theta, 1 - phi)
        dgt3 = (m_ans * (ht3 - gt3 - theta)) / (a_ans * (1 - theta - gamma))

        def general_gt(p):
            c1, c2 = p
            # g(t3) = previously estimated gt3
            est_gt3 = a4_gt(c1, c2, t3)
            # g'(t3) = manually estimated derivative
            est_dgt3 = a4_dgt(c1, c2, t3)
            return abs(est_gt3 - gt3), abs(est_dgt3 - dgt3)

        # solve for c1 and c2
        res = optimize.fsolve(general_gt, x0=np.array([0, 0]))
        s_c1 = float(res[0])
        s_c2 = float(res[1])

        def a4_ht(t):
            # EQ(9) with constants for g(t) and g'(t)
            return ((a_ans * (1 - theta - gamma)) / m_ans) * a4_dgt(s_c1, s_c2, t) + a4_gt(s_c1, s_c2, t) + theta

        # find end of A4 where g(t4) = (1-gamma)
        t4 = optimize.fsolve(lambda t: abs((1 - gamma) - a4_ht(t)), x0=np.array([t3]))[0]

        # return t4 and g(t4)
        return t4, a4_gt(s_c1, s_c2, t4)

    @staticmethod
    def phase_a5(t4: float, gt4: float, p: float, a_anf: float, a_ans: float, m_ae: float, m_ans: float,
                 theta: float, gamma: float, phi: float) -> (float, float):

        # phase A5 is not applicable if phi is greater or equal to gamma
        if phi >= gamma:
            return t4, gt4

        def a5_gt(c, t):
            # generalised g(t) for phase A5
            return (1 - theta - gamma) + c * math.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # find c that fits to known g(t4) = gt4
        s_c1 = optimize.fsolve(lambda c: abs(gt4 - a5_gt(c, t4)), x0=np.array([1]))[0]

        # as defined for EQ(21)
        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = -m_ae / ((1 - phi) * a_anf)
        b = (m_ans * s_c1) / ((1 - theta - gamma) * a_anf)
        g = p / a_anf

        def a5_ht(c, t):
            return ((-b * math.exp(-k * t)) / (a + k)) + (c * math.exp(a * t)) - (g / a)

        # find c that matches g(t4) = (1-gamma)
        s_c2 = optimize.fsolve(lambda c: abs((1 - gamma) - a5_ht(c, t4)), x0=np.array([1]))[0]

        # solve for time point where phase A5 ends h(t5)=(1-phi)
        t5 = optimize.fsolve(lambda t: abs((1 - phi) - a5_ht(s_c2, t)), x0=np.array([t4]))[0]

        # return t5 and g(t5)
        return t5, a5_gt(s_c1, t5)

    @staticmethod
    def phase_a6(t5: float, gt5: float, p: float, a_anf: float, a_ans: float, m_ae: float, m_ans: float, theta: float,
                 gamma: float, phi: float) -> (float, float):

        def a6_gt(c, t):
            # generalised g(t) for phase A6
            return (1 - theta - gamma) + c * math.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # find c that fits to known g(t5) = given gt5
        s_cg = optimize.fsolve(lambda c: abs(gt5 - a6_gt(c, t5)), x0=np.array([1]))[0]

        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = (p - m_ae) / a_anf
        b = (m_ans * s_cg) / ((1 - theta - gamma) * a_anf)

        def a6_ht(cx, t):
            # generalised h(t) for phase A6
            return (a * t) - ((b * math.exp(-k * t)) / k) + cx

        # find ch that matches h(t5) = (1-phi)
        s_ch = optimize.fsolve(lambda c: abs((1 - phi) - a6_ht(c, t5)), x0=np.array([1]))[0]

        # find end of phase A6. The time point where h(t6)=1
        # condition t > t5 added
        t6 = optimize.fsolve(lambda t: abs(1 - a6_ht(s_ch, t)) + int(t < t5), x0=np.array([t5]))[0]

        # return time to exhaustion (t6) and g(t6)
        return t6, a6_gt(s_cg, t6)
