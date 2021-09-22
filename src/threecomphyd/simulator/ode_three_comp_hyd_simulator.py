import logging
import math

import numpy as np


class ODEThreeCompHydSimulator:
    """
    Simulates Three Component Hydraulic Model responses using Ordinary Differential Equations
    """

    # precision epsilon for threshold checks
    eps = 0.000001
    max_time = 5000

    @staticmethod
    def get_recovery_ratio_wb1_wb2(conf: list, p_exp: float, p_rec: float, t_rec: float) -> float:

        t_max = ODEThreeCompHydSimulator.max_time

        # Start with first time to exhaustion bout
        tte_1, h, g = ODEThreeCompHydSimulator.tte(conf=conf,
                                                   start_h=0, start_g=0,
                                                   p_exp=p_exp, t_max=t_max)
        if tte_1 >= t_max:
            logging.info("Exhaustion not reached during TTE")
            return 0

        # now recovery
        rec, h, g = ODEThreeCompHydSimulator.rec(conf=conf,
                                                 start_h=h, start_g=g,
                                                 p_rec=p_rec, t_max=t_rec)

        # and work bout two
        tte_2, h, g = ODEThreeCompHydSimulator.tte(conf=conf,
                                                   start_h=h, start_g=g,
                                                   p_exp=p_exp, t_max=t_max)

        return tte_2 / tte_1 * 100.0

    @staticmethod
    def tte(conf: list, start_h: float, start_g: float, p_exp: float, t_max: float = 5000) -> (float, float, float):

        phases = [ODEThreeCompHydSimulator.work_lAe,
                  ODEThreeCompHydSimulator.work_lAe_rAnS,
                  ODEThreeCompHydSimulator.work_fAe,
                  ODEThreeCompHydSimulator.work_fAe_rAnS,
                  ODEThreeCompHydSimulator.work_lAe_lAnS,
                  ODEThreeCompHydSimulator.work_fAe_lAnS,
                  ODEThreeCompHydSimulator.work_lAe_fAns,
                  ODEThreeCompHydSimulator.work_fAe_fAnS]

        # start with fully reset agent
        t, h, g = 0, start_h, start_g
        # iterate through all phases until end is reached
        for phase in phases:
            t, h, g = phase(t, h, g,
                            p_exp=p_exp, t_max=t_max, conf=conf)

            # if recovery time is reached return fill levels at that point
            if t == np.inf or t == t_max:
                return t, h, g

        # if all phases complete full exhaustion is reached
        return t, h, g

    @staticmethod
    def work_lAe(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        m_ae = conf[2]
        theta = conf[5]
        phi = conf[7]

        # phase ends at bottom of Ae or top of AnS
        h_target = min(theta, 1 - phi)

        # This phase is not applicable if fill-level of AnF below pipe exit Ae or top of AnS, ..
        # ... or AnS not full
        if h_s > h_target or g_s > 0.0 + ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # constant can be derived from known t_s and h_s
        c1 = (h_s - p_exp * (1 - phi) / m_ae) * np.exp(-m_ae * t_s / a_anf * (phi - 1))

        # general solution for h(t) using c1
        def ht(t):
            return p_exp * (1 - phi) / m_ae + c1 * np.exp(m_ae * t / (a_anf * (phi - 1)))

        # check if max time is reached in this phase
        if ht(t_max) <= h_target:
            return t_max, ht(t_max), g_s
        else:
            # end of phase A1 -> the time when h(t) = min(theta,1-phi)
            t_end = a_anf * (phi - 1) / m_ae * np.log((h_target - p_exp * (1 - phi) / m_ae) / c1)
            return t_end, h_target, g_s

    @staticmethod
    def work_lAe_rAnS(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (
            float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_anf = conf[4]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # This phase is not applicable if fill-level of AnF below pipe exit Ae, ..
        # ... or AnS fill-level is above AnF fill-level ...
        # ... or AnS is full
        if h_s > 1 - phi or g_s + theta <= h_s or g_s <= 0.0 + ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # TODO: mostly copied from rec A3R2 find ways to combine equations
        # EQ 16 and 17 substituted in EQ 8
        a = m_ae / (a_anf * (1 - phi)) + \
            m_anf / (a_ans * (1 - gamma)) + \
            m_anf / (a_anf * (1 - gamma))

        b = m_ae * m_anf / \
            (a_anf * a_ans * (1 - phi) * (1 - gamma))

        c = m_anf * (p_exp * (1 - phi) - m_ae * theta) / \
            (a_anf * a_ans * (1 - phi) * (1 - gamma))

        # wolfram alpha gave these estimations as solutions for l''(t) + a*l'(t) + b*l(t) = c
        r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
        r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

        # uses Al dt/dl part of EQ(16) == dl/dt of EQ(14) solved for c2
        # and then substituted in EQ(14) and solved for c1
        s_c1 = (c / b - (m_anf * (g_s + theta - h_s)) / (a_ans * r2 * (1 - gamma)) - g_s) / \
               (np.exp(r1 * t_s) * (r1 / r2 - 1))

        # uses EQ(14) with solution for c1 and solves for c2
        s_c2 = (g_s - s_c1 * np.exp(r1 * t_s) - c / b) / np.exp(r2 * t_s)

        def gt(t):
            # the general solution for g(t)
            return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

        # substitute into EQ(9) for h
        def ht(t):
            k1 = a_ans * (1 - gamma) / m_anf * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - gamma) / m_anf * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # find the point where g(t) == 0
        g0 = ODEThreeCompHydSimulator.optimize(func=lambda t: gt(t),
                                               initial_guess=t_s,
                                               max_steps=t_max)

        # find the point where fill-levels AnS and AnF are at equal
        gtht = ODEThreeCompHydSimulator.optimize(func=lambda t: gt(t) + theta - ht(t),
                                                 initial_guess=t_s,
                                                 max_steps=t_max)

        # find the point where h drops below pipe exit of Ae
        h_end = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - phi - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # phase ends at the earliest of these time points
        t_end = min(g0, gtht, h_end)
        return t_end, ht(t_end), gt(t_end)

    @staticmethod
    def work_fAe(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (float, float, float):

        a_anf = conf[0]
        m_ae = conf[2]
        theta = conf[5]
        phi = conf[7]

        # this phase not applicable if h is not in-between 1-theta and phi and ...
        # ... AnS is not full
        if not theta >= h_s >= (1 - phi) or g_s > 0.0 + ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # linear utilization -> no equilibrium possible
        t_end = t_s + ((phi - (1 - theta)) * a_anf) / (p_exp - m_ae)

        # check if max time is reached before phase end
        if t_end > t_max:
            h_end = h_s + (t_max - t_s) * (p_exp - m_ae) / a_anf
            return t_end, h_end, g_s

        else:
            return t_end, theta, g_s

    @staticmethod
    def work_fAe_rAnS(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (
            float, float, float):
        # TODO: mostly copied from rec A4R2 find ways to combine equations

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_anf = conf[4]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # This phase is not applicable if fill-level of AnF above or at pipe exit Ae, ..
        # ... or AnS fill-level is above AnF fill-level ...
        # ... or AnS is full
        if h_s <= 1 - phi - ODEThreeCompHydSimulator.eps or \
                g_s + theta <= h_s or \
                g_s <= 0.0 + ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # if h is above g (flow from AnF into AnS)
        a_hg = (a_anf + a_ans) * m_anf / (a_anf * a_ans * (1 - gamma))
        b_hg = (p_exp - m_ae) * m_anf / (a_anf * a_ans * (1 - gamma))

        # derivative g'(t4) can be calculated manually
        dgt4_hg = - m_anf * (g_s + theta - h_s) / (a_ans * (1 - gamma))

        # which then allows to derive c1 and c2
        s_c1_gh = ((p_exp - m_ae) / (a_anf + a_ans) - dgt4_hg) * np.exp(a_hg * t_s)
        s_c2_gh = (-t_s * b_hg + dgt4_hg) / a_hg - (p_exp - m_ae) / ((a_anf + a_ans) * a_hg) + g_s

        def gt(t):
            # general solution for g(t)
            return t * (p_exp - m_ae) / (a_anf + a_ans) + s_c2_gh + s_c1_gh / a_hg * np.exp(-a_hg * t)

        def dgt(t):
            # first derivative g'(t)
            return (p_exp - m_ae) / (a_anf + a_ans) - s_c1_gh * np.exp(-a_hg * t)

        def ht(t):
            # EQ(16) with constants for g(t) and g'(t)
            return a_ans * (1 - gamma) / m_anf * dgt(t) + gt(t) + theta

        # find the point where g(t) == 0
        g0 = ODEThreeCompHydSimulator.optimize(func=lambda t: gt(t),
                                               initial_guess=t_s,
                                               max_steps=t_max)

        # find the point where fill-levels AnS and AnF are at equal
        gtht = ODEThreeCompHydSimulator.optimize(func=lambda t: gt(t) + theta - ht(t),
                                                 initial_guess=t_s,
                                                 max_steps=t_max)

        t_end = min(g0, gtht)
        return t_end, ht(t_end), gt(t_end)

    @staticmethod
    def work_lAe_lAnS(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (
            float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # This phase is not applicable h is lower than pipe exit Ae...
        # ... or if reflow into AnS is happening
        if h_s > (1 - phi) or g_s + theta > h_s + ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # taken from Equation 11 by Morton 1986
        # a = (m_ae * a_ans * (1 - theta - gamma) + m_ans * (a_anf + a_ans) * (1 - phi)) / (
        #         a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # my simplified form
        a = m_ae / (a_anf * (1 - phi)) + \
            m_ans / (a_ans * (1 - theta - gamma)) + \
            m_ans / (a_anf * (1 - theta - gamma))

        b = m_ae * m_ans / \
            (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        c = m_ans * (p_exp * (1 - phi) - m_ae * theta) / \
            (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # wolfram alpha gave these estimations as solutions for l''(t) + a*l'(t) + b*l(t) = c
        r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
        r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

        # uses Al dt/dl part of EQ(9) == dl/dt of EQ(12) solved for c2
        # and then substituted in EQ(12) and solved for c1
        s_c1 = (g_s - m_ans * (h_s - g_s - theta) / (a_ans * r2 * (1 - theta - gamma)) - c / b) / (
                np.exp(r1 * t_s) * (1 - r1 / r2))
        # uses EQ(12) with solution for c1 and solves for c2
        s_c2 = (g_s - s_c1 * np.exp(r1 * t_s) - c / b) / np.exp(r2 * t_s)

        def gt(t):
            # the general solution for g(t)
            return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

        # substitute into EQ(9) for h
        def ht(t):
            k1 = a_ans * (1 - theta - gamma) / m_ans * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - theta - gamma) / m_ans * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # if phi > gamma, then phase A3 transitions into phase A4 before AnS is empty
        h_target = 1 - max(phi, gamma)

        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: h_target - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        return t_end, ht(t_end), gt(t_end)

    @staticmethod
    def work_fAe_lAnS(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (
            float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # This phase is not applicable h is not in-between pipe exit Ae and pipe exit AnS...
        # ... or if reflow into AnS is happening
        if not 1 - phi - ODEThreeCompHydSimulator.eps <= h_s <= 1 - gamma + ODEThreeCompHydSimulator.eps or \
                g_s + theta > h_s + ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # b/a can be simplified as (p-m_ae)/(a_anf + a_ans)
        a = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
        b = (p_exp - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

        # derivative g'(t3) can be calculated manually
        dgt3 = m_ans * (h_s - g_s - theta) / (a_ans * (1 - theta - gamma))

        # which then allows to derive c1 and c2
        s_c1 = ((p_exp - m_ae) / (a_anf + a_ans) - dgt3) * np.exp(a * t_s)
        s_c2 = (-t_s * b + dgt3) / a - (p_exp - m_ae) / ((a_anf + a_ans) * a) + g_s

        def gt(t):
            # general solution for g(t)
            return t * (p_exp - m_ae) / (a_anf + a_ans) + s_c2 + s_c1 / a * np.exp(-a * t)

        def dgt(t):
            # first derivative g'(t)
            return (p_exp - m_ae) / (a_anf + a_ans) - s_c1 * np.exp(-a * t)

        def ht(t):
            # EQ(9) with constants for g(t) and g'(t)
            return a_ans * (1 - theta - gamma) / m_ans * dgt(t) + gt(t) + theta

        h_target = 1 - gamma

        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: h_target - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        return t_end, ht(t_end), gt(t_end)

    @staticmethod
    def work_lAe_fAns(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (
            float, float, float):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # this phase is not applicable if phi is greater or equal to gamma...
        # ... or h is already below pipe exit Ae
        if phi >= gamma or h_s >= 1 - phi:
            return t_s, h_s, g_s

        # g(t4) = g4 can be solved for c
        s_cg = (g_s - (1 - theta - gamma)) * np.exp((m_ans * t_s) / ((1 - theta - gamma) * a_ans))

        def gt(t):
            # generalised g(t) for phase A5
            return (1 - theta - gamma) + s_cg * np.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # as defined for EQ(21)
        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = -m_ae / ((1 - phi) * a_anf)
        g = p_exp / a_anf
        b = m_ans * s_cg / ((1 - theta - gamma) * a_anf)

        # find c that matches h(t4) = h4
        s_ch = (h_s + b / ((a + k) * np.exp(k) ** t_s) + g / a) / np.exp(a) ** t_s

        def ht(t):
            return -b / ((a + k) * np.exp(k) ** t) + s_ch * np.exp(a) ** t - g / a

        h_target = 1 - phi
        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: h_target - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        return t_end, ht(t_end), gt(t_end)

    @staticmethod
    def work_fAe_fAnS(t_s: float, h_s: float, g_s: float, p_exp: float, t_max: float, conf: list) -> (
            float, float, float):
        """
        Final phase before exhaustion.
        :param t_s: time at which A5 ended
        :param h_s: h(t5)
        :param g_s: g(t5)
        :param p_exp: constant power output
        :param t_max: maximal time limit
        :param conf: configuration of hydraulic model
        :return: [t_end: time until h=1, h(t_end)=1, g(t_end)]
        """
        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # this phase is not applicable if Ae or AnS are directly at the bottom of the model
        if phi == 0.0 or gamma == 0.0:
            return t_s, h_s, g_s

        # g(t5) = gt5 can be solved for c
        s_cg = (g_s - (1 - theta - gamma)) / np.exp(-m_ans * t_s / ((1 - theta - gamma) * a_ans))

        def gt(t):
            # generalised g(t) for phase A6
            return (1 - theta - gamma) + s_cg * math.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        k = m_ans / ((1 - theta - gamma) * a_ans)
        # a = -m_ae / a_anf
        b = (m_ans * s_cg) / ((1 - theta - gamma) * a_anf)
        # g = p / a_anf
        ag = (p_exp - m_ae) / a_anf

        # h(t5) = ht5 can be solved for c
        s_ch = -t_s * ag + ((b * math.exp(-k * t_s)) / k) + h_s

        def ht(t):
            # generalised h(t) for phase A6
            return t * ag - ((b * math.exp(-k * t)) / k) + s_ch

        h_target = 1.0
        # find end of phase A6. The time point where h(t_end)=1
        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: h_target - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        return t_end, ht(t_end), gt(t_end)

    @staticmethod
    def rec(conf: list, start_h: float, start_g: float, p_rec: float = 0.0, t_max: float = 5000.0) -> (
            float, float, float):

        # now iterate through all recovery phases
        phases = [ODEThreeCompHydSimulator.rec_a6,
                  ODEThreeCompHydSimulator.rec_a5,
                  ODEThreeCompHydSimulator.rec_a4_r1,
                  ODEThreeCompHydSimulator.rec_a4_r2,
                  ODEThreeCompHydSimulator.rec_a3_r1,
                  ODEThreeCompHydSimulator.rec_a3_r2,
                  ODEThreeCompHydSimulator.rec_a2,
                  ODEThreeCompHydSimulator.rec_a1]

        # start time from 0 and given start fill level
        t = 0
        h, g = start_h, start_g

        # iterate through all phases until end is reached
        for phase in phases:
            t, h, g = phase(t, h, g, p_rec=p_rec, t_max=t_max, conf=conf)

            # if recovery time is reached return fill levels at that point
            if t == t_max:
                return t, h, g

        # if all phases complete full recovery is reached
        return t, h, g

    @staticmethod
    def rec_a6(t6: float, h6: float, g6: float, p_rec: float, t_max: float, conf: list):
        """
        recovery from exhaustive exercise.
        :param t6: time in seconds at which recovery starts
        :param h6: depletion state of AnF when recovery starts
        :param g6: depletion state of AnS when recovery starts
        :param p_rec: constant recovery intensity
        :param t_max: the maximal recovery time
        :param conf: hydraulic model configuration
        :return: [rt5 = min(time at which A6 rec ends, t_rec), h(rt5), g(rt5)]
        """

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # no recovery possible
        if p_rec >= m_ae:
            return np.inf, 1.0, 1.0

        # A6 rec ends either at beginning of A4 or A5
        h_target = max(1 - gamma, 1 - phi)

        # check whether phase is applicable or if h is
        # already above the end of the phase
        if not h6 > h_target:
            return t6, h6, g6

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

        # estimate an initial guess that assumes no contribution from g
        initial_guess = t6
        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: a6_ht(t) - h_target,
                                                  initial_guess=initial_guess,
                                                  max_steps=t_max)
        return t_end, a6_ht(t_end), a6_gt(t_end)

    @staticmethod
    def rec_a5(t5: float, h5: float, g5: float, p_rec: float, t_max: float, conf: list):
        """
        recovery from exhaustive exercise.
        :param t5: time in seconds at which recovery starts
        :param h5: depletion state of AnF when recovery starts
        :param g5: depletion state of AnS when recovery starts
        :param p_rec: constant recovery intensity
        :param t_max: the maximal recovery time
        :param conf: hydraulic model configuration
        :return: [rt4 = min(time at which A5 rec ends, t_rec), h(rt4), g(rt4)]
        """

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # A5 always ends when h reaches pipe exit of AnS
        h_target = 1 - gamma

        # check whether phase is applicable or if h is
        # already above the end of the phase
        if not h5 > h_target:
            return t5, h5, g5

        # g(t5) = g5 can be solved for c
        s_cg = (g5 - (1 - theta - gamma)) * np.exp((m_ans * t5) / ((1 - theta - gamma) * a_ans))

        def a5_gt(t):
            # generalised g(t) for phase A5
            return (1 - theta - gamma) + s_cg * np.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # as defined for EQ(21)
        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = -m_ae / ((1 - phi) * a_anf)
        g = p_rec / a_anf
        b = m_ans * s_cg / ((1 - theta - gamma) * a_anf)

        # find c that matches h(t5) = h5
        s_ch = (h5 + b / ((a + k) * np.exp(k) ** t5) + g / a) / np.exp(a) ** t5

        def a5_ht(t):
            return -b / ((a + k) * np.exp(k) ** t) + s_ch * np.exp(a) ** t - g / a

        # find the time at which the phase stops
        initial_guess = t5
        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: a5_ht(t) - h_target,
                                                  initial_guess=initial_guess,
                                                  max_steps=t_max)
        # return with  fill levels at that time
        return t_end, a5_ht(t_end), a5_gt(t_end)

    @staticmethod
    def rec_a4_r1(t4: float, h4: float, g4: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # A4 R1 is only applicable h below pipe exit of Ae ...
        # ... and if g is above h (allows error of epsilon)
        if not h4 > 1 - phi \
                or not h4 > g4 + theta + ODEThreeCompHydSimulator.eps:
            return t4, h4, g4

        # if g is above h (flow from AnS into AnF)
        a_gh = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
        b_gh = (p_rec - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

        # derivative g'(t4) can be calculated manually
        dgt4_gh = m_ans * (h4 - g4 - theta) / (a_ans * (1 - theta - gamma))

        # ... which then allows to derive c1 and c2
        s_c1_gh = ((p_rec - m_ae) / (a_anf + a_ans) - dgt4_gh) * np.exp(a_gh * t4)
        s_c2_gh = (-t4 * b_gh + dgt4_gh) / a_gh - (p_rec - m_ae) / ((a_anf + a_ans) * a_gh) + g4

        def a4_gt(t):
            # general solution for g(t)
            return t * (p_rec - m_ae) / (a_anf + a_ans) + s_c2_gh + s_c1_gh / a_gh * np.exp(-a_gh * t)

        def a4_dgt(t):
            # first derivative g'(t)
            return (p_rec - m_ae) / (a_anf + a_ans) - s_c1_gh * np.exp(-a_gh * t)

        def a4_ht(t):
            # EQ(9) with constants for g(t) and g'(t)
            return a_ans * (1 - theta - gamma) / m_ans * a4_dgt(t) + a4_gt(t) + theta

        initial_guess = t4

        # phase ends when g drops below h...
        tgth = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_ht(t) - (theta + a4_gt(t)),
                                                 initial_guess=initial_guess,
                                                 max_steps=t_max)

        # ...or if h reaches 1-phi
        tphi = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_ht(t) - (1 - phi),
                                                 initial_guess=initial_guess,
                                                 max_steps=t_max)

        # choose minimal time at which this phase ends
        t_end = min(tphi, tgth)

        return t_end, a4_ht(t_end), a4_gt(t_end)

    @staticmethod
    def rec_a4_r2(t4: float, h4: float, g4: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_anf = conf[4]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # A4 R2 is only applicable if h below pipe exit of Ae...
        # ... and h is above g (error of epsilon tolerated)
        if not h4 > 1 - phi or \
                not h4 < g4 + theta + ODEThreeCompHydSimulator.eps:
            return t4, h4, g4

        # if h is above g simulate flow from AnF into AnS
        a_hg = (a_anf + a_ans) * m_anf / (a_anf * a_ans * (1 - gamma))
        b_hg = (p_rec - m_ae) * m_anf / (a_anf * a_ans * (1 - gamma))

        # derivative g'(t4) can be calculated manually from g4, t4, and h4
        dgt4_hg = - m_anf * (g4 + theta - h4) / (a_ans * (1 - gamma))

        # which then allows to derive c1 and c2
        s_c1_gh = ((p_rec - m_ae) / (a_anf + a_ans) - dgt4_hg) * np.exp(a_hg * t4)
        s_c2_gh = (-t4 * b_hg + dgt4_hg) / a_hg - (p_rec - m_ae) / ((a_anf + a_ans) * a_hg) + g4

        def a4_gt(t):
            # general solution for g(t)
            return t * (p_rec - m_ae) / (a_anf + a_ans) + s_c2_gh + s_c1_gh / a_hg * np.exp(-a_hg * t)

        def a4_dgt(t):
            # first derivative g'(t)
            return (p_rec - m_ae) / (a_anf + a_ans) - s_c1_gh * np.exp(-a_hg * t)

        def a4_ht(t):
            # EQ(16) with constants for g(t) and g'(t)
            return a_ans * (1 - gamma) / m_anf * a4_dgt(t) + a4_gt(t) + theta

        h_target = 1 - phi

        # A4 ends if AnS is completely refilled
        t_fill = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_gt(t),
                                                   initial_guess=t4,
                                                   max_steps=t_max)

        # A4 also ends by surpassing 1 - phi
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_ht(t) - h_target,
                                                  initial_guess=t4,
                                                  max_steps=t_max)

        # choose minimal time at which phase ends
        t_end = min(t_fill, t_phi)

        return t_end, a4_ht(t_end), a4_gt(t_end)

    @staticmethod
    def rec_a3_r1(t3: float, h3: float, g3: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # A3 R1 is only applicable if h is above or at pipe exit of Ae...
        # ... and g is above h (error of epsilon tolerated)
        if not h3 <= 1 - phi + ODEThreeCompHydSimulator.eps \
                or not h3 > g3 + theta + ODEThreeCompHydSimulator.eps:
            return t3, h3, g3

        # my simplified form
        a = m_ae / (a_anf * (1 - phi)) + \
            m_ans / (a_ans * (1 - theta - gamma)) + \
            m_ans / (a_anf * (1 - theta - gamma))

        b = m_ae * m_ans / \
            (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # c' for p_rec
        c = m_ans * (p_rec * (1 - phi) - m_ae * theta) / \
            (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

        # wolfram alpha gave these estimations as solutions for l''(t) + a*l'(t) + b*l(t) = c
        r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
        r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

        # uses Al dt/dl part of EQ(8) solved for c2
        # r1 * c1 * exp(r1*t3) + r2 * c2 * exp(r2*t3) = m_ans * (ht3 - gt3 - theta)) / (a_ans * r2 * (1 - theta - gamma))
        # and then substituted in EQ(14) and solved for c1
        s_c1 = (c / b + (m_ans * (h3 - g3 - theta)) / (a_ans * r2 * (1 - theta - gamma)) - g3) / \
               (np.exp(r1 * t3) * (r1 / r2 - 1))

        # uses EQ(14) with solution for c1 and solves for c2
        s_c2 = (g3 - s_c1 * np.exp(r1 * t3) - c / b) / np.exp(r2 * t3)

        def a3_gt(t):
            # the general solution for g(t)
            return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

        # substitute into EQ(9) for h
        def a3_ht(t):
            k1 = a_ans * (1 - theta - gamma) / m_ans * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - theta - gamma) / m_ans * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # phase A3 R1 only ends if h(t) rises above g(t)
        # find the point where h(t) == g(t)
        # As h(t3) is assumed to be below g(t3) and AnS is limited, this point must be reached
        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: a3_ht(t) - (a3_gt(t) + theta),
                                                  initial_guess=t3,
                                                  max_steps=t_max)
        return t_end, a3_ht(t_end), a3_gt(t_end)

    @staticmethod
    def rec_a3_r2(t3: float, h3: float, g3: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_anf = conf[4]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # A3 R2 is only applicable if h is above or at pipe exit of Ae...
        # ... and h is above g (error of epsilon tolerated)...
        # ... and g is not 0
        if not h3 <= 1 - phi + ODEThreeCompHydSimulator.eps \
                or not h3 < g3 + theta + ODEThreeCompHydSimulator.eps \
                or not g3 > 0.0:
            return t3, h3, g3

        # EQ 16 and 17 substituted in EQ 8
        a = m_ae / (a_anf * (1 - phi)) + \
            m_anf / (a_ans * (1 - gamma)) + \
            m_anf / (a_anf * (1 - gamma))

        b = m_ae * m_anf / \
            (a_anf * a_ans * (1 - phi) * (1 - gamma))

        c = m_anf * (p_rec * (1 - phi) - m_ae * theta) / \
            (a_anf * a_ans * (1 - phi) * (1 - gamma))

        # solutions for l''(t) + a*l'(t) + b*l(t) = c
        r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
        r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

        # uses Al dt/dl part of EQ(16) == dl/dt of EQ(14) solved for c2
        # and then substituted in EQ(14) and solved for c1
        s_c1 = (c / b - (m_anf * (g3 + theta - h3)) / (a_ans * r2 * (1 - gamma)) - g3) / \
               (np.exp(r1 * t3) * (r1 / r2 - 1))

        # uses EQ(14) with solution for c1 and solves for c2
        s_c2 = (g3 - s_c1 * np.exp(r1 * t3) - c / b) / np.exp(r2 * t3)

        def a3_gt(t):
            # the general solution for g(t)
            return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

        # substitute into EQ(9) for h
        def a3_ht(t):
            k1 = a_ans * (1 - gamma) / m_anf * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - gamma) / m_anf * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # Recovery phase A3R2 only ends if AnS is refilled
        if a3_gt(t_max) > 0:
            return t_max, a3_ht(t_max), a3_gt(t_max)
        else:
            # find the point where g(t) == 0
            t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: a3_gt(t),
                                                      initial_guess=t3,
                                                      max_steps=t_max)
            return t_end, a3_ht(t_end), a3_gt(t_end)

    @staticmethod
    def rec_a2(t2: float, h2: float, g2: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        m_ae = conf[2]
        phi = conf[7]

        # Recovery phase A2 is only applicable if h is below pipe exit of Ae and AnS is full
        if 1 - phi > h2 or g2 > ODEThreeCompHydSimulator.eps:
            return t2, h2, g2

        def a2_ht(t):
            return h2 - (t - t2) * (m_ae - p_rec) / a_anf

        # the total duration of recovery phase A2 from t2 on
        t_end = (h2 - 1 + phi) * a_anf / (m_ae - p_rec) + t2

        # check if targeted recovery time is before phase end time
        t_end = min(t_end, t_max)

        return t_end, a2_ht(t_end), g2

    @staticmethod
    def rec_a1(t1: float, h1: float, g1: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        m_ae = conf[2]
        phi = conf[7]

        def a1_ht(t):
            return (h1 - p_rec * (1 - phi) / m_ae) * \
                   np.exp(m_ae * (t1 - t) / (a_anf * (1 - phi))) + \
                   p_rec * (1 - phi) / m_ae

        # full recovery can't be reached as log(inf) only approximates 0
        # a1_ht(max rec time) is the most recovery possible
        target_h = a1_ht(t_max)

        return t_max, target_h, g1

        # # return min(h) if even after the maximal recovery time h>epsilon
        # if target_h > ODEThreeCompHydSimulator.eps:
        #     return t_rec, target_h, g1
        #
        # # otherwise return time when h(t) reaches approximately 0 (epsilon)
        # t_end = a_anf * (1 - phi) / - m_ae * (
        #         np.log(target_h - p_rec * (1 - phi) / m_ae) - (
        #         np.log(h1 - p_rec * (1 - phi) / m_ae) + m_ae * t1 / (a_anf * (1 - phi))
        # )
        # )
        #
        # return t_end, a1_ht(t_end), g1

    @staticmethod
    def optimize(func, initial_guess, max_steps):
        """
        This optimiser finds t with func(t) == 0 by increasing t at increasing precision
        until the sing switches (func(t) becomes negative). It is assumed that func(t0 = initial guess) is positive.
        If it becomes clear that t>=max_steps, max_steps is returned.
        :return: found t
        """

        # start precision
        step_size = 1000.0

        # check if initial guess conforms to underlying optimizer assumption
        t = initial_guess
        if func(t) < 0:
            raise UserWarning("initial guess for func is not positive")

        # while maximal precision is not reached
        while step_size > 0.0000001:
            t_p = t
            # increase t until function turns negative
            while t <= max_steps + 1 and func(t) >= 0:
                t_p = t
                t += step_size

            # back to the one before values turned negative
            t = t_p
            # if t is above defined max steps we know our target is beyond the time limit
            if t >= max_steps:
                return max_steps

            # increase precision
            step_size = step_size / 10.0

        return t
