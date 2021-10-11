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
            # Exhaustion not reached during TTE
            return 200

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

        # start with fully reset agent
        t, h, g = 0, start_h, start_g

        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        func = None
        while t < t_max:
            if func is None:
                # first distinguish between fAe and lAe
                if h >= 1 - phi:
                    # fAe
                    if h <= theta and g < ODEThreeCompHydSimulator.eps:
                        func = ODEThreeCompHydSimulator.fAe
                    # fAe_rAnS
                    elif h < g + theta and g > ODEThreeCompHydSimulator.eps:
                        func = ODEThreeCompHydSimulator.fAe_rAn
                    # fAe_lAnS
                    elif h >= g + theta and h <= 1 - gamma:
                        func = ODEThreeCompHydSimulator.fAe_lAn
                    # fAe_fAnS
                    elif h >= 1 - gamma:
                        func = ODEThreeCompHydSimulator.fAe_fAn
                    else:
                        raise UserWarning(
                            "unhandled state with h {} g {} and conf theta {} gamma {} phi {}".format(h, g, theta,
                                                                                                      gamma,
                                                                                                      phi))
                else:
                    # lAr
                    if h <= theta and g < ODEThreeCompHydSimulator.eps:
                        func = ODEThreeCompHydSimulator.lAe
                    elif h < g + theta and g > ODEThreeCompHydSimulator.eps:
                        func = ODEThreeCompHydSimulator.lAe_rAn
                    elif h >= g + theta and h <= 1 - gamma:
                        func = ODEThreeCompHydSimulator.lAe_lAn
                    elif h >= 1 - gamma:
                        func = ODEThreeCompHydSimulator.lAe_fAn
                    else:
                        raise UserWarning(
                            "unhandled state with h {} g {} and conf theta {} gamma {} phi {}".format(h, g, theta,
                                                                                                      gamma,
                                                                                                      phi))

            # iterate through all phases until end is reached
            t, h, g, n_func = func(t, h, g, p_exp, t_max=t_max, conf=conf)
            func = n_func

            # if recovery time is reached return fill levels at that point
            if t == np.nan or n_func is None:
                return t, h, g

        # if all phases complete full exhaustion is reached
        return t, h, g

    @staticmethod
    def lAe(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):
        """
        The phase l_Ae where only Ae contributes and flow through p_Ae is limited by liquid pressure.
        """

        a_anf = conf[0]
        m_ae = conf[2]
        theta = conf[5]
        phi = conf[7]

        # phase ends at bottom of Ae or top of AnS
        h_bottom = min(theta, 1 - phi)

        # This phase is not applicable if fill-level of AnF below pipe exit Ae or top of AnS, ..
        # ... or AnS not full ...
        # ... or pipe exit of Ae is at the top of the model

        # in some exp equations values exceed float capacity if t gets too large
        # Therefore, t_s is set to 0 and added later
        t_p = t_s
        t_max = t_max - t_s
        t_s = 0

        # constant can be derived from known t_s and h_s
        c1 = (h_s - p * (1 - phi) / m_ae) * np.exp(-m_ae * t_s / (a_anf * (phi - 1)))

        # general solution for h(t) using c1
        ht_max = p * (1 - phi) / m_ae + c1 * np.exp(m_ae * t_max / (a_anf * (phi - 1)))

        # check if max time is reached in this phase
        # for example this will always happen during recovery as in h, log(inf) only approximates 0
        if ht_max <= h_bottom:
            return t_p + t_max, ht_max, g_s, None
        else:
            # end of phase lAe -> the time when h(t) = min(theta,1-phi)
            t_end = a_anf * (phi - 1) / m_ae * np.log((h_bottom - p * (1 - phi) / m_ae) / c1)
            if h_bottom == theta:
                # phase transfers into lAe_lAn
                func = ODEThreeCompHydSimulator.lAe_lAn
            else:
                # phase transfers int fAe
                func = ODEThreeCompHydSimulator.fAe
            return t_p + t_end, h_bottom, g_s, func

    @staticmethod
    def lAe_rAn(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_anf = conf[4]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # This phase is not applicable if fill-level of AnF below pipe exit Ae, ..
        # ... or AnS fill-level is above AnF fill-level ...
        # ... or AnS is full ...
        # ... or pipe exit of Ae is at the top of the model

        # in some exp equations values exceed float value range if t gets too large
        # Therefore, t_s is set to 0 and added later
        t_p = t_s
        t_max = t_max - t_s
        t_s = 0

        # EQ 16 and 17 substituted in EQ 8
        a = m_ae / (a_anf * (1 - phi)) + \
            m_anf / (a_ans * (1 - gamma)) + \
            m_anf / (a_anf * (1 - gamma))

        b = m_ae * m_anf / \
            (a_anf * a_ans * (1 - phi) * (1 - gamma))

        c = m_anf * (p * (1 - phi) - m_ae * theta) / \
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

        # determine what the current phase lAe_rAn transitions into
        if t_end >= t_max:
            func = None
        elif t_end == g0:
            func = ODEThreeCompHydSimulator.lAe
        elif t_end == gtht:
            func = ODEThreeCompHydSimulator.lAe_lAn
        else:
            func = ODEThreeCompHydSimulator.fAe_rAn

        return t_p + t_end, ht(t_end), gt(t_end), func

    @staticmethod
    def fAe(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):
        """
        The phase f_Ae where only Ae contributes and flow through p_Ae is at maximal capacity m_Ae. This phase is linear
        and does not allow an equilibrium because power p is assumed to be constant.
        """
        a_anf = conf[0]
        m_ae = conf[2]
        theta = conf[5]
        phi = conf[7]

        # this phase not applicable if h is not in-between 1-theta and phi and ...
        # ... AnS is not full

        # the first derivative. Change in ht
        ht_p = (p - m_ae) / a_anf

        if ht_p > 0:
            # expenditure
            h_target = theta
            func = ODEThreeCompHydSimulator.fAe_lAn
        elif ht_p < 0:
            # recovery ends at 1 - phi
            h_target = 1 - phi
            func = ODEThreeCompHydSimulator.lAe
        else:
            # no change
            return t_max, h_s, g_s, None

            # linear utilization -> no equilibrium possible
        t_end = (h_target - h_s) * a_anf / (p - m_ae) + t_s

        # check if max time is reached before phase end
        if t_end > t_max:
            h_end = h_s + (t_max - t_s) * ht_p
            return t_max, h_end, g_s, None
        else:
            return t_end, h_target, g_s, func

    @staticmethod
    def fAe_rAn(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_anf = conf[4]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # This phase only starts if is not applicable if fill-level of AnF above or at pipe exit Ae, ..
        # ... or AnS fill-level is above AnF fill-level ...
        # ... or AnS is full

        # in some exp equations values exceed float capacity if t gets too large
        # Therefore, t_s is set to 0 and added later
        t_p = t_s
        t_max = t_max - t_s
        t_s = 0

        # if h is above g (flow from AnF into AnS)
        a_hg = (a_anf + a_ans) * m_anf / (a_anf * a_ans * (1 - gamma))
        b_hg = (p - m_ae) * m_anf / (a_anf * a_ans * (1 - gamma))

        # derivative g'(t) can be calculated manually
        dgt_hg = - m_anf * (g_s + theta - h_s) / (a_ans * (1 - gamma))

        # which then allows to derive c1 and c2
        s_c1_gh = ((p - m_ae) / (a_anf + a_ans) - dgt_hg) * np.exp(a_hg * t_s)
        s_c2_gh = (-t_s * b_hg + dgt_hg) / a_hg - (p - m_ae) / ((a_anf + a_ans) * a_hg) + g_s

        def gt(t):
            # general solution for g(t)
            return t * (p - m_ae) / (a_anf + a_ans) + s_c2_gh + s_c1_gh / a_hg * np.exp(-a_hg * t)

        def dgt(t):
            # first derivative g'(t)
            return (p - m_ae) / (a_anf + a_ans) - s_c1_gh * np.exp(-a_hg * t)

        def ht(t):
            # EQ(16) with constants for g(t) and g'(t)
            return a_ans * (1 - gamma) / m_anf * dgt(t) + gt(t) + theta

        # find the point where AnS is full g(t) == 0
        t_g0 = ODEThreeCompHydSimulator.optimize(func=lambda t: gt(t),
                                                 initial_guess=t_s,
                                                 max_steps=t_max)

        # find the point where fill-levels AnS and AnF are at equal
        t_gtht = ODEThreeCompHydSimulator.optimize(func=lambda t: gt(t) + theta - ht(t),
                                                   initial_guess=t_s,
                                                   max_steps=t_max)

        # ... fAe_rAn also ends by surpassing 1 - phi
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - (1 - phi),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        t_end = min(t_g0, t_gtht, t_phi)

        # determine what the current phase fAe_rAn transitions into
        if t_end >= t_max:
            func = None
        elif t_end == t_g0:
            func = ODEThreeCompHydSimulator.fAe
        elif t_end == t_gtht:
            func = ODEThreeCompHydSimulator.fAe_lAn
        else:
            func = ODEThreeCompHydSimulator.lAe_rAn

        return t_end + t_p, ht(t_end), gt(t_end), func

    @staticmethod
    def lAe_lAn(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # lAe and lAn is only applicable if h is above or at pipe exit of Ae...
        # ... and above or at pipe exit AnS ...
        # ... and g is above h (error of epsilon tolerated)

        # in some exp equations values exceed float capacity if t gets too large
        # Therefore, t_s is set to 0 and added later
        t_p = t_s
        t_max = t_max - t_s
        t_s = 0

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

        # uses Al dt/dl part of EQ(8) solved for c2
        # r1 * c1 * exp(r1*t3) + r2 * c2 * exp(r2*t3) = m_ans * (ht3 - gt3 - theta)) / (a_ans * r2 * (1 - theta - gamma))
        # and then substituted in EQ(14) and solved for c1
        # ... or ...
        # uses Al dt/dl part of EQ(9) == dl/dt of EQ(12) solved for c2
        # and then substituted in EQ(12) and solved for c1
        s_c1 = (g_s - m_ans * (h_s - g_s - theta) / (a_ans * r2 * (1 - theta - gamma)) - c / b) / (
                np.exp(r1 * t_s) * (1 - r1 / r2))
        # uses EQ(12) with solution for c1 and solves for c2
        # ... or uses EQ(14) with solution for c1 and solves for c2
        s_c2 = (g_s - s_c1 * np.exp(r1 * t_s) - c / b) / np.exp(r2 * t_s)

        def gt(t):
            # the general solution for g(t)
            return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

        # substitute into EQ(9) for h
        def ht(t):
            k1 = a_ans * (1 - theta - gamma) / m_ans * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - theta - gamma) / m_ans * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # find the point where h(t) == g(t) ...
        t_gh = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - (gt(t) + theta),
                                                 initial_guess=t_s,
                                                 max_steps=t_max)

        # ... or where h(t) reaches 1 - gamma, changing lAn to fAn
        t_gam = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - gamma - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # ... or where h(t) drops back to 1 - phi, changing lAe to fAe
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - phi - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        t_end = min(t_gh, t_gam, t_phi)

        # Determine what the current phase lAe_lAn transitions into
        if t_end >= t_max:
            func = None
        elif t_end == t_gh:
            func = ODEThreeCompHydSimulator.lAe_rAn
        elif t_end == t_gam:
            func = ODEThreeCompHydSimulator.lAe_fAn
        else:
            func = ODEThreeCompHydSimulator.fAe_lAn

        return t_end + t_p, ht(t_end), gt(t_end), func

    @staticmethod
    def fAe_lAn(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # phase full Ae and limited AnS is only applicable h below pipe exit of Ae
        # ... and above pipe exist of AnS ...
        # ... and if g is above h (allows error of epsilon)

        # in some exp equations values exceed float capacity if t gets too large
        # Therefore, t_s is set to 0 and added later
        t_p = t_s
        t_max = t_max - t_s
        t_s = 0

        # if g is above h (flow from AnS into AnF)
        a_gh = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
        b_gh = (p - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

        # derivative g'(t) can be calculated manually
        dgt_gh = m_ans * (h_s - g_s - theta) / (a_ans * (1 - theta - gamma))

        # ... which then allows to derive c1 and c2
        s_c1_gh = ((p - m_ae) / (a_anf + a_ans) - dgt_gh) * np.exp(a_gh * t_s)
        s_c2_gh = (-t_s * b_gh + dgt_gh) / a_gh - (p - m_ae) / ((a_anf + a_ans) * a_gh) + g_s

        def gt(t):
            # general solution for g(t)
            return t * (p - m_ae) / (a_anf + a_ans) + s_c2_gh + s_c1_gh / a_gh * np.exp(-a_gh * t)

        def dgt(t):
            # first derivative g'(t)
            return (p - m_ae) / (a_anf + a_ans) - s_c1_gh * np.exp(-a_gh * t)

        def ht(t):
            # EQ(9) with constants for g(t) and g'(t)
            return a_ans * (1 - theta - gamma) / m_ans * dgt(t) + gt(t) + theta

        # phase ends when g == h...
        t_gth = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - (theta + gt(t)),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # ...or if h drops to 1-gamma changing lAn into fAn
        t_gam = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - gamma - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # ...or if h reaches 1-phi changing fAe into lAe
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - (1 - phi),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # choose minimal time at which this phase ends
        t_end = min(t_phi, t_gth, t_gam)

        # Determine what the current phase fAe_lAn transitions into
        if t_end >= t_max:
            func = None
        elif t_end == t_gth:
            func = ODEThreeCompHydSimulator.fAe_rAn
        elif t_end == t_gam:
            func = ODEThreeCompHydSimulator.fAe_fAn
        else:
            func = ODEThreeCompHydSimulator.lAe_lAn

        return t_end + t_p, ht(t_end), gt(t_end), func

    @staticmethod
    def lAe_fAn(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):
        """
        :param t_s: time in seconds at which phase starts
        :param h_s: depletion state of AnF when phase starts
        :param g_s: depletion state of AnS when phase starts
        :param p: constant intensity
        :param t_max: the maximal recovery time
        :param conf: hydraulic model configuration
        :return:
        """

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # this phase is not applicable if phi is greater or equal to gamma...
        # ... or h is already below pipe exit Ae

        # in some exp equations values exceed float capacity if t gets too large
        # Therefore, t_s is set to 0 and added later
        t_p = t_s
        t_max = t_max - t_s
        t_s = 0

        # g(t_s) = g_s can be solved for c
        s_cg = (g_s - (1 - theta - gamma)) * np.exp((m_ans * t_s) / ((1 - theta - gamma) * a_ans))

        # generalised g(t)
        def gt(t):
            return (1 - theta - gamma) + s_cg * np.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # as defined for EQ(21)
        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = -m_ae / ((1 - phi) * a_anf)
        g = p / a_anf
        b = m_ans * s_cg / ((1 - theta - gamma) * a_anf)

        # find c that matches h(t_s) = h_s
        s_ch = (h_s + b / ((a + k) * np.exp(k) ** t_s) + g / a) / np.exp(a) ** t_s

        def ht(t):
            return -b / ((a + k) * np.exp(k) ** t) + s_ch * np.exp(a) ** t - g / a

        # find the time at which the phase stops
        t_gam = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - (1 - gamma),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # phase also ends if h drops back to 1-phi changing lAe into fAe
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: (1 - phi) - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        t_end = min(t_gam, t_phi)

        # Determine what the current phase lAe_lAn transitions into
        if t_end >= t_max:
            func = None
        elif t_end == t_gam:
            func = ODEThreeCompHydSimulator.lAe_lAn
        else:
            func = ODEThreeCompHydSimulator.fAe_fAn

        return t_p + t_end, ht(t_end), gt(t_end), func

    @staticmethod
    def fAe_fAn(t_s: float, h_s: float, g_s: float, p: float, t_max: float, conf: list):
        """
        :param t_s:
        :param h_s:
        :param g_s:
        :param p: constant power output
        :param t_max: maximal time limit
        :param conf: configuration of hydraulic model
        :return:
        """
        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # this phase is not applicable if Ae or AnS are directly at the bottom of the model

        # in some exp equations values exceed float capacity if t gets too large
        # Therefore, t_s is set to 0 and added later
        t_p = t_s
        t_max = t_max - t_s
        t_s = 0

        # g(t_s) = g_s can be solved for c
        s_cg = (g_s - (1 - theta - gamma)) / np.exp(-m_ans * t_s / ((1 - theta - gamma) * a_ans))

        # generalised g(t)
        def gt(t):
            return (1 - theta - gamma) + s_cg * np.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        k = m_ans / ((1 - theta - gamma) * a_ans)
        # a = -m_ae / a_anf
        b = (m_ans * s_cg) / ((1 - theta - gamma) * a_anf)
        # g = p / a_anf
        ag = (p - m_ae) / a_anf

        # h(t_s) = h_s can be solved for c
        s_ch = -t_s * ag + ((b * np.exp(-k * t_s)) / k) + h_s

        # generalised h(t)
        def ht(t):
            return t * ag - ((b * np.exp(-k * t)) / k) + s_ch

        # expenditure: find the time point where h(t_end)=1
        t_exp = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        # recovery
        h_target = max(1 - gamma, 1 - phi)
        t_rec = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - h_target,
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        t_end = min(t_exp, t_rec)
        # Determine what the current phase fAe_fAn transitions into
        if t_end >= t_max:
            func = None
        elif t_end == t_rec:
            if h_target == 1 - gamma:
                func = ODEThreeCompHydSimulator.fAe_lAn
            else:
                func = ODEThreeCompHydSimulator.lAe_fAn
        else:
            func = None

        return t_p + t_end, ht(t_end), gt(t_end), func

    @staticmethod
    def rec(conf: list, start_h: float, start_g: float, p_rec: float = 0.0, t_max: float = 5000.0) -> (
            float, float, float):

        # all recovery phases in order
        phases = [ODEThreeCompHydSimulator.rec_fAe_fAnS,
                  ODEThreeCompHydSimulator.rec_lAe_fAnS,
                  ODEThreeCompHydSimulator.fAe_lAn,
                  ODEThreeCompHydSimulator.fAe_rAn,
                  ODEThreeCompHydSimulator.lAe_lAn,
                  ODEThreeCompHydSimulator.lAe_rAn,
                  ODEThreeCompHydSimulator.fAe,
                  ODEThreeCompHydSimulator.lAe]

        # start time from 0 and given start fill level
        t = 0
        h, g = start_h, start_g

        # cycle through phases until t_max is reached
        while t < t_max:
            for phase in phases:
                t, h, g = phase(t, h, g, p_rec, t_max=t_max, conf=conf)

                # if recovery time is reached return fill levels at that point
                if t == np.nan or t >= t_max:
                    return t, h, g

        # if all phases complete full recovery is reached
        return t, h, g

    @staticmethod
    def rec_fAe_fAnS(t_s: float, h_s: float, g_s: float, p_rec: float, t_max: float, conf: list):
        """
        recovery from exhaustive exercise.
        :param t_s: time in seconds at which recovery starts
        :param h_s: depletion state of AnF when recovery starts
        :param g_s: depletion state of AnS when recovery starts
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

        # A6 rec ends either at beginning of A4 or A5
        h_target = max(1 - gamma, 1 - phi)

        # check whether phase is applicable or if h is
        # already at the end of the phase
        if abs(h_s - h_target) < ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # g(t_s) = g_s can be solved for c
        s_cg = (g_s - (1 - theta - gamma)) / np.exp(-m_ans * t_s / ((1 - theta - gamma) * a_ans))

        # generalised g(t)
        def gt(t):
            return (1 - theta - gamma) + s_cg * np.exp(-m_ans * t / ((1 - theta - gamma) * a_ans))

        k = m_ans / ((1 - theta - gamma) * a_ans)
        # a = -m_ae / a_anf
        b = m_ans * s_cg / ((1 - theta - gamma) * a_anf)
        # g = p / a_anf
        ag = (p_rec - m_ae) / a_anf

        # h(t_s) = h_s can be solved for c
        s_ch = -t_s * ag + b * math.exp(-k * t_s) / k + h_s

        # generalised h(t)
        def ht(t):
            return t * ag - b * math.exp(-k * t) / k + s_ch

        # estimate an initial guess that assumes no contribution from g
        t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - h_target,
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        h_end = min(ht(t_end), 1.0)
        return t_end, h_end, gt(t_end)

    @staticmethod
    def rec_lAe_fAnS(t_s: float, h_s: float, g_s: float, p_rec: float, t_max: float, conf: list):
        """
        recovery from exhaustive exercise.
        :param t_s: time in seconds at which recovery starts
        :param h_s: depletion state of AnF when recovery starts
        :param g_s: depletion state of AnS when recovery starts
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

        # in this recovery phase h is assumed to continuously increase
        # this recovery phase ends when h reaches pipe exit of AnS
        h_target = 1 - gamma

        # this phase is not applicable if phi is greater or equal to gamma...
        # ... or if h is already above the end of the phase
        if phi >= gamma or h_s <= h_target - ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # g(t_s) = g_s can be solved for c
        s_cg = (g_s - (1 - theta - gamma)) * np.exp((m_ans * t_s) / ((1 - theta - gamma) * a_ans))

        # generalised g(t)
        def gt(t):
            return (1 - theta - gamma) + s_cg * np.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))

        # as defined for EQ(21)
        k = m_ans / ((1 - theta - gamma) * a_ans)
        a = -m_ae / ((1 - phi) * a_anf)
        g = p_rec / a_anf
        b = m_ans * s_cg / ((1 - theta - gamma) * a_anf)

        # find c that matches h(t_s) = h_s
        s_ch = (h_s + b / (a + k) * np.exp(-k * t_s) + g / a) * np.exp(-a * t_s)

        def ht(t):
            return -b / (a + k) * np.exp(-k * t) + s_ch * np.exp(a * t) - g / a

        # find the time at which the phase stops
        t_ht = ODEThreeCompHydSimulator.optimize(func=lambda t: ht(t) - h_target,
                                                 initial_guess=t_s,
                                                 max_steps=t_max)

        # phase also ends if h drops back to 1-phi changing lAe into fAe
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: (1 - phi) - ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)
        t_end = min(t_ht, t_phi)
        return t_end, ht(t_end), gt(t_end)

    @staticmethod
    def rec_fAe_lAnS(t_s: float, h_s: float, g_s: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # phase full Ae and limited AnS is only applicable h below pipe exit of Ae
        # ... and above AnS ...
        # ... and if g is above h (allows error of epsilon)
        if h_s > 1 - gamma + ODEThreeCompHydSimulator.eps \
                or h_s < 1 - phi - ODEThreeCompHydSimulator.eps \
                or h_s < g_s + theta - ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # if g is above h (flow from AnS into AnF)
        a_gh = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
        b_gh = (p_rec - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

        # derivative g'(t4) can be calculated manually
        dgt4_gh = m_ans * (h_s - g_s - theta) / (a_ans * (1 - theta - gamma))

        # ... which then allows to derive c1 and c2
        s_c1_gh = ((p_rec - m_ae) / (a_anf + a_ans) - dgt4_gh) * np.exp(a_gh * t_s)
        s_c2_gh = (-t_s * b_gh + dgt4_gh) / a_gh - (p_rec - m_ae) / ((a_anf + a_ans) * a_gh) + g_s

        def a4_gt(t):
            # general solution for g(t)
            return t * (p_rec - m_ae) / (a_anf + a_ans) + s_c2_gh + s_c1_gh / a_gh * np.exp(-a_gh * t)

        def a4_dgt(t):
            # first derivative g'(t)
            return (p_rec - m_ae) / (a_anf + a_ans) - s_c1_gh * np.exp(-a_gh * t)

        def a4_ht(t):
            # EQ(9) with constants for g(t) and g'(t)
            return a_ans * (1 - theta - gamma) / m_ans * a4_dgt(t) + a4_gt(t) + theta

        initial_guess = t_s

        # phase ends when g drops below h...
        t_gth = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_ht(t) - (theta + a4_gt(t)),
                                                  initial_guess=initial_guess,
                                                  max_steps=t_max)

        # ...or if h reaches 1-phi changing fAe into lAe
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_ht(t) - (1 - phi),
                                                  initial_guess=initial_guess,
                                                  max_steps=t_max)

        # ...or if h drops to 1-gamma changing lAnS into fAnS
        t_gam = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - gamma - a4_ht(t),
                                                  initial_guess=initial_guess,
                                                  max_steps=t_max)

        # choose minimal time at which this phase ends
        t_end = min(t_phi, t_gth, t_gam)

        return t_end, a4_ht(t_end), a4_gt(t_end)

    @staticmethod
    def rec_fAe_rAnS(t_s: float, h_s: float, g_s: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_anf = conf[4]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # A4 R2 is only applicable if h below pipe exit of Ae...
        # ... and h is above g (error of epsilon tolerated)
        if h_s <= 1 - phi or \
                h_s >= g_s + theta + ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

        # if h is above g simulate flow from AnF into AnS
        a_hg = (a_anf + a_ans) * m_anf / (a_anf * a_ans * (1 - gamma))
        b_hg = (p_rec - m_ae) * m_anf / (a_anf * a_ans * (1 - gamma))

        # derivative g'(t4) can be calculated manually from g4, t4, and h4
        dgt4_hg = - m_anf * (g_s + theta - h_s) / (a_ans * (1 - gamma))

        # which then allows to derive c1 and c2
        s_c1_gh = ((p_rec - m_ae) / (a_anf + a_ans) - dgt4_hg) * np.exp(a_hg * t_s)
        s_c2_gh = (-t_s * b_hg + dgt4_hg) / a_hg - (p_rec - m_ae) / ((a_anf + a_ans) * a_hg) + g_s

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
                                                   initial_guess=t_s,
                                                   max_steps=t_max)

        # A4 also ends by surpassing 1 - phi
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_ht(t) - h_target,
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # choose minimal time at which phase ends
        t_end = min(t_fill, t_phi)

        return t_end, a4_ht(t_end), a4_gt(t_end)

    @staticmethod
    def rec_lAe_lAnS(t_s: float, h_s: float, g_s: float, p_rec: float, t_max: float, conf: list):

        a_anf = conf[0]
        a_ans = conf[1]
        m_ae = conf[2]
        m_ans = conf[3]
        theta = conf[5]
        gamma = conf[6]
        phi = conf[7]

        # lAe and lAnS is only applicable if h is above or at pipe exit of Ae...
        # ... and above or at pipe exit AnS ...
        # ... and g is above h (error of epsilon tolerated)
        if h_s > 1 - phi + ODEThreeCompHydSimulator.eps \
                or h_s > 1 - gamma + ODEThreeCompHydSimulator.eps \
                or h_s < g_s + theta - ODEThreeCompHydSimulator.eps:
            return t_s, h_s, g_s

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
        s_c1 = (c / b + (m_ans * (h_s - g_s - theta)) / (a_ans * r2 * (1 - theta - gamma)) - g_s) / \
               (np.exp(r1 * t_s) * (r1 / r2 - 1))

        # uses EQ(14) with solution for c1 and solves for c2
        s_c2 = (g_s - s_c1 * np.exp(r1 * t_s) - c / b) / np.exp(r2 * t_s)

        def a3_gt(t):
            # the general solution for g(t)
            return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

        # substitute into EQ(9) for h
        def a3_ht(t):
            k1 = a_ans * (1 - theta - gamma) / m_ans * s_c1 * r1 + s_c1
            k2 = a_ans * (1 - theta - gamma) / m_ans * s_c2 * r2 + s_c2
            return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

        # find the point where h(t) == g(t) ...
        t_gh = ODEThreeCompHydSimulator.optimize(func=lambda t: a3_ht(t) - (a3_gt(t) + theta),
                                                 initial_guess=t_s,
                                                 max_steps=t_max)

        # ... or where h(t) drops back to 1 - gamma, changing lAnS to fAnS
        t_gam = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - gamma - a3_ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        # ... or where h(t) drops back to 1 - phi, changing lAe to fAe
        t_phi = ODEThreeCompHydSimulator.optimize(func=lambda t: 1 - phi - a3_ht(t),
                                                  initial_guess=t_s,
                                                  max_steps=t_max)

        t_end = min(t_gh, t_gam, t_phi)
        return t_end, a3_ht(t_end), a3_gt(t_end)

    @staticmethod
    def optimize(func, initial_guess, max_steps):
        """
        This optimiser finds t with func(t) == 0 by increasing t at increasing precision
        until the sing switches (func(t) becomes negative). It is assumed that func(t0 = initial guess) is positive.
        If it becomes clear that t>=max_steps, max_steps is returned.
        :return: found t
        """

        # start precision
        step_size = 10.0
        step_min = 0.0000001

        # check if initial guess conforms to underlying optimizer assumption
        t = initial_guess
        if func(t) < 0:
            # if not, the value might just be slightly off.
            # Check if an increase fixes it
            check = step_size
            if func(t + check) >= 0:
                # find minimal step size that fixes it
                while func(t + check) >= 0:
                    check = check / 10.0
                t += check * 10.0  # undo the last step
            else:
                raise UserWarning("initial guess for func is not positive. Tried with \n"
                                  "f(t) = {} \n "
                                  "f(t+check) = {}".format(
                    func(t),
                    func(t + check)))

        # while maximal precision is not reached
        while step_size > step_min:
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
