import logging
import math

import numpy as np
from scipy import optimize
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation


def phase_a1():
    # end of phase A1 -> the time when h(t)=theta
    t1 = (-(a_anf * (1 - phi)) / m_ae) * math.log(1 - ((m_ae * the) / (p * (1 - phi))))
    return t1


def phase_a2(t1):
    # taken from Equation 11 by Morton 1986
    a = (m_ae * a_ans * (1 - the - gam) + m_ans * (a_anf + a_ans) * (1 - phi)) / (
            a_anf * a_ans * (1 - phi) * (1 - the - gam))

    b = (m_ae * m_ans) / (a_anf * a_ans * (1 - phi) * (1 - the - gam))

    c = (m_ans * (p * (1 - phi) - m_ae * the)) / (a_anf * a_ans * (1 - phi) * (1 - the - gam))

    # use auxillary equation to derive both negative roots r1 and r2
    coeff = [1, a, b]
    r1, r2 = np.roots(coeff)

    def a2_lt(c1, c2, t):
        # the general solution for l(t)
        return c1 * math.exp(r1 * t) + c2 * math.exp(r2 * t) + c / b

    def a2_dlt(c1, c2, t):
        # derivative of general solution
        return r1 * c1 * math.exp(r1 * t) + r2 * c2 * math.exp(r2 * t)

    # l(t1) and dl(t1) equal to 0. Find c1 and c2 that satisfy that
    res = optimize.fsolve(lambda cx: (a2_lt(cx[0], cx[1], t1), a2_dlt(cx[0], cx[1], t1)), x0=np.array([1, 1]))
    s_c1 = float(res[0])
    s_c2 = float(res[1])

    # substitute into EQ(9) for h
    def a2_ht(t):
        return (a_ans * (1 - the - gam)) / m_ans * a2_dlt(s_c1, s_c2, t) + a2_lt(s_c1, s_c2, t) + the

    if phi < gam:
        # the whole of AnS is utilized in phase A2
        t2 = optimize.fsolve(lambda t: abs((1 - gam) - a2_ht(t)), x0=np.array([1]))[0]
    else:
        # phase A2 transitions into phase A3 before AnS is empty
        t2 = optimize.fsolve(lambda t: abs((1 - phi) - a2_ht(t)), x0=np.array([1]))[0]

    # use t2 to get l(t2)
    return t2, a2_lt(s_c1, s_c2, t2)


def phase_a3(t2, lt2):
    # phase A3 is not applicable if gamma is greater or equal to phi
    if gam >= phi:
        return t2, lt2

    a = ((a_anf + a_ans) * m_ans) / (a_anf * a_ans * (1 - the - gam))
    b = ((p - m_ae) * m_ans) / (a_anf * a_ans * (1 - the - gam))

    def a3_lt(c1, c2, t):
        # general solution for l(t)
        return (((b * t) + (c1 * math.exp(-a * t))) / a) + c2

    def a3_dlt(c1, c2, t):
        # first derivative l'(t)
        return (b / a) - c1 * math.exp(-a * t)

    # derivative l'(t2) can be calculated manually
    dlt2 = (m_ans * ((1 - phi) - lt2 - the)) / (a_ans * (1 - the - gam))

    def general_l(p):
        c1, c2 = p
        # l(t2) = previously estimated lt2
        est_lt2 = a3_lt(c1, c2, t2)
        # l'(t2) = manually estimated derivative
        est_dlt2 = a3_dlt(c1, c2, t2 + 1)
        # add a small tolerance for derivative estimation
        com_dl = int(abs(est_dlt2 - dlt2) > 0.00001)
        return abs(est_lt2 - lt2), com_dl

    # solve for c1 and c2
    res = optimize.fsolve(general_l, x0=np.array([0, 0]))
    s_c1 = float(res[0])
    s_c2 = float(res[1])

    def a3_ht(t):
        # EQ(9) with constants for l(t) and l'(t)
        return ((a_ans * (1 - the - gam)) / m_ans) * a3_dlt(s_c1, s_c2, t) + a3_lt(s_c1, s_c2, t) + the

    # find end of A3 where h(t3) = (1-gamma)
    t3 = optimize.fsolve(lambda t: abs((1 - gam) - a3_ht(t)), x0=np.array([t2]))[0]

    # return t3 and l(t3)
    return t3, a3_lt(s_c1, s_c2, t3)


def phase_a4(t2, lt2):
    # phase A4 is not applicable if phi is greater or equal to gamma
    if phi >= gam:
        return t2, lt2

    # FIRST: Determine dynamics of l(t)
    def a4_lt(c, t):
        return (1 - the - gam) + c * math.exp((-m_ans * t) / ((1 - the - gam) * a_ans))

    # find c that fits to known l(t2) = l
    s_c3 = optimize.fsolve(lambda c: abs(lt2 - a4_lt(c, t2)), x0=np.array([1]))[0]

    # # double-check if C3 estimation lines up
    # t_l0 = optimize.fsolve(lambda t: abs((1 - the - gam - 0.1) - a4_lt(s_c3, t)), x0=np.array([t2]))[0]
    # # simulate with the agent for estimated amount of time steps
    # agent.reset()
    # for _ in range(int(t_l0 * hz)):
    #     agent.set_power(p)
    #     agent.perform_one_step()
    # assert abs(agent.get_g() - (1 - the - gam - 0.1)) < 0.001

    # SECOND: Determine dynamics of h(t)

    # as defined for EQ(21)
    k = m_ans / ((1 - the - gam) * a_ans)
    a = -m_ae / ((1 - phi) * a_anf)
    b = (m_ans * s_c3) / ((1 - the - gam) * a_anf)
    g = p / a_anf

    def a4_ht(c, t):
        return ((-b * math.exp(-k * t)) / (a + k)) + (c * math.exp(a * t)) - (g / a)

    # find c4 that matches h(t2) = (1-gamma)
    s_c4 = optimize.fsolve(lambda c: abs((1 - gam) - a4_ht(c, t2)), x0=np.array([1]))[0]

    # double-check with first derivative and time step t2+1
    # h_act_t2 = (1 - gam)
    # dh_act = (
    #         ((-m_ae * (1 - gam)) / ((1 - phi) * a_anf)) -
    #         ((m_ans * (1 - the - gam - lt2)) / ((1 - the - gam) * a_anf)) +
    #         (p / a_anf)
    # )
    # h_act_t2p = h_act_t2 + dh_act
    # assert abs(a4_ht(s_c4, t2 + 1) - h_act_t2p) < 0.0001

    # find end of phase A4. The time point where h(t4)=(1-phi)
    t4 = optimize.fsolve(lambda t: abs((1 - phi) - a4_ht(s_c4, t)), x0=np.array([t2]))[0]

    # finally return t4 result and l
    return t4, a4_lt(s_c3, t4)


def phase_a5(t4, lt4):
    # FIRST: Determine dynamics of l(t)
    def a5_lt(c, t):
        return (1 - the - gam) + c * math.exp((-m_ans * t) / ((1 - the - gam) * a_ans))

    # find c that fits to known l(t4) = given l
    s_cl = optimize.fsolve(lambda c: abs(lt4 - a5_lt(c, t4)), x0=np.array([1]))[0]

    # SECOND: determine dynamics of h(t)
    k = m_ans / ((1 - the - gam) * a_ans)
    a = (p - m_ae) / a_anf
    b = (m_ans * s_cl) / ((1 - the - gam) * a_anf)

    def a5_ht(cx, t):
        return (a * t) - ((b * math.exp(-k * t)) / k) + cx

    # find c5 that matches h(t4) = (1-phi)
    s_ch = optimize.fsolve(lambda c: abs((1 - phi) - a5_ht(c, t4)), x0=np.array([1]))[0]

    # find end of phase A5. The time point where h(t5)=1
    t5 = optimize.fsolve(lambda t: abs(1 - a5_ht(s_ch, t)) + int(t < t4), x0=np.array([t4]))[0]

    # return time to exhaustion and l at exhaustion
    return t5, a5_lt(s_cl, t5)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    hz = 100

    example_conf = [11532.526538727172,
                    424324.257042239595,
                    249.7641585019016,
                    286.26673813946095,
                    7.988078323028352,
                    0.45486842730772163,
                    0.2641766056862277,
                    0.641766056862277]

    a_anf = example_conf[0]
    a_ans = example_conf[1]
    m_ae = example_conf[2]
    m_ans = example_conf[3]
    m_anf = example_conf[4]
    the = example_conf[5]
    gam = example_conf[6]
    phi = example_conf[7]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae, m_ans=m_ans,
                              m_anf=m_anf, the=the, gam=gam, phi=phi)

    ThreeCompVisualisation(agent)

    p = 400

    # all TTE phases
    t1 = phase_a1()
    t2, lt2 = phase_a2(t1)
    t3, lt3 = phase_a3(t2, lt2)
    t4, lt4 = phase_a4(t3, lt3)
    t5, lt5 = phase_a5(t4, lt4)

    # confirm with assert tests
    eps = 0.001
    # A1
    agent.reset()
    for _ in range(int(t1 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - the) < eps
    # A2
    agent.reset()
    for _ in range(int(t2 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - (1 - max(phi, gam))) < eps
    # A3
    agent.reset()
    for _ in range(int(t3 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - (1 - gam)) < eps
    # A4
    agent.reset()
    for _ in range(int(t4 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    if phi >= gam:
        assert abs(agent.get_h() - (1 - gam)) < eps
    else:
        # A4 is only different from t3 if phi >= gamma
        assert abs(agent.get_h() - (1 - phi)) < eps
    # A5
    agent.reset()
    for _ in range(int(t5 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - 1) < eps

    print(t5)