import logging
import math

import numpy as np
from scipy import optimize
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent


def phase_a1():
    # end of phase A1 -> the time whe h=theta
    t_end = (-(a_anf * (1 - phi)) / m_ae) * math.log(1 - ((m_ae * the) / (p * (1 - phi))))

    # simulate with the agent for estimated amount of time steps
    agent.reset()
    for _ in range(int(t_end * hz)):
        agent.set_power(p)
        agent.perform_one_step()

    # check if t1 time estimation lines up
    assert abs(agent.get_h() - the) < 0.001

    return t_end


def phase_a2():
    # taken from Equation 11 by Morton 1986
    a = (m_ae * a_ans * (1 - the - gam) + m_ans * (a_anf + a_ans) * (1 - phi)) / (
            a_anf * a_ans * (1 - phi) * (1 - the - gam))

    b = (m_ae * m_ans) / (a_anf * a_ans * (1 - phi) * (1 - the - gam))

    c = (m_ans * (p * (1 - phi) - m_ae * the)) / (a_anf * a_ans * (1 - phi) * (1 - the - gam))

    # use auxillary equation to derive both negative roots r1 and r2
    coeff = [1, a, b]
    r1, r2 = np.roots(coeff)

    def eq_12_and_derivative(p):
        c1, c2 = p
        return (c1 * math.exp(r1 * t1) + c2 * math.exp(r2 * t1) + c / b,
                r1 * c1 * math.exp(r1 * t1) + r2 * c2 * math.exp(r2 * t1))

    # solve for c1 and c2
    res = optimize.fsolve(eq_12_and_derivative, x0=np.array([1, 1]))
    s_c1 = float(res[0])
    s_c2 = float(res[1])

    def eq_9_substituted(t):
        k1 = r1 * s_c1 * ((a_ans * (1 - the - gam)) / m_ans) + s_c1
        k2 = r2 * s_c2 * ((a_ans * (1 - the - gam)) / m_ans) + s_c2
        h = k1 * math.exp(r1 * t) + k2 * math.exp(r2 * t) + c / b + the
        return h - (1 - gam)

    t_end = optimize.fsolve(eq_9_substituted, x0=np.array([1]))[0]

    # simulate with the agent for estimated amount of time steps
    agent.reset()
    for _ in range(int((t_end) * hz)):
        agent.set_power(p)
        agent.perform_one_step()

    # check if t1 time estimation lines up
    assert abs(agent.get_h() - (1 - gam)) < 0.001

    # use t2 to get l(t2)
    l_t2 = s_c1 * math.exp(r1 * t_end) + s_c2 * math.exp(r2 * t_end) + c / b

    return l_t2, t_end


def phase_a4(l_t2, t_end_t2):
    # PHASE A4

    # FIRST: Determine dynamics of l(t)
    def a4_lt(cx, t):
        return (1 - the - gam) + cx * math.exp((-m_ans * t) / ((1 - the - gam) * a_ans))

    # solve EQ(20)
    def eq_20(c3):
        # fit to t2 and known l(t2)
        l_est_t2 = a4_lt(c3, t_end_t2)
        return abs(l_t2 - l_est_t2)

    # find root
    s_c3 = optimize.fsolve(eq_20, x0=np.array([1]))[0]

    # check if C3 estimation lines up
    t_l0 = optimize.fsolve(lambda t: abs((1 - the - gam - 0.1) - a4_lt(s_c3, t)), x0=np.array([t_end_t2]))[0]
    # simulate with the agent for estimated amount of time steps
    agent.reset()
    for _ in range(int(t_l0 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_g() - (1 - the - gam - 0.1)) < 0.01

    # SECOND: Determine dynamics of h(t)

    # as defined for EQ(21)
    k = m_ans / ((1 - the - gam) * a_ans)
    a = -m_ae / ((1 - phi) * a_anf)
    b = (m_ans * s_c3) / ((1 - the - gam) * a_anf)
    g = p / a_anf

    def a4_ht(cx, t):
        return ((-b * math.exp(-k * t)) / (a + k)) + (cx * math.exp(a * t)) - (g / a)

    def eq_21_v3(c4):
        # we know that at t2 h=(1-gamma)
        h_act_t2 = (1 - gam)
        h_est_t2 = a4_ht(c4, t_end_t2)
        return abs(h_act_t2 - h_est_t2)

    s_c4 = optimize.fsolve(eq_21_v3, x0=np.array([1]))[0]

    # double-check with first derivative and time step t2+1
    h_act_t2 = (1 - gam)
    dh_act = (
            ((-m_ae * (1 - gam)) / ((1 - phi) * a_anf)) -
            ((m_ans * (1 - the - gam - l_t2)) / ((1 - the - gam) * a_anf)) +
            (p / a_anf)
    )
    h_act_t2p = h_act_t2 + dh_act
    assert abs(a4_ht(s_c4, t_end_t2 + 1) - h_act_t2p) < 0.001

    # find time point where h=(1-phi)
    t_end = optimize.fsolve(lambda t: abs((1 - phi) - a4_ht(s_c4, t)), x0=np.array([t_end_t2]))[0]

    # check if t4 time estimation lines up
    agent.reset()
    for _ in range(int(t_end * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - (1 - phi)) < 0.001

    # finally return t4 result
    return t_end


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    hz = 40

    example_conf = [11532.526538727172,
                    543240.257042239595,
                    249.7641585019016,
                    286.26673813946095,
                    7.988078323028352,
                    0.25486842730772163,
                    0.26874299216869681,
                    0.2141766056862277]

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

    p = 400

    # all TTE phases
    t1 = phase_a1()
    lt2, t2 = phase_a2()
    t4 = phase_a4(lt2, t2)
