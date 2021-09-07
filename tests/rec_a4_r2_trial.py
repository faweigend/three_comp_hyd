from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 100

    # estimations per second for discrete agent
    hz = 250

    conf = [15101.24769778409, 486209.27743067988, 252.71702367096787,
            363.2970828395908, 43.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              a_anf=conf[0], a_ans=conf[1],
                              m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    # PHASE A4
    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    # derived from tte estimator and
    t4 = 0
    h4 = 0.6
    g4 = h4 - theta

    # if h is above g (flow from AnF into AnS)
    a_hg = (a_anf + a_ans) * m_anf / (a_anf * a_ans * (1 - gamma))
    b_hg = (p_rec - m_ae) * m_anf / (a_anf * a_ans * (1 - gamma))

    # derivative g'(t4) can be calculated manually
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


    ht_end = 1 - phi
    # check if equilibrium in this phase

    t_end = optimize.fsolve(lambda t: ht_end - a4_ht(t), x0=np.array([0]))[0]
    gt_end = a4_gt(t_end)

    agent.reset()
    agent.set_g(g4)
    agent.set_h(h4)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for _ in range(int(t_end * agent.hz)):
        agent.perform_one_step()

    logging.info("predicted time: {} \n"
                 "diff h: {}\n"
                 "diff g: {}".format(t_end,
                                     a4_ht(t_end) - agent.get_h(),
                                     a4_gt(t_end) - agent.get_g()))
    ThreeCompVisualisation(agent)
