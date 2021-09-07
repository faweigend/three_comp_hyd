import math

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

from tests import configurations

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 100

    # estimations per second for discrete agent
    hz = 250

    conf = configurations.a

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=conf[0], a_ans=conf[1], m_ae=conf[2],
                              m_ans=conf[3], m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    # PHASE A5
    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    t5 = 0
    h5 = 1.0 - phi
    g5 = 0.1

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


    h_target = 1 - gamma

    # estimate an initial guess that assumes no contribution from g
    initial_guess = 0
    rt5 = optimize.fsolve(lambda t: a5_ht(t) - h_target, x0=np.array([initial_guess]))[0]

    agent.reset()
    agent.set_g(g5)
    agent.set_h(h5)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for _ in range(int(rt5 * agent.hz)):
        agent.perform_one_step()

    logging.info("predicted time: {} \n"
                 "diff h: {}\n"
                 "diff g: {}".format(rt5,
                                     a5_ht(rt5) - agent.get_h(),
                                     a5_gt(rt5) - agent.get_g()))
    ThreeCompVisualisation(agent)
