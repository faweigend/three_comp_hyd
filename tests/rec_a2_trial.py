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
    p_rec = 0

    # estimations per second for discrete agent
    hz = 250

    # a D configuration
    conf = [15101.24769778409, 86209.27743067988,  # anf, ans
            252.71702367096787, 363.2970828395908,  # m_ae, m_ans
            38.27073086773415, 0.64892228099402588,  # m_anf, theta
            0.1580228306857272, 0.6580228306857272]  # gamma, phi

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              a_anf=conf[0], a_ans=conf[1],
                              m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    t2 = 0
    h2 = 0.5
    g2 = 0.0

    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]


    def a2_ht(t):
        return h2 - t * (m_ae - p_rec) / a_anf

    t_end = (h2 - 1 + phi) * a_anf / (m_ae - p_rec)

    # check in simulation
    agent.reset()
    agent.set_g(g2)
    agent.set_h(h2)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for _ in range(int(t_end * agent.hz)):
        agent.perform_one_step()

    logging.info("predicted time: {} \n"
                 "diff h: {}\n"
                 "diff g: {}".format(t_end,
                                     (1 - phi) - agent.get_h(),
                                     0 - agent.get_g()))
    ThreeCompVisualisation(agent)
