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

    conf = [15101.24769778409, 86209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              a_anf=conf[0], a_ans=conf[1],
                              m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    t1 = 5000
    ht1 = 0.016964525316181377
    gt1 = 0.0

    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    # our general solution to the integral
    s_c1 = (ht1 + p_rec * phi / m_ae - p_rec / m_ae) * np.exp(-m_ae * t1 / (a_anf * (phi - 1)))

    # full recovery is only possible if p_rec is 0
    # use equation (4) from morton with own addition
    def a1_ht(t):
        return p_rec * (1 - phi) / m_ae * s_c1 * np.exp(- m_ae * t / (a_anf * (1 - phi)))

    # h(t) = 0 is never reached and causes a log(0) estimation. A close approximation is h(t) = 0.0001
    t0 = a_anf * (1 + phi) / - m_ae * np.log(0.0001 / s_c1 + p_rec * (phi + 1) / m_ae * s_c1)

    # check in simulation
    agent.reset()
    agent.set_g(gt1)
    agent.set_h(ht1)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for _ in range(int(t0 * agent.hz)):
        agent.perform_one_step()

    logging.info("predicted time: {} \n"
                 "diff h: {}\n"
                 "diff g: {}".format(t0,
                                     0 - agent.get_h(),
                                     0 - agent.get_g()))
    ThreeCompVisualisation(agent)
