from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 260
    p_rec = 0

    # estimations per second for discrete agent
    hz = 1000

    # a D configuration
    conf = [15101.24769778409, 86209.27743067988, 252.71702367096788, 363.2970828395908, 38.27073086773415,
            0.14892228099402588, 0.3524379644134216, 0.1580228306857272]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              a_anf=conf[0], a_ans=conf[1],
                              m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])
    phi = conf[7]

    t2 = 2643.143819400002
    h2 = 0.025100624266976845
    g2 = 4.883399240540598e-12

    t_end, h_end, g_end = ODEThreeCompHydSimulator.fAe(t_s=t2,
                                                       h_s=h2,
                                                       g_s=g2,
                                                       p=p_rec,
                                                       t_max=5000,
                                                       conf=conf)

    # check in simulation
    agent.reset()
    agent.set_g(g2)
    agent.set_h(h2)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for _ in range(int((t_end - t2) * agent.hz)):
        agent.perform_one_step()

    logging.info("predicted time: {} \n"
                 "diff h: {}\n"
                 "diff g: {}".format(t_end,
                                     h_end - agent.get_h(),
                                     g_end - agent.get_g()))
    ThreeCompVisualisation(agent)
