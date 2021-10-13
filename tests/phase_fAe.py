from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation


def test(func, h, g, p, conf, t=0, hz=500, t_max=2000, log=2):
    # get results of ODE function
    t_end, h_end, g_end, func = func(t_s=t, h_s=h, g_s=g, p=p, t_max=t_max, conf=conf)

    # confirm with iterative agent
    agent = ThreeCompHydAgent(hz=hz, a_anf=conf[0], a_ans=conf[1], m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5], gam=conf[6], phi=conf[7])
    agent.reset()
    agent.set_g(g)
    agent.set_h(h)
    agent.set_power(p)

    # visual check
    if log > 1:
        ThreeCompVisualisation(agent)

    # run simulation
    for _ in range(int((t_end - t) * agent.hz)):
        agent.perform_one_step()

    # verify results
    assert abs(h_end - agent.get_h()) < 0.0001, "{} vs {}".format(h_end, agent.get_h())
    assert abs(g_end - agent.get_g()) < 0.0001, "{} vs {}".format(g_end, agent.get_g())

    # log outputs according to level
    if log > 0:
        logging.info("predicted time: {} \n "
                     "assigned next phase: {}".format(t_end, func))
    if log > 1:
        ThreeCompVisualisation(agent)

    return func


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    conf = [15101.24769778409, 86209.27743067988, 252.71702367096788, 363.2970828395908, 38.27073086773415,
            0.4892228099402588, 0.1524379644134216, 0.780228306857272]
    log = 0
    h2 = 0.325100624266976845
    g2 = 0

    # fAe has three possible ends
    func = ODEThreeCompHydSimulator.fAe
    logging.info("Func {}".format(func))

    p = 0
    logging.info("h = 1 - phi")
    n_func = test(func, h2, g2, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.lAe

    p = 280
    logging.info("h = 1 - gamma")
    n_func = test(func, h2, g2, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe_lAn

    logging.info("time limit")
    p = 254
    n_func = test(func, h2, g2, p, conf, t_max=100, log=log)
    assert n_func is None
