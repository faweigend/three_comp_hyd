from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging

import numpy as np


def rec_trial_procedure(p_exp, p_rec, t_rec, t_max, hz, eps, conf, agent, log_level=0):
    # Start with first time to exhaustion bout
    tte_1, h, g = ODEThreeCompHydSimulator.tte(conf=conf,
                                               start_h=0, start_g=0,
                                               p_exp=p_exp, t_max=t_max)

    if tte_1 == np.inf or int(tte_1) == int(t_max):
        logging.info("Exhaustion not reached during TTE")
        return

    # double-check with discrete agent
    for _ in range(int(round(tte_1 * hz))):
        agent.set_power(p_exp)
        agent.perform_one_step()
    g_diff = agent.get_g() - g
    h_diff = agent.get_h() - h
    assert abs(g_diff) < eps, "TTE1 g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "TTE1 h is off by {}".format(h_diff)

    rec, h, g = ODEThreeCompHydSimulator.rec(conf=conf,
                                             start_h=h, start_g=g,
                                             p_rec=p_rec, t_max=t_rec)

    # double-check with discrete agent
    for _ in range(int(round(rec * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - g
    h_diff = agent.get_h() - h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)

    tte_2, h, g = ODEThreeCompHydSimulator.tte(conf=conf,
                                               start_h=h, start_g=g,
                                               p_exp=p_exp, t_max=t_max)

    # double-check with discrete agent
    for _ in range(int(round(tte_2 * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - g
    h_diff = agent.get_h() - h
    assert abs(g_diff) < eps, "TTE2 g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "TTE2 h is off by {}".format(h_diff)

    rec_ratio = tte_2 / tte_1 * 100.0
    est_ratio = ThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(agent=agent, p_exp=p_exp,
                                                                 p_rec=p_rec, t_rec=t_rec)
    logging.info("ODE ratio: {} \nEST ratio: {}".format(rec_ratio, est_ratio))


def the_loop(p_exp: float = 350.0, p_rec: float = 100.0, t_rec=180.0,
             t_max: float = 240, hz: int = 250, eps: float = 0.001):
    """
    creates random agents and tests the discretised against the differential one
    """

    while True:
        udp = MultiObjectiveThreeCompUDP(None, None)

        example_conf = udp.create_educated_initial_guess()
        logging.info(example_conf)
        # create three component hydraulic agent with example configuration
        agent = ThreeCompHydAgent(hz=hz, a_anf=example_conf[0], a_ans=example_conf[1], m_ae=example_conf[2],
                                  m_ans=example_conf[3], m_anf=example_conf[4], the=example_conf[5],
                                  gam=example_conf[6], phi=example_conf[7])

        rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max,
                            hz=hz, eps=eps, conf=example_conf,
                            agent=agent, log_level=2)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 560
    t_rec = 180
    p_rec = 0
    t_max = 5000

    # estimations per second for discrete agent
    hz = 250
    # required precision of discrete to differential agent
    eps = 0.001

    # a configuration
    c = [17530.530747393303, 37625.72364566721, 268.7372285266482, 223.97570400889148,
         7.895654547752743, 0.1954551343626819, 0.224106497474462, 0.01]
    agent = ThreeCompHydAgent(hz=hz, a_anf=c[0], a_ans=c[1], m_ae=c[2],
                              m_ans=c[3], m_anf=c[4], the=c[5],
                              gam=c[6], phi=c[7])
    ThreeCompVisualisation(agent)

    rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max,
                        hz=hz, eps=eps, conf=c,
                        agent=agent, log_level=2)

    # the_loop(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max, hz=hz, eps=eps)
