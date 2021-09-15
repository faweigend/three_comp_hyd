from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging

import numpy as np


def rec_trial_procedure(p_exp, p_rec, t_max, hz, eps, conf, agent, log_level=0):
    # Start with first time to exhaustion bout
    t, h, g = ODEThreeCompHydSimulator.tte(p_exp=p_exp, conf=conf)

    if t == np.inf:
        return

    # double-check with discrete agent
    for _ in range(int(round(t * hz))):
        agent.set_power(p_exp)
        agent.perform_one_step()
    g_diff = agent.get_g() - g
    h_diff = agent.get_h() - h
    assert abs(g_diff) < eps, "TTE g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "TTE h is off by {}".format(h_diff)
    # ThreeCompVisualisation(agent)

    # now iterate through all recovery phases
    phases = [ODEThreeCompHydSimulator.rec_a6,
              ODEThreeCompHydSimulator.rec_a5,
              ODEThreeCompHydSimulator.rec_a4_r1,
              ODEThreeCompHydSimulator.rec_a4_r2,
              ODEThreeCompHydSimulator.rec_a3_r1,
              ODEThreeCompHydSimulator.rec_a3_r2,
              ODEThreeCompHydSimulator.rec_a2,
              ODEThreeCompHydSimulator.rec_a1]

    # restart time from 0
    t = 0

    # detailed checks for every phase
    for phase in phases:
        # save previous time to estimate time difference
        t_p = t

        # get estimated time of phase end
        t, h, g = phase(t, h, g, p_rec=p_rec, t_max=t_max, conf=conf)
        # logging.info("{}\nt {}\nh {}\ng {}".format(phase, t, h, g))

        # double-check with discrete agent
        for _ in range(int(round((t - t_p) * hz))):
            agent.set_power(p_rec)
            agent.perform_one_step()
        g_diff = agent.get_g() - g
        h_diff = agent.get_h() - h

        # ThreeCompVisualisation(agent)

        assert abs(g_diff) < eps, "{} g is off by {}".format(phase, g_diff)
        assert abs(h_diff) < eps, "{} h is off by {}".format(phase, h_diff)

        if t == t_max:
            logging.info("Max recovery reached in {}".format(phase))
            # ThreeCompVisualisation(agent)
            return


def the_loop(p_exp: float = 350.0, p_rec: float = 100.0,
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

        rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_max=t_max,
                            hz=hz, eps=eps, conf=example_conf,
                            agent=agent, log_level=2)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 260
    t_max = 180
    p_rec = 0

    # estimations per second for discrete agent
    hz = 250
    # required precision of discrete to differential agent
    eps = 0.0001

    # a configuration
    # c = [15101.24769778409, 86209.27743067988, 252.71702367096788, 363.2970828395908, 38.27073086773415,
    #      0.14892228099402588, 0.3524379644134216, 0.1580228306857272]
    # agent = ThreeCompHydAgent(hz=hz, a_anf=c[0], a_ans=c[1], m_ae=c[2],
    #                           m_ans=c[3], m_anf=c[4], the=c[5],
    #                           gam=c[6], phi=c[7])
    # ThreeCompVisualisation(agent)
    #
    # rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_max=t_max,
    #                     hz=hz, eps=eps, conf=c,
    #                     agent=agent, log_level=2)

    the_loop(p_exp=p_exp, p_rec=p_rec, t_max=t_max, hz=hz, eps=eps)
