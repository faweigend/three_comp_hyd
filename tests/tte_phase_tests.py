import numpy as np
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging


def tte_test_procedure(p, hz, eps, conf, log_level=0):
    # all TTE phases
    t_max = 5000

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=conf[0], a_ans=conf[1], m_ae=conf[2],
                              m_ans=conf[3], m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    # set initial conditions
    h_s = 0
    g_s = 0  # 1 - conf[6] - conf[5]
    t, h, g = 0, h_s, g_s

    # display initial state if log level is high enough
    if log_level > 0:
        agent.set_h(h_s)
        agent.set_g(g_s)
        ThreeCompVisualisation(agent)

    func = None
    while t < t_max:
        if func is None:
            # first distinguish between fAe and lAe
            func = ODEThreeCompHydSimulator.state_to_phase(conf, h, g)

        # iterate through all phases until end is reached
        t, h, g, n_func = func(t, h, g, p, t_max=t_max, conf=conf)

        # display intermediate state if log level is high enough
        if log_level > 0:
            logging.info("PHASE {} t {} ".format(func, t))
            agent.set_h(h)
            agent.set_g(g)
            logging.info("ODE".format(func, t))
            ThreeCompVisualisation(agent)

        func = n_func
        if log_level > 0:
            logging.info("next PHASE {}".format(func))

        # exit loop if end of iteration is reached
        if t >= t_max or n_func is None:
            logging.info("END IN {}: t: {} h: {} g: {}".format(func, t, h, g))
            break

        # if recovery time is reached return fill levels at that point
        if t == np.nan:
            break

        # now confirm with iterative agent
        # set to initial state
        agent.reset()
        agent.set_h(h_s)
        agent.set_g(g_s)

        # simulate tte
        for _ in range(int(round(t * hz))):
            agent.set_power(p)
            agent.perform_one_step()

        # estimate estimation differences
        g_diff = agent.get_g() - g
        h_diff = agent.get_h() - h

        if log_level >= 2:
            logging.info("error phase {}. h is off by {}".format(func, h_diff))
            logging.info("error phase {}. g is off by {}".format(func, g_diff))
            logging.info("ITERATIVE".format(func, t))
            ThreeCompVisualisation(agent)

        assert abs(g_diff) < eps, "error phase {}. g is off by {}".format(func, g_diff)
        assert abs(h_diff) < eps, "error phase {}. h is off by {}".format(func, h_diff)

    # confirm the tte time with an entire iterative simulation
    c_tte = ThreeCompHydSimulator.tte(agent, p_work=p, t_max=t_max)
    assert abs(c_tte - t) < eps, "confirmation error. Difference betwen " \
                                 "ODE TTE {} and Iterative TTE {} is {}".format(t,
                                                                                c_tte,
                                                                                abs(c_tte - t))

    # if all phases complete full exhaustion is reached
    return t, h, g


def the_loop(p: float = 350.0,
             hz: int = 250,
             eps: float = 0.001):
    """
    creates random agents and tests the discretised against the differential one
    """

    while True:
        udp = MultiObjectiveThreeCompUDP(None, None)
        example_conf = udp.create_educated_initial_guess()
        logging.info(example_conf)
        tte_test_procedure(p, hz, eps, example_conf)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p = 260
    # estimations per second for discrete agent
    hz = 500
    # required precision of discrete to differential agent
    eps = 0.01

    example_conf = [15101.24769778409, 86209.27743067988, 252.71702367096788, 363.2970828395908,
                    38.27073086773415, 0.14892228099402588, 0.3524379644134216, 1.0]

    tte_test_procedure(p, hz, eps, example_conf, log_level=1)

    # the_loop(p=p, hz=hz, eps=eps)
