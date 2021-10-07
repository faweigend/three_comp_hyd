from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging


def tte_test_procedure(p, hz, eps, conf, log_level=0):
    # all TTE phases
    max_time = 5000

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=conf[0], a_ans=conf[1], m_ae=conf[2],
                              m_ans=conf[3], m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    phases = [ODEThreeCompHydSimulator.lAe,
              ODEThreeCompHydSimulator.lAe_rAn,
              ODEThreeCompHydSimulator.fAe,
              ODEThreeCompHydSimulator.work_fAe_rAnS,
              ODEThreeCompHydSimulator.lAe_lAnS,
              ODEThreeCompHydSimulator.work_fAe_lAnS,
              ODEThreeCompHydSimulator.work_lAe_fAns,
              ODEThreeCompHydSimulator.work_fAe_fAnS]

    # set initial conditions
    h_s = 0
    g_s = 0  # 1 - conf[6] - conf[5]
    t, h, g = 0, h_s, g_s

    # display initial state if log level is high enough
    if log_level > 0:
        agent.set_h(h_s)
        agent.set_g(g_s)
        ThreeCompVisualisation(agent)

    # iterate through all phases until end is reached
    for phase in phases:
        t, h, g = phase(t, h, g, p, t_max=max_time, conf=conf)

        # display intermediate state if log level is high enough
        if log_level > 0:
            logging.info("PHASE {} t {} ".format(phase, t))
            agent.set_h(h)
            agent.set_g(g)
            logging.info("ODE".format(phase, t))
            ThreeCompVisualisation(agent)

        # exit loop if end is reached
        if t >= max_time:
            logging.info("EQUILIBRIUM IN {}: t: {} h: {} g: {}".format(phase, t, h, g))
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
            logging.info("error phase {}. h is off by {}".format(phase, h_diff))
            logging.info("error phase {}. g is off by {}".format(phase, g_diff))
            logging.info("ITERATIVE".format(phase, t))
            ThreeCompVisualisation(agent)

        assert abs(g_diff) < eps, "error phase {}. g is off by {}".format(phase, g_diff)
        assert abs(h_diff) < eps, "error phase {}. h is off by {}".format(phase, h_diff)

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

    p = 560
    # estimations per second for discrete agent
    hz = 500
    # required precision of discrete to differential agent
    eps = 0.001

    example_conf = [21704.77778915587, 61925.84797188902, 212.76772005473063, 140.0897845828814, 32.4028329961532,
         0.3217431159932008, 0.2683727457040581, 0.7190401470030847]
    tte_test_procedure(p, hz, eps, example_conf, log_level=2)

    the_loop(p=p, hz=hz, eps=eps)
