from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging

def tte_test_procedure(p, hz, eps, conf, agent, log_level=0):
    # all TTE phases
    max_time = 5000

    phases = [ODEThreeCompHydSimulator.work_lAe,
              ODEThreeCompHydSimulator.work_lAe_rAnS,
              ODEThreeCompHydSimulator.work_fAe,
              ODEThreeCompHydSimulator.work_fAe_rAnS,
              ODEThreeCompHydSimulator.work_lAe_lAnS,
              ODEThreeCompHydSimulator.work_fAe_lAnS,
              ODEThreeCompHydSimulator.work_lAe_fAns,
              ODEThreeCompHydSimulator.work_fAe_fAnS]

    # set initial conditions
    h_s = 0
    g_s = 1 - conf[6] - conf[5]
    t, h, g = 0, h_s, g_s
    ts, hts, gts = [], [], []

    # display initial state if log level is high enough
    if log_level > 0:
        agent.set_h(h_s)
        agent.set_g(g_s)
        ThreeCompVisualisation(agent)

    # iterate through all phases until end is reached
    for phase in phases:
        t, h, g = phase(t, h, g, p_exp=p, t_max=max_time, conf=conf)

        # display intermediate state if log level is high enough
        if log_level > 0:
            logging.info("PHASE {}".format(phase))
            agent.set_h(h)
            agent.set_g(g)
            ThreeCompVisualisation(agent)

        # exit loop if end is reached
        if t >= max_time:
            logging.info("EQUILIBRIUM IN {}: t: {} h: {} g: {}".format(phase, t, h, g))
            break

        # keep track of times and states at phase ends
        ts.append(t)
        hts.append(h)
        gts.append(g)

    # now confirm with iterative agent
    for i, t in enumerate(ts):
        # set to initial state
        agent.reset()
        agent.set_h(h_s)
        agent.set_g(g_s)

        # simulate tte
        for _ in range(int(round(t * hz))):
            agent.set_power(p)
            agent.perform_one_step()

        # estimate estimation differences
        g_diff = agent.get_g() - gts[i]
        h_diff = agent.get_h() - hts[i]

        if log_level >= 2:
            print("error phase {}. h is off by {}".format(i + 1, h_diff))
            print("error phase {}. g is off by {}".format(i + 1, g_diff))

        assert abs(g_diff) < eps, "error phase {}. g is off by {}".format(i + 1, g_diff)
        assert abs(h_diff) < eps, "error phase {}. h is off by {}".format(i + 1, h_diff)


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
        # create three component hydraulic agent with example configuration
        agent = ThreeCompHydAgent(hz=hz, a_anf=example_conf[0], a_ans=example_conf[1], m_ae=example_conf[2],
                                  m_ans=example_conf[3], m_anf=example_conf[4], the=example_conf[5],
                                  gam=example_conf[6], phi=example_conf[7])

        # ThreeCompVisualisation(agent)
        tte_test_procedure(p, hz, eps, example_conf, agent)


def test_one_config(example_conf=None):
    """
    tests given configuration and puts out some more details
    """

    # just a default value
    if example_conf is None:
        example_conf = [25925.53993526785, 60694.43170965706, 219.8524740824735, 216.83073328159165,
                        26.323382867622215, 0.15040127238313122, 0.28924204946747195, 0.1762119589774433]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=example_conf[0], a_ans=example_conf[1], m_ae=example_conf[2],
                              m_ans=example_conf[3], m_anf=example_conf[4], the=example_conf[5],
                              gam=example_conf[6], phi=example_conf[7])
    tte_test_procedure(p, hz, eps, example_conf, agent, log_level=2)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p = 450
    # estimations per second for discrete agent
    hz = 250
    # required precision of discrete to differential agent
    eps = 0.001

    test_one_config()

    the_loop(p=p, hz=hz, eps=eps)
