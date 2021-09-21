from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging
import warnings

import numpy as np


# warnings.filterwarnings("error")


def tte_test_procedure(p, hz, eps, conf, agent, log_level=0):
    # all TTE phases
    max_time = 5000

    # A1
    t1, ht1, gt1 = ODEThreeCompHydSimulator.lAe(t_s=0, h_s=0, g_s=0, p_exp=p, t_max=max_time, conf=conf)
    if t1 == np.inf or t1 >= max_time:
        logging.info("EQUILIBRIUM IN A1: t: {} h: {} g: {}".format(t1, ht1, gt1))
        return
    logging.info("A1: {}".format(t1))

    t2, ht2, gt2 = ODEThreeCompHydSimulator.mAe(t_s=t1, h_s=ht1, g_s=gt1, p_exp=p, t_max=max_time, conf=conf)
    logging.info("A2: {}".format(t2))

    # A3
    t3, ht3, gt3 = ODEThreeCompHydSimulator.tte_a3(t3=t2, h3=ht2, g3=gt2, p_exp=p, t_max=max_time, conf=conf)
    if t3 == np.inf or t3 >= max_time:
        logging.info("EQUILIBRIUM IN A3: t: {} h: {} g: {}".format(t3, ht3, gt3))
        return
    logging.info("A3: {}".format(t3))

    # A4
    t4, ht4, gt4 = ODEThreeCompHydSimulator.tte_a4(t4=t3, h4=ht3, g4=gt3, p_exp=p, t_max=max_time, conf=conf)
    if t4 == np.inf or t4 >= max_time:
        logging.info("EQUILIBRIUM IN A4: t: {} h: {} g: {}".format(t4, ht4, gt4))
        return
    logging.info("A4: {}".format(t4))

    # A5
    t5, ht5, gt5 = ODEThreeCompHydSimulator.tte_a5(t5=t4, h5=ht4, g5=gt4, p_exp=p, t_max=max_time, conf=conf)
    if t5 == np.inf or t5 >= max_time:
        logging.info("EQUILIBRIUM IN A5: t: {} h: {} g: {}".format(t5, ht5, gt5))
        return
    logging.info("A5: {}".format(t5))

    # A6
    t6, ht6, gt6 = ODEThreeCompHydSimulator.tte_a6(t6=t5, h6=ht5, g6=gt5, p_exp=p, t_max=max_time, conf=conf)
    if t6 == np.inf or t6 >= max_time:
        logging.info("EQUILIBRIUM IN A6: t: {} h: {} g: {}".format(t6, ht6, gt6))
        return
    logging.info("A6: {}".format(t6))

    ts = [t1, t2, t3, t4, t5, t6]
    hts = [ht1, ht2, ht3, ht4, ht5, ht6]
    gts = [gt1, gt2, gt3, gt4, gt5, gt6]

    for i, t in enumerate(ts):
        agent.reset()
        for _ in range(int(round(t * hz))):
            agent.set_power(p)
            agent.perform_one_step()

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
        example_conf = [22960.500530676287, 77373.91670678859, 234.24391186170348, 382.9246247635444, 32.281951614864944, 0.4382785572308323, 0.29408404649271447, 0.18875064978567485]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=example_conf[0], a_ans=example_conf[1], m_ae=example_conf[2],
                              m_ans=example_conf[3], m_anf=example_conf[4], the=example_conf[5],
                              gam=example_conf[6], phi=example_conf[7])

    ThreeCompVisualisation(agent)

    tte_test_procedure(p, hz, eps, example_conf, agent, log_level=2)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p = 260
    # estimations per second for discrete agent
    hz = 250
    # required precision of discrete to differential agent
    eps = 0.001

    test_one_config()

    the_loop(p=p, hz=hz, eps=eps)
