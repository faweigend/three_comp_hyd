from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging
import warnings

import numpy as np


# warnings.filterwarnings("error")


def rec_trial_procedure(p_exp, p_rec, t_rec, hz, eps, conf, agent, log_level=0):
    max_time = 5000
    # A6
    t6, ht6, gt6 = ODEThreeCompHydSimulator.tte(p_exp=p_exp, conf=conf, max_time=max_time)

    # if TTE too long
    if t6 > max_time:
        return

    agent.reset()
    for _ in range(int(round(t6 * hz))):
        agent.set_power(p_exp)
        agent.perform_one_step()

    g_diff = agent.get_g() - gt6
    h_diff = agent.get_h() - ht6

    if log_level >= 2:
        print("error tte1. h is off by {}".format(h_diff))
        print("error tte1. g is off by {}".format(g_diff))

    assert abs(g_diff) < eps, "error tte1. g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "error tte1. h is off by {}".format(h_diff)

    # recovery begins here
    rt6, rh6, rg6 = ODEThreeCompHydSimulator.phase_a6(t5=t6, ht5=ht6, gt5=gt6, p=p_rec, conf=conf)
    print(t6, ht6, gt6)
    print(rt6, rh6, rg6)


def the_loop(p_exp: float = 350.0, p_rec: float = 100.0,
             t_rec: int = 240, hz: int = 250, eps: float = 0.001):
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
        rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec,
                            hz=hz, eps=eps, conf=example_conf,
                            agent=agent)

        break


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 100
    t_rec = 240

    # estimations per second for discrete agent
    hz = 250
    # required precision of discrete to differential agent
    eps = 0.005

    the_loop(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, hz=hz, eps=eps)
