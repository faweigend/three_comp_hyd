import time

import numpy as np
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging


def rec_trial_procedure(p_exp, p_rec, t_rec, t_max, hz, conf, log_level=0):
    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              lf=conf[0], ls=conf[1],
                              m_u=conf[2], m_ls=conf[3],
                              m_lf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    # simulator step limit needs to be adjusted
    est_t0 = time.process_time_ns()
    est_ratio = ThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(agent=agent, p_exp=p_exp, p_rec=p_rec,
                                                                 t_rec=t_rec, t_max=t_max)
    est_t = time.process_time_ns() - est_t0

    ode_t0 = time.process_time_ns()
    rec_ratio = ODEThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(conf=conf, p_exp=p_exp, p_rec=p_rec,
                                                                    t_rec=t_rec, t_max=t_max)
    ode_t = time.process_time_ns() - ode_t0

    # rec ratio estimation difference
    e_diff = abs(est_ratio - rec_ratio)
    # >0 if ODE faster
    t_diff = est_t - ode_t
    return t_diff, e_diff


def the_loop(t_max: float = 5000, hz: int = 250):
    """
    creates random agents and tests the discretised against the differential one
    """

    e_results = []
    t_results = []
    n = 1000
    for i in range(n):
        udp = MultiObjectiveThreeCompUDP(None, None)

        example_conf = udp.create_educated_initial_guess()

        p_exp = example_conf[3] * 3 * abs(np.random.randn())
        p_rec = example_conf[3] * 0.5 * abs(np.random.randn())
        t_rec = 240 * abs(np.random.randn())

        logging.info("{}/{} Pwork {} Prec {} Trec {} conf {}".format(i, n, p_exp, p_rec, t_rec, example_conf))

        try:
            t_diff, e_diff = rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max,
                                                 hz=hz, conf=example_conf, log_level=0)
            e_results.append(e_diff)
            t_results.append(t_diff)
        except UserWarning:
            continue

    mean_t = np.mean(t_results)
    str_mean_t = ""
    if mean_t > 0:
        str_mean_t += "ODE"
    else:
        str_mean_t += "Iterative"
    str_mean_t += " is faster with {}".format(mean_t)

    logging.info("\nsuccessful samples: {}\ntime performance: {}\nest error {}".format(
        len(t_results), str_mean_t, np.mean(e_results)))


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # estimations per second for discrete agent
    hz = 5
    t_max = 5000

    the_loop(t_max=t_max, hz=hz)
