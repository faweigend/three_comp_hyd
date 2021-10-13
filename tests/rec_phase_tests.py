from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging

import numpy as np


def rec_phase_procedure(p_exp: float, p_rec: float, t_rec: float, t_max: float, hz: int, eps: float,
                        conf: list, log_level: int = 0):
    """
    Conducts a TTE and follows it up with a recovery at given conditions. Estimation results of the ODE integral
    estimations are compared to differential results of the interative agent.
    :param p_exp: intensity for TTE
    :param p_rec: intensity for recovery
    :param t_rec: duration of recovery
    :param t_max: maximal duration of exercise or recovery bout
    :param hz: delta t for iterative agent
    :param eps: error tolerance for difference between differential and integral agent estimations
    :param conf: configuration of hydraulic model to investigate
    :param log_level: amount of detail about results to be logged
    """

    agent = ThreeCompHydAgent(hz=hz, a_anf=conf[0], a_ans=conf[1], m_ae=conf[2],
                              m_ans=conf[3], m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])
    if log_level > 0:
        logging.info("Agent to be examined")
        ThreeCompVisualisation(agent)

    # Start with first time to exhaustion bout
    t, h, g = ODEThreeCompHydSimulator.constant_power_trial(p=p_exp, start_h=0, start_g=0, conf=conf, t_max=t_max)

    if t >= t_max:
        logging.info("Exhaustion not reached during TTE")
        return

    # double-check with discrete agent
    for _ in range(int(round(t * hz))):
        agent.set_power(p_exp)
        agent.perform_one_step()
    g_diff = agent.get_g() - g
    h_diff = agent.get_h() - h
    assert abs(g_diff) < eps, "TTE g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "TTE h is off by {}".format(h_diff)

    # display fill levels at exhaustion
    if log_level > 0:
        logging.info("[SUCCESS] Agent after TTE at {}".format(p_exp))
        ThreeCompVisualisation(agent)

    # restart time from 0
    t = 0

    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]
    func = None

    # cycles through phases until t_max is reached
    while t < t_rec:

        if func is None:
            # first distinguish between fAe and lAe
            if h >= 1 - phi:
                # fAe
                if h < theta and g < ODEThreeCompHydSimulator.eps:
                    func = ODEThreeCompHydSimulator.fAe
                # fAe_rAnS
                elif h < g + theta and g > ODEThreeCompHydSimulator.eps:
                    func = ODEThreeCompHydSimulator.fAe_rAn
                # fAe_lAnS
                elif h > g + theta and h < 1 - gamma:
                    func = ODEThreeCompHydSimulator.fAe_lAn
                # fAe_fAnS
                elif h > 1 - gamma:
                    func = ODEThreeCompHydSimulator.fAe_fAn
                else:
                    raise UserWarning(
                        "unhandled state with h {} g {} and conf theta {} gamma {} phi {}".format(h, g, theta, gamma,
                                                                                                  phi))
            else:
                # lAr
                if h < theta and g < ODEThreeCompHydSimulator.eps:
                    func = ODEThreeCompHydSimulator.lAe
                elif h < g + theta and g > ODEThreeCompHydSimulator.eps:
                    func = ODEThreeCompHydSimulator.lAe_rAn
                elif h > g + theta and h <= 1 - gamma:
                    func = ODEThreeCompHydSimulator.lAe_lAn
                elif h > 1 - gamma:
                    func = ODEThreeCompHydSimulator.lAe_fAn
                else:
                    raise UserWarning(
                        "unhandled state with h {} g {} and conf theta {} gamma {} phi {}".format(h, g, theta, gamma,
                                                                                                  phi))

        if log_level > 1:
            logging.info("[intermediate result] Try {}".format(func))

        # save previous time to estimate time difference
        t_p = t

        # iterate through all phases until end is reached
        t, h, g, n_func = func(t, h, g, p_rec, t_max=t_rec, conf=conf)

        # double-check with discrete agent
        for _ in range(int(round((t - t_p) * hz))):
            agent.set_power(p_rec)
            agent.perform_one_step()
        g_diff = agent.get_g() - g
        h_diff = agent.get_h() - h

        if log_level > 1:
            logging.info("[intermediate result] {}\n"
                         "t {}\n"
                         "h {} g {}\n"
                         "Diff h {}\n"
                         "Diff g {}".format(func, t, h, g, h_diff, g_diff))
            ThreeCompVisualisation(agent)

        assert abs(g_diff) < eps, "{} g is off by {}".format(func, g_diff)
        assert abs(h_diff) < eps, "{} h is off by {}".format(func, h_diff)

        if t >= t_rec:
            logging.info("Max recovery reached in {}".format(func))
            # ThreeCompVisualisation(agent)
            break

        # exit loop if end of iteration is reached
        if n_func is None:
            logging.info("END IN {}: t: {} h: {} g: {}".format(func, t, h, g))
            break

        func = n_func

    if log_level > 0:
        # display fill levels after recovery
        logging.info("[SUCCESS] Agent after recovery at {} for {}".format(p_rec, t_rec))
        ThreeCompVisualisation(agent)


def the_loop(p_exp: float = 350.0, p_rec: float = 100.0, t_rec=180.0,
             t_max: float = 240, hz: int = 250, eps: float = 0.001):
    """
    creates random agents and tests the discretised against the differential one
    """

    # create random examples for all eternity
    while True:
        udp = MultiObjectiveThreeCompUDP(None, None)
        example_conf = udp.create_educated_initial_guess()
        logging.info(example_conf)
        rec_phase_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max,
                            hz=hz, eps=eps, conf=example_conf)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 260
    t_rec = 240
    p_rec = 10
    t_max = 6000

    # estimations per second for discrete agent
    hz = 500
    # required precision of discrete to differential agent
    eps = 0.001

    # a configuration
    c = [15101.24769778409, 86209.27743067988, 252.71702367096788,
         363.2970828395908, 38.27073086773415, 0.0, 0.3524379644134216, 1.0]

    rec_phase_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max,
                        hz=hz, eps=eps, conf=c, log_level=2)

    the_loop(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max, hz=hz, eps=eps)
