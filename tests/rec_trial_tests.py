from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging


def rec_trial_procedure(p_exp, p_rec, t_rec, t_max, hz, eps, conf, log_level=0):
    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=conf[0], a_ans=conf[1], m_ae=conf[2],
                              m_ans=conf[3], m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    if log_level > 0:
        ThreeCompVisualisation(agent)

    # Start with first time to exhaustion bout
    tte_1, h, g = ODEThreeCompHydSimulator.constant_power_trial(conf=conf,
                                                                start_h=0, start_g=0,
                                                                p=p_exp, t_max=t_max)
    if tte_1 >= t_max:
        logging.info("Exhaustion not reached during TTE")
        return
    
    # confirm the tte time with an entire iterative simulation
    c_tte = ThreeCompHydSimulator.tte(agent, p_work=p_exp, t_max=t_max)
    assert abs(c_tte - tte_1) < eps, "TTE1 confirmation error. Difference between " \
                                     "ODE TTE {} and Iterative TTE {} is {}".format(tte_1,
                                                                                    c_tte,
                                                                                    abs(c_tte - tte_1))

    agent.reset()
    # double-check with discrete agent
    for _ in range(int(round(tte_1 * hz))):
        agent.set_power(p_exp)
        agent.perform_one_step()
    g_diff = agent.get_g() - g
    h_diff = agent.get_h() - h
    assert abs(g_diff) < eps, "TTE1 g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "TTE1 h is off by {}".format(h_diff)

    if log_level > 0:
        logging.info("TTE1 {} h: {} g: {}".format(tte_1, h, g))
        ThreeCompVisualisation(agent)

    rec, h, g = ODEThreeCompHydSimulator.constant_power_trial(conf=conf,
                                                              start_h=h, start_g=g,
                                                              p=p_rec, t_max=t_rec)
    # double-check with discrete agent
    for _ in range(int(round(t_rec * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - g
    h_diff = agent.get_h() - h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)

    if log_level > 0:
        logging.info("REC {} h: {} g: {}".format(rec, h, g))
        ThreeCompVisualisation(agent)

    # confirm the tte time with an entire iterative simulation
    tte_2, h2, g2 = ODEThreeCompHydSimulator.constant_power_trial(conf=conf,
                                                                  start_h=h, start_g=g,
                                                                  p=p_exp, t_max=t_max)

    c_tte = ThreeCompHydSimulator.tte(agent, start_h=h, start_g=g,
                                      p_work=p_exp, t_max=t_max)
    assert abs(c_tte - tte_2) < eps, "TTE2 confirmation error. Difference betwen " \
                                     "ODE TTE {} and Iterative TTE {} is {}".format(tte_2,
                                                                                    c_tte,
                                                                                    abs(c_tte - tte_2))

    agent.reset()
    agent.set_h(h)
    agent.set_g(g)
    # double-check with discrete agent
    for _ in range(int(round(tte_2 * hz))):
        agent.set_power(p_exp)
        agent.perform_one_step()
    g_diff = agent.get_g() - g2
    h_diff = agent.get_h() - h2
    assert abs(g_diff) < eps, "TTE2 g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "TTE2 h is off by {}".format(h_diff)

    if log_level > 0:
        logging.info("TTE2 {} h: {} g: {}".format(tte_2, h2, g2))
        ThreeCompVisualisation(agent)

    # simulator step limit needs to be adjusted
    est_ratio = ThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(agent=agent,
                                                                 p_exp=p_exp,
                                                                 p_rec=p_rec,
                                                                 t_rec=t_rec,
                                                                 t_max=t_max)

    est_ratio_2 = ODEThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(conf=conf,
                                                                      p_exp=p_exp,
                                                                      p_rec=p_rec,
                                                                      t_rec=t_rec,
                                                                      t_max=t_max)
    rec_ratio = tte_2 / tte_1 * 100.0
    diff = abs(rec_ratio - est_ratio)
    diff2 = abs(rec_ratio - est_ratio_2)

    logging.info("ODE ratio: {}    Iterative ratio: {} ODE est ratio: {}".format(rec_ratio, est_ratio, est_ratio_2))
    assert diff < eps, "Ratio estimations are too different {}".format(diff)
    assert diff2 < eps, "Ratio estimations are too different {}".format(diff2)


def the_loop(p_exp: float = 350.0, p_rec: float = 100.0, t_rec=180.0,
             t_max: float = 240, hz: int = 250, eps: float = 0.001):
    """
    creates random agents and tests the discretised against the differential one
    """

    while True:
        udp = MultiObjectiveThreeCompUDP(None, None)

        example_conf = udp.create_educated_initial_guess()
        logging.info(example_conf)

        rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max,
                            hz=hz, eps=eps, conf=example_conf, log_level=0)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 260
    t_rec = 240
    p_rec = 10
    t_max = 5000

    # estimations per second for discrete agent
    hz = 1000
    # required precision of discrete to differential agent
    eps = 0.01

    # a configuration
    c = [15101.24769778409, 86209.27743067988, 252.71702367096788,
         363.2970828395908, 38.27073086773415, 0.0, 0.3524379644134216, 1.0]

    rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max,
                        hz=hz, eps=eps, conf=c, log_level=2)

    # the_loop(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, t_max=t_max, hz=hz, eps=eps)
