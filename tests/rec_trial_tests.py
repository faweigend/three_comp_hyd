from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging
import warnings

import numpy as np


# warnings.filterwarnings("error")


def rec_trial_procedure(p_exp, p_rec, t_rec, hz, eps, conf, agent, log_level=0):
    # Start with first time to exhaustion bout
    tte, h_tte, g_tte = ODEThreeCompHydSimulator.tte(p_exp=p_exp, conf=conf)

    # double-check with discrete agent
    for _ in range(int(round(tte * hz))):
        agent.set_power(p_exp)
        agent.perform_one_step()
    g_diff = agent.get_g() - g_tte
    h_diff = agent.get_h() - h_tte
    assert abs(g_diff) < eps, "TTE1 g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "TTE1 h is off by {}".format(h_diff)

    logging.info("TTE END t: {} h: {} g: {}".format(tte, abs(h_diff), abs(g_diff)))
    ThreeCompVisualisation(agent)

    # Now Recovery
    # A6
    a6_t, a6_h, a6_g = ODEThreeCompHydSimulator.rec_a6(t6=0, h6=h_tte, g6=g_tte,
                                                       p_rec=p_rec, t_rec=t_rec, conf=conf)
    # double-check with discrete agent
    for _ in range(int(round(a6_t * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - a6_g
    h_diff = agent.get_h() - a6_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)

    # A5
    a5_t, a5_h, a5_g = ODEThreeCompHydSimulator.rec_a5(t5=a6_t, h5=a6_h, g5=a6_g,
                                                       p_rec=p_rec, t_rec=t_rec, conf=conf)
    # double-check with discrete agent
    for _ in range(int(round((a5_t - a6_t) * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - a5_g
    h_diff = agent.get_h() - a5_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)

    # A4R1
    a4r1_t, a4r1_h, a4r1_g = ODEThreeCompHydSimulator.rec_a4_r1(t4=a5_t, h4=a5_h, g4=a5_g,
                                                                p_rec=p_rec, t_rec=t_rec, conf=conf)
    # double-check with discrete agent
    for _ in range(int(round((a4r1_t - a5_t) * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - a4r1_g
    h_diff = agent.get_h() - a4r1_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)

    # A4R2
    a4r2_t, a4r2_h, a4r2_g = ODEThreeCompHydSimulator.rec_a4_r2(t4=a4r1_t, h4=a4r1_h, g4=a4r1_g,
                                                                p_rec=p_rec, t_rec=t_rec, conf=conf)
    # double-check with discrete agent
    for _ in range(int(round((a4r2_t - a4r1_t) * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - a4r2_g
    h_diff = agent.get_h() - a4r2_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)

    # A3R1
    a3r1_t, a3r1_h, a3r1_g = ODEThreeCompHydSimulator.rec_a3_r1(t3=a4r2_t, h3=a4r2_h, g3=a4r2_g,
                                                                p_rec=p_rec, t_rec=t_rec, conf=conf)
    # double-check with discrete agent
    for _ in range(int(round((a3r1_t - a4r2_t) * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - a3r1_g
    h_diff = agent.get_h() - a3r1_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)

    # A3R2
    a3r2_t, a3r2_h, a3r2_g = ODEThreeCompHydSimulator.rec_a3_r2(t3=a3r1_t, h3=a3r1_h, g3=a3r1_g,
                                                                p_rec=p_rec, t_rec=t_rec, conf=conf)
    # double-check with discrete agent
    for _ in range(int(round((a3r2_t - a3r1_t) * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - a3r2_g
    h_diff = agent.get_h() - a3r2_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)

    # A2
    a2_t, a2_h = ODEThreeCompHydSimulator.rec_a2(t2=a3r2_t, h2=a3r2_h,
                                                 p_rec=p_rec, t_rec=t_rec, conf=conf)
    # double-check with discrete agent
    for _ in range(int(round((a2_t - a3r2_t) * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - 0
    h_diff = agent.get_h() - a2_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)

    # A1
    a1_t, a1_h = ODEThreeCompHydSimulator.rec_a1(t1=a2_t, h1=a2_h,
                                                 p_rec=p_rec, t_rec=t_rec, conf=conf)

    print(a1_t, a1_h)
    # double-check with discrete agent
    for _ in range(int(round((a1_t - a2_t) * hz))):
        agent.set_power(p_rec)
        agent.perform_one_step()
    g_diff = agent.get_g() - 0
    h_diff = agent.get_h() - a1_h
    assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
    assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)
    ThreeCompVisualisation(agent)


def the_loop(p_exp: float = 350.0, p_rec: float = 100.0,
             t_rec: int = 240, hz: int = 250, eps: float = 0.001):
    """
    creates random agents and tests the discretised against the differential one
    """

    while True:
        udp = MultiObjectiveThreeCompUDP(None, None)

        example_conf = udp.create_educated_initial_guess()
        example_conf = [15101.24769778409, 86209.27743067988, 52.71702367096787, 363.2970828395908, 38.27073086773415,
                        0.14892228099402588, 0.3524379644134216, 0.4580228306857272]
        logging.info(example_conf)
        # create three component hydraulic agent with example configuration
        agent = ThreeCompHydAgent(hz=hz, a_anf=example_conf[0], a_ans=example_conf[1], m_ae=example_conf[2],
                                  m_ans=example_conf[3], m_anf=example_conf[4], the=example_conf[5],
                                  gam=example_conf[6], phi=example_conf[7])

        ThreeCompVisualisation(agent)

        rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec,
                            hz=hz, eps=eps, conf=example_conf,
                            agent=agent, log_level=2)

        break


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 0
    t_rec = 5000

    # estimations per second for discrete agent
    hz = 250
    # required precision of discrete to differential agent
    eps = 0.005

    # a C configuration
    c = [15101.24769778409, 86209.27743067988,  # anf, ans
         252.71702367096787, 363.2970828395908,  # m_ae, m_ans
         38.27073086773415, 0.14892228099402588,  # m_anf, theta
         0.1580228306857272, 0.3580228306857272]  # gamma, phi

    agent = ThreeCompHydAgent(hz=hz, a_anf=c[0], a_ans=c[1], m_ae=c[2],
                              m_ans=c[3], m_anf=c[4], the=c[5],
                              gam=c[6], phi=c[7])

    rec_trial_procedure(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec,
                        hz=hz, eps=eps, conf=c,
                        agent=agent, log_level=2)

    # the_loop(p_exp=p_exp, p_rec=p_rec, t_rec=t_rec, hz=hz, eps=eps)
