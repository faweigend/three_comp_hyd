from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 100
    t_max = 5000

    # estimations per second for discrete agent
    hz = 250

    conf = [15101.24769778409, 86209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              a_anf=conf[0], a_ans=conf[1],
                              m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    # PHASE A4
    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    # derived from tte estimator and
    t4 = 0
    ht4 = 0.6475620355865783
    gt4 = 0.15679831105786776

    # if g is above h (flow from AnS into AnF)
    a_gh = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
    b_gh = (p_rec - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

    # derivative g'(t4) can be calculated manually
    dgt4_gh = m_ans * (ht4 - gt4 - theta) / (a_ans * (1 - theta - gamma))

    # which then allows to derive c1 and c2
    s_c1_gh = ((p_rec - m_ae) / (a_anf + a_ans) - dgt4_gh) * np.exp(a_gh * t4)
    s_c2_gh = (-t4 * b_gh + dgt4_gh) / a_gh - (p_rec - m_ae) / ((a_anf + a_ans) * a_gh) + gt4


    def a4_gt(t):
        # general solution for g(t)
        return t * (p_rec - m_ae) / (a_anf + a_ans) + s_c2_gh + s_c1_gh / a_gh * np.exp(-a_gh * t)


    def a4_dgt(t):
        # first derivative g'(t)
        return (p_rec - m_ae) / (a_anf + a_ans) - s_c1_gh * np.exp(-a_gh * t)


    def a4_ht(t):
        # EQ(9) with constants for g(t) and g'(t)
        return a_ans * (1 - theta - gamma) / m_ans * a4_dgt(t) + a4_gt(t) + theta


    ht_end = 1 - phi

    # check if equilibrium in this phase
    import time
    t0 = time.process_time_ns()
    t_end = ODEThreeCompHydSimulator.optimize(func=lambda t: a4_ht(t) - ht_end,
                                              initial_guess=0,
                                              max_steps=t_max)
    t1 = time.process_time_ns()
    print("Time elapsed own: ", t1 - t0)  # CPU seconds elapsed (floating point)

    t0 = time.process_time_ns()
    optimize.fsolve(lambda t: a4_ht(t) - ht_end, x0=np.array([0]))
    t1 = time.process_time_ns()
    print("Time elapsed scipy: ", t1 - t0)  # CPU seconds elapsed (floating point)

    gt_end = a4_gt(t_end)

    agent.reset()
    agent.set_g(gt4)
    agent.set_h(ht4)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for _ in range(int(t_end * agent.hz)):
        agent.perform_one_step()

    logging.info("predicted time: {} \n"
                 "diff h: {}\n"
                 "diff g: {}".format(t_end,
                                     a4_ht(t_end) - agent.get_h(),
                                     a4_gt(t_end) - agent.get_g()))
    ThreeCompVisualisation(agent)
