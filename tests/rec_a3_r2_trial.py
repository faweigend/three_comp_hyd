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
    p_rec = 0
    t_max = 5000

    # estimations per second for discrete agent
    hz = 250

    conf = [10101.24769778409, 80209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              a_anf=conf[0], a_ans=conf[1],
                              m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    # End of A3 R1
    t3 = 0
    ht3 = 0.2
    gt3 = 0.2

    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    # EQ 16 and 17 substituted in EQ 8
    a = m_ae / (a_anf * (1 - phi)) + \
        m_anf / (a_ans * (1 - gamma)) + \
        m_anf / (a_anf * (1 - gamma))

    b = m_ae * m_anf / \
        (a_anf * a_ans * (1 - phi) * (1 - gamma))

    # c = (p_rec - (m_ae * theta) / (1 - phi)) * m_anf / \
    #     (a_anf * a_ans * (1 - gamma))
    c = m_anf * (p_rec * (1 - phi) - m_ae * theta) / \
        (a_anf * a_ans * (1 - phi) * (1 - gamma))

    # wolfram alpha gave these estimations as solutions for l''(t) + a*l'(t) + b*l(t) = c
    r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
    r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

    # uses Al dt/dl part of EQ(16) == dl/dt of EQ(14) solved for c2
    # and then substituted in EQ(14) and solved for c1
    s_c1 = (c / b - (m_anf * (gt3 + theta - ht3)) / (a_ans * r2 * (1 - gamma)) - gt3) / \
           (np.exp(r1 * t3) * (r1 / r2 - 1))

    # uses EQ(14) with solution for c1 and solves for c2
    s_c2 = (gt3 - s_c1 * np.exp(r1 * t3) - c / b) / np.exp(r2 * t3)

    def a3_gt(t):
        # the general solution for g(t)
        return s_c1 * np.exp(r1 * t) + s_c2 * np.exp(r2 * t) + c / b

    # substitute into EQ(9) for h
    def a3_ht(t):
        k1 = a_ans * (1 - gamma) / m_anf * s_c1 * r1 + s_c1
        k2 = a_ans * (1 - gamma) / m_anf * s_c2 * r2 + s_c2
        return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta

    # use the quickest possible recovery as the initial guess (assumes h=0)
    in_c1 = (gt3 + theta) * np.exp(-m_anf * t3 / ((gamma - 1) * a_ans))
    in_t = (gamma - 1) * a_ans * (np.log(theta) - np.log(in_c1)) / m_anf

    # find the point where g(t) == 0
    import time

    t0 = time.process_time_ns()
    g0 = ODEThreeCompHydSimulator.optimize(func=lambda t: a3_gt(t),
                                           initial_guess=t3,
                                           max_steps=t_max)
    t1 = time.process_time_ns()
    print("Time elapsed own: ", t1 - t0)  # CPU seconds elapsed (floating point)

    t0 = time.process_time_ns()
    optimize.fsolve(lambda t: a3_gt(t), x0=np.array([in_t]))[0]
    t1 = time.process_time_ns()
    print("Time elapsed scipy: ", t1 - t0)  # CPU seconds elapsed (floating point)

    # check in simulation
    agent.reset()
    agent.set_g(gt3)
    agent.set_h(ht3)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for i in range(int(g0 * agent.hz)):
        agent.perform_one_step()

        if i % agent.hz == 0:
            test_t = i / agent.hz
            logging.info("predicted time: {} \n"
                         "diff h: {} - {} = {}\n"
                         "diff g: {} - {} = {}".format(test_t,
                                                       a3_ht(test_t),
                                                       agent.get_h(),
                                                       a3_ht(test_t) - agent.get_h(),
                                                       a3_gt(test_t),
                                                       agent.get_g(),
                                                       a3_gt(test_t) - agent.get_g()))

    logging.info("predicted time: {} \n"
                 "diff h: {} - {} = {}\n"
                 "diff g: {} - {} = {}".format(g0,
                                               a3_ht(g0),
                                               agent.get_h(),
                                               a3_ht(g0) - agent.get_h(),
                                               a3_gt(g0),
                                               agent.get_g(),
                                               a3_gt(g0) - agent.get_g()))
    ThreeCompVisualisation(agent)
