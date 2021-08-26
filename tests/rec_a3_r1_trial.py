from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 100

    # estimations per second for discrete agent
    hz = 250

    conf = [15101.24769778409, 86209.27743067988, 52.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz,
                              a_anf=conf[0], a_ans=conf[1],
                              m_ae=conf[2], m_ans=conf[3],
                              m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    t3_exp = 200.58698045085373
    t3 = 0
    ht3 = 0.7294083375575103
    gt3 = 0.31236471544758654

    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    # my simplified form
    a = m_ae / (a_anf * (1 - phi)) + \
        m_ans / (a_ans * (1 - theta - gamma)) + \
        m_ans / (a_anf * (1 - theta - gamma))

    b = m_ae * m_ans / \
        (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

    c = m_ans * (p_exp * (1 - phi) - m_ae * theta) / \
        (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))
    
    # a new c' for p_rec
    c_p = m_ans * (p_rec * (1 - phi) - m_ae * theta) / \
          (a_anf * a_ans * (1 - phi) * (1 - theta - gamma))

    # wolfram alpha gave these estimations as solutions for l''(t) + a*l'(t) + b*l(t) = c
    r1 = 0.5 * (-np.sqrt(a ** 2 - 4 * b) - a)
    r2 = 0.5 * (np.sqrt(a ** 2 - 4 * b) - a)

    # derive the solution from a tte with p_exp
    s_c1 = -c / (b * (1 - r1 / r2)) * np.exp(-r1 * t3_exp)
    s_c2 = s_c1 * np.exp(r1 * t3_exp) * np.exp(-r2 * t3_exp) * -r1 / r2

    # use tte solution to determine c1' and c2'
    s_c1_p = (gt3 - c_p / b) / ((1 - r1 / r2) * np.exp(-r1 * t3))
    s_c2_p = (r1 * s_c1 * np.exp(r1 * t3_exp) +
              r2 * s_c2 * np.exp(r2 * t3_exp) -
              r1 * s_c1_p * np.exp(r1 * t3_exp)) / \
             (r2 * np.exp(r2 * t3_exp))

    def a3_gt(t):
        # the general solution for g(t)
        return s_c1_p * np.exp(r1 * t) + s_c2_p * np.exp(r2 * t) + c / b


    # substitute into EQ(9) for h
    def a3_ht(t):
        k1 = a_ans * (1 - theta - gamma) / m_ans * s_c1_p * r1 + s_c1_p
        k2 = a_ans * (1 - theta - gamma) / m_ans * s_c2_p * r2 + s_c2_p
        return k1 * np.exp(r1 * t) + k2 * np.exp(r2 * t) + c / b + theta


    eq_gh = optimize.fsolve(lambda t: (a3_gt(t) + theta) - a3_ht(t), x0=np.array([0]))[0]


    print(a3_gt(0) + theta, a3_ht(0))
    print(gt3 + theta, ht3)

    print(eq_gh)
    print(a3_gt(eq_gh) + theta, a3_ht(eq_gh))

    agent.reset()
    agent.set_g(gt3)
    agent.set_h(ht3)
    agent.set_power(p_rec)

    # for i in range(int(eq_gh)):
    #     print("step {}".format(i))
    #     print("h: ", a3_ht(int(eq_gh)) - agent.get_h())
    #     print("g: ", a3_gt(int(eq_gh)) - agent.get_g())
    #     for s in range(agent.hz):
    #         agent.perform_one_step()
    #
    # print("h: ", a3_ht(int(eq_gh)) - agent.get_h())
    # print("g: ", a3_gt(int(eq_gh)) - agent.get_g())
