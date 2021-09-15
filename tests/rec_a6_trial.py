import math

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

from tests import configurations

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 100
    t_max = 5000

    # estimations per second for discrete agent
    hz = 250

    conf = configurations.d

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=conf[0], a_ans=conf[1], m_ae=conf[2],
                              m_ans=conf[3], m_anf=conf[4], the=conf[5],
                              gam=conf[6], phi=conf[7])

    # PHASE A6
    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    t6 = 0
    ht6 = 1.0
    gt6 = 0.1

    # g(t6) = gt6 can be solved for c
    s_cg = (gt6 - (1 - theta - gamma)) / np.exp(-m_ans * t6 / ((1 - theta - gamma) * a_ans))


    def a6_gt(t):
        # generalised g(t) for phase A6
        return (1 - theta - gamma) + s_cg * math.exp((-m_ans * t) / ((1 - theta - gamma) * a_ans))


    k = m_ans / ((1 - theta - gamma) * a_ans)
    # a = -m_ae / a_anf
    b = (m_ans * s_cg) / ((1 - theta - gamma) * a_anf)
    # g = p / a_anf
    ag = (p_rec - m_ae) / a_anf

    # h(t6) = 1 can be solved for c
    s_ch = -t6 * ag + ((b * math.exp(-k * t6)) / k) + ht6


    def a6_ht(t):
        # generalised h(t) for recovery phase A6
        return t * ag - ((b * math.exp(-k * t)) / k) + s_ch


    # A6 rec ends either at beginning of A4 or A5
    h_target = max(1 - gamma, 1 - phi)

    # estimate an initial guess that assumes no contribution from g
    initial_guess = 0

    import time

    t0 = time.process_time_ns()
    rt6 = ODEThreeCompHydSimulator.optimize(func=lambda t: a6_ht(t) - h_target,
                                            initial_guess=initial_guess,
                                            max_steps=t_max)
    t1 = time.process_time_ns()
    print("Time elapsed own: ", t1 - t0)  # CPU seconds elapsed (floating point)

    t0 = time.process_time_ns()
    optimize.fsolve(lambda t: a6_ht(t) - h_target, x0=np.array([initial_guess]))
    t1 = time.process_time_ns()
    print("Time elapsed scipy: ", t1 - t0)  # CPU seconds elapsed (floating point)

    agent.reset()
    agent.set_g(gt6)
    agent.set_h(1.0)
    ThreeCompVisualisation(agent)
    agent.set_power(p_rec)

    for _ in range(int(rt6 * agent.hz)):
        agent.perform_one_step()

    logging.info("predicted time: {} \n"
                 "diff h: {}\n"
                 "diff g: {}".format(rt6,
                                     a6_ht(rt6) - agent.get_h(),
                                     a6_gt(rt6) - agent.get_g()))
    ThreeCompVisualisation(agent)
