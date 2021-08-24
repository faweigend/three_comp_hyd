import math

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from scipy import optimize

import logging
import warnings

import numpy as np

# warnings.filterwarnings("error")

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    p_exp = 350
    p_rec = 100
    t_rec = 240
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

    # PHASE A4
    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    theta = conf[5]
    gamma = conf[6]
    phi = conf[7]

    # derived from tte estimator and
    t4 = 0
    ht4 = 0.6475620355865783
    gt4 = 0.15679831105786776

    # b/a can be simplified as (p-m_ae)/(a_anf + a_ans)
    a = (a_anf + a_ans) * m_ans / (a_anf * a_ans * (1 - theta - gamma))
    b = (p_rec - m_ae) * m_ans / (a_anf * a_ans * (1 - theta - gamma))

    # derivative g'(t3) can be calculated manually
    dgt3 = m_ans * (ht3 - gt3 - theta) / (a_ans * (1 - theta - gamma))

    # which then allows to derive c1 and c2
    s_c1 = ((p_rec - m_ae) / (a_anf + a_ans) - dgt3) * np.exp(a * t3)
    s_c2 = (-t3 * b + dgt3) / a - (p_rec - m_ae) / ((a_anf + a_ans) * a) + gt3


    def a4_gt(t):
        # general solution for g(t)
        return t * (p_rec - m_ae) / (a_anf + a_ans) + s_c2 + s_c1 / a * np.exp(-a * t)


    def a4_dgt(t):
        # first derivative g'(t)
        return (p_rec - m_ae) / (a_anf + a_ans) - s_c1 * np.exp(-a * t)


    def a4_ht(t):
        # EQ(9) with constants for g(t) and g'(t)
        return a_ans * (1 - theta - gamma) / m_ans * a4_dgt(t) + a4_gt(t) + theta


    ht4 = 1 - gamma
    # check if equilibrium in this phase
    if ht3 <= a4_ht(np.inf) <= ht4:
        return np.inf, a4_ht(np.inf), a4_gt(np.inf)
    else:
        t4 = optimize.fsolve(lambda t: ht4 - a4_ht(t), x0=np.array([t3]))[0]
        return t4, ht4, a4_gt(t4)

    agent.reset()
    agent.set_g(gt4)
    agent.set_h(ht4)
    agent.set_power(p_rec)

    for i in range(int(rt6+1)):
        for s in range(agent.hz):
            agent.perform_one_step()

    print("h: ", a6_ht(rt6) - agent.get_h())
    print("g: ", a6_gt(rt6) - agent.get_g())