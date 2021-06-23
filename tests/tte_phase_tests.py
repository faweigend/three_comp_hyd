import logging
import math

import numpy as np
from scipy import optimize
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    hz = 1000

    example_conf = [11532.526538727172,
                    23240.257042239595,
                    249.7641585019016,
                    286.26673813946095,
                    7.988078323028352,
                    0.25486842730772163,
                    0.26874299216869681,
                    0.2141766056862277
                    ]

    a_anf = example_conf[0]
    a_ans = example_conf[1]
    m_ae = example_conf[2]
    m_ans = example_conf[3]
    m_anf = example_conf[4]
    the = example_conf[5]
    gam = example_conf[6]
    phi = example_conf[7]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae, m_ans=m_ans,
                              m_anf=m_anf, the=the, gam=gam, phi=phi)

    p = 250

    # end of phase A1 -> the time whe h=theta
    t1 = (-(a_anf * (1 - phi)) / m_ae) * math.log(1 - ((m_ae * the) / (p * (1 - phi))))

    # simulate with the agent for estimated amount of time steps
    agent.reset()
    for _ in range(int(t1 * hz)):
        agent.set_power(p)
        agent.perform_one_step()

    # check if t1 time estimation lines up
    assert agent.get_h() - the < 0.01

    # taken from Equation 11 by Morton 1986
    a = (m_ae * a_ans * (1 - the - gam) + m_ans * (a_anf + a_ans) * (1 - phi)) / (
            a_anf * a_ans * (1 - phi) * (1 - the - gam))

    b = (m_ae * m_ans) / (a_anf * a_ans * (1 - phi) * (1 - the - gam))

    c = (m_ans * (p * (1 - phi) - m_ae * the)) / (a_anf * a_ans * (1 - phi) * (1 - the - gam))

    # use auxillary equation to derive both negative roots r1 and r2
    coeff = [1, a, b]
    r1, r2 = np.roots(coeff)


    def eq_12_and_derivative(p):
        c1, c2 = p
        return (c1 * math.exp(r1 * t1) + c2 * math.exp(r2 * t1) + c / b,
                r1 * c1 * math.exp(r1 * t1) + r2 * c2 * math.exp(r2 * t1))


    # solve for c1 and c2
    res = optimize.fsolve(eq_12_and_derivative, x0=np.array([1, 1]))
    s_c1 = float(res[0])
    s_c2 = float(res[1])


    def eq_9_substituted(t):
        k1 = r1 * s_c1 * ((a_ans * (1 - the - gam)) / m_ans) + s_c1
        k2 = r2 * s_c2 * ((a_ans * (1 - the - gam)) / m_ans) + s_c2
        h = k1 * math.exp(r1 * t) + k2 * math.exp(r2 * t) + c / b + the
        return h - (1 - gam)


    t2 = optimize.fsolve(eq_9_substituted, x0=np.array([1]))[0]

    # simulate with the agent for estimated amount of time steps
    agent.reset()
    for _ in range(int((t2) * hz)):
        agent.set_power(p)
        agent.perform_one_step()

    # check if t1 time estimation lines up
    assert agent.get_h() - (1 - gam) < 0.01
