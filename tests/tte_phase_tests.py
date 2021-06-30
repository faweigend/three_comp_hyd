from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging
import warnings

import numpy as np

warnings.filterwarnings("error")


def test_procedure(hz, eps, conf, agent):
    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    the = conf[5]
    gam = conf[6]
    phi = conf[7]

    p = 350

    # all TTE phases
    t1, ht1, gt1 = ODEThreeCompHydSimulator.phase_a1(p=p, a_anf=a_anf, m_ae=m_ae, theta=the, phi=phi)
    if t1 == np.inf:
        logging.info("EQUILIBRIUM IN A1: t: {} h: {} g: {}".format(t1, ht1, gt1))
        return
    logging.info("A1: {}".format(t1))

    t2, ht2, gt2 = ODEThreeCompHydSimulator.phase_a2(t1=t1, ht1=ht1, gt1=gt1, p=p, a_anf=a_anf, m_ae=m_ae, theta=the,
                                                     phi=phi)
    logging.info("A2: {}".format(t2))

    t3, ht3, gt3 = ODEThreeCompHydSimulator.phase_a3(t2=t2, ht2=ht2, gt2=gt2, p=p, a_anf=a_anf, a_ans=a_ans,
                                                     m_ae=m_ae, m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    if t3 == np.inf:
        logging.info("EQUILIBRIUM IN A3: t: {} h: {} g: {}".format(t3, ht3, gt3))
        return
    logging.info("A3: {}".format(t3))

    t4, ht4, gt4 = ODEThreeCompHydSimulator.phase_a4(t3=t3, ht3=ht3, gt3=gt3, p=p, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae,
                                                     m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    if t4 == np.inf:
        logging.info("EQUILIBRIUM IN A4: t: {} h: {} g: {}".format(t4, ht4, gt4))
        return
    logging.info("A4: {}".format(t4))

    t5, ht5, gt5 = ODEThreeCompHydSimulator.phase_a5(t4=t4, ht4=ht4, gt4=gt4, p=p, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae,
                                                     m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    if t5 == np.inf:
        logging.info("EQUILIBRIUM IN A5: t: {} h: {} g: {}".format(t5, ht5, gt5))
        return
    logging.info("A5: {}".format(t5))

    t6, ht6, gt6 = ODEThreeCompHydSimulator.phase_a6(t5=t5, ht5=ht5, gt5=gt5, p=p, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae,
                                                     m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    if t6 == np.inf:
        logging.info("EQUILIBRIUM IN A6: t: {} h: {} g: {}".format(t6, ht6, gt6))
        return
    logging.info("A6: {}".format(t6))

    # A1
    agent.reset()
    for _ in range(int(t1 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - min(the, 1 - phi)) < eps
    # A3
    agent.reset()
    for _ in range(int(t3 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    if phi >= (1 - the):
        assert abs(agent.get_h() - the) < eps
    else:
        assert abs(agent.get_h() - (1 - max(phi, gam))) < eps
    # A4
    agent.reset()
    for _ in range(int(t4 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - (1 - gam)) < eps
    # A5
    agent.reset()
    for _ in range(int(t5 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    if phi >= gam:
        assert abs(agent.get_h() - (1 - gam)) < eps
    else:
        # A5 is only different from t4 if phi >= gamma
        assert abs(agent.get_h() - (1 - phi)) < eps
    # A6
    agent.reset()
    for _ in range(int(t6 * hz)):
        agent.set_power(p)
        agent.perform_one_step()
    assert abs(agent.get_h() - 1) < eps

    print(t6)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # estimations per second for discrete agent
    hz = 100
    # required precision of discrete to differential agent
    eps = 0.001

    udp = MultiObjectiveThreeCompUDP(None, None)

    example_conf = udp.create_educated_initial_guess()

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=example_conf[0], a_ans=example_conf[1], m_ae=example_conf[2],
                              m_ans=example_conf[3], m_anf=example_conf[4], the=example_conf[5],
                              gam=example_conf[6], phi=example_conf[7])

    ThreeCompVisualisation(agent)

    test_procedure(hz, eps, example_conf, agent)
