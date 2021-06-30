import logging
import warnings

warnings.filterwarnings("error")

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator


def test_procedure(hz, eps, conf):
    a_anf = conf[0]
    a_ans = conf[1]
    m_ae = conf[2]
    m_ans = conf[3]
    m_anf = conf[4]
    the = conf[5]
    gam = conf[6]
    phi = conf[7]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae, m_ans=m_ans,
                              m_anf=m_anf, the=the, gam=gam, phi=phi)

    # ThreeCompVisualisation(agent)

    p = 300

    # all TTE phases
    try:
        t1 = ODEThreeCompHydSimulator.phase_a1(p=p, a_anf=a_anf, m_ae=m_ae, theta=the, phi=phi)
        t2 = ODEThreeCompHydSimulator.phase_a2(t1=t1, p=p, a_anf=a_anf, m_ae=m_ae, theta=the, phi=phi)
    except ValueError:
        logging.info("A1/2 error with {}".format(example_conf))
        return

    try:
        t3, gt3 = ODEThreeCompHydSimulator.phase_a3(t2=t2, p=p, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae,
                                                    m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    except RuntimeWarning:
        logging.info("A3 warning with {}".format(example_conf))
        return

    try:
        t4, gt4 = ODEThreeCompHydSimulator.phase_a4(t3=t3, gt3=gt3, p=p, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae,
                                                    m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    except RuntimeWarning:
        logging.info("A4 warning with {}".format(example_conf))
        return

    try:
        t5, gt5 = ODEThreeCompHydSimulator.phase_a5(t4=t4, gt4=gt4, p=p, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae,
                                                    m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    except RuntimeWarning:
        logging.info("A5 warning with {}".format(example_conf))
        return

    try:
        t6, gt6 = ODEThreeCompHydSimulator.phase_a6(t5=t5, gt5=gt5, p=p, a_anf=a_anf, a_ans=a_ans, m_ae=m_ae,
                                                    m_ans=m_ans, theta=the, gamma=gam, phi=phi)
    except RuntimeWarning:
        logging.info("A6 warning with {}".format(example_conf))
        return

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

    test_procedure(hz, eps, example_conf)
