from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.evolutionary_fitter.three_comp_tools import MultiObjectiveThreeCompUDP
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

import logging
import warnings

import numpy as np


# warnings.filterwarnings("error")


def test_procedure(hz, eps, conf, agent):
    p = 350

    # all TTE phases

    # A1
    t1, ht1, gt1 = ODEThreeCompHydSimulator.phase_a1(p=p, conf=conf)
    if t1 == np.inf:
        logging.info("EQUILIBRIUM IN A1: t: {} h: {} g: {}".format(t1, ht1, gt1))
        return
    logging.info("A1: {}".format(t1))

    t2, ht2, gt2 = ODEThreeCompHydSimulator.phase_a2(t1=t1, ht1=ht1, gt1=gt1, p=p, conf=conf)
    logging.info("A2: {}".format(t2))

    # A3
    t3, ht3, gt3 = ODEThreeCompHydSimulator.phase_a3(t2=t2, ht2=ht2, gt2=gt2, p=p, conf=conf)
    if t3 == np.inf:
        logging.info("EQUILIBRIUM IN A3: t: {} h: {} g: {}".format(t3, ht3, gt3))
        return
    logging.info("A3: {}".format(t3))

    # A4
    t4, ht4, gt4 = ODEThreeCompHydSimulator.phase_a4(t3=t3, ht3=ht3, gt3=gt3, p=p, conf=conf)
    if t4 == np.inf:
        logging.info("EQUILIBRIUM IN A4: t: {} h: {} g: {}".format(t4, ht4, gt4))
        return
    logging.info("A4: {}".format(t4))

    # A5
    t5, ht5, gt5 = ODEThreeCompHydSimulator.phase_a5(t4=t4, ht4=ht4, gt4=gt4, p=p, conf=conf)
    if t5 == np.inf:
        logging.info("EQUILIBRIUM IN A5: t: {} h: {} g: {}".format(t5, ht5, gt5))
        return
    logging.info("A5: {}".format(t5))

    # A6
    t6, ht6, gt6 = ODEThreeCompHydSimulator.phase_a6(t5=t5, ht5=ht5, gt5=gt5, p=p, conf=conf)
    if t6 == np.inf:
        logging.info("EQUILIBRIUM IN A6: t: {} h: {} g: {}".format(t6, ht6, gt6))
        return
    logging.info("A6: {}".format(t6))

    ts = [t1, t2, t3, t4, t5, t6]
    hts = [ht1, ht2, ht3, ht4, ht5, ht6]
    gts = [gt1, gt2, gt3, gt4, gt5, gt6]

    for i, t in enumerate(ts):
        try:
            agent.reset()
            for _ in range(int(t * hz)):
                agent.set_power(p)
                agent.perform_one_step()

            g_diff = agent.get_g() - gts[i]
            h_diff = agent.get_h() - hts[i]
            assert abs(h_diff) < eps, "error phase {}. h is off by {}".format(i + 1, h_diff)
            assert abs(g_diff) < eps, "error phase {}. g is off by {}".format(i + 1, g_diff)
        except AssertionError as e:
            logging.info(e)

    logging.info(conf)


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # estimations per second for discrete agent
    hz = 50
    # required precision of discrete to differential agent
    eps = 0.001

    udp = MultiObjectiveThreeCompUDP(None, None)

    example_conf = udp.create_educated_initial_guess()
    # example_conf = [23300.62830028783, 55208.70561675477, 81.80334699080105, 4484.892862385766, 28.970996944175734,
    #                 0.24068111663641112, 0.2755929224243251, 0.5794389684110414]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(hz=hz, a_anf=example_conf[0], a_ans=example_conf[1], m_ae=example_conf[2],
                              m_ans=example_conf[3], m_anf=example_conf[4], the=example_conf[5],
                              gam=example_conf[6], phi=example_conf[7])

    ThreeCompVisualisation(agent)

    test_procedure(hz, eps, example_conf, agent)
