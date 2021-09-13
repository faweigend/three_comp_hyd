import logging

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

# an A configuration
a = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.14892228099402588,  # m_anf, theta
     0.3524379644134216, 0.1580228306857272]  # gamma, phi
# a B configuration
b = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.14892228099402588,  # m_anf, theta
     0.2580228306857272, 0.2580228306857272]  # gamma, phi
# a C configuration
c = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.14892228099402588,  # m_anf, theta
     0.1580228306857272, 0.3580228306857272]  # gamma, phi
# a D configuration
d = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.64892228099402588,  # m_anf, theta
     0.1580228306857272, 0.6580228306857272]  # gamma, phi

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    hz = 250
    p_exp = 450
    p_rec = 100
    t_rec = 150
    eps = 0.005

    configs = [
        # a,
        # b,
        # c,
        d
    ]

    for conf in configs:
        # create three component hydraulic agent with example configuration
        agent = ThreeCompHydAgent(hz=hz,
                                  a_anf=conf[0], a_ans=conf[1],
                                  m_ae=conf[2], m_ans=conf[3],
                                  m_anf=conf[4], the=conf[5],
                                  gam=conf[6], phi=conf[7])
        ThreeCompVisualisation(agent)

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

        # Now recovery
        rec, h_rec, g_rec = ODEThreeCompHydSimulator.rec(conf=conf, start_h=h_tte,
                                                         start_g=g_tte, p_rec=p_rec,
                                                         t_rec=t_rec)

        # double-check with discrete agent
        for _ in range(int(round(rec * hz))):
            agent.set_power(p_rec)
            agent.perform_one_step()
        g_diff = agent.get_g() - g_rec
        h_diff = agent.get_h() - h_rec
        assert abs(g_diff) < eps, "REC g is off by {}".format(g_diff)
        assert abs(h_diff) < eps, "REC h is off by {}".format(h_diff)

        logging.info("REC END t: {} h: {} g: {}".format(rec, abs(h_diff), abs(g_diff)))
        ThreeCompVisualisation(agent)
