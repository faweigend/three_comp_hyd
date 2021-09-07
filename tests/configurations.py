import logging

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
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
     38.27073086773415, 0.64892228099402588,  # m_anf, theta
     0.1580228306857272, 0.6580228306857272]  # gamma, phi

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    hz = 10
    configs = [a, b, c, d]

    for conf in configs:
        # create three component hydraulic agent with example configuration
        agent = ThreeCompHydAgent(hz=hz,
                                  a_anf=conf[0], a_ans=conf[1],
                                  m_ae=conf[2], m_ans=conf[3],
                                  m_anf=conf[4], the=conf[5],
                                  gam=conf[6], phi=conf[7])
        ThreeCompVisualisation(agent)
