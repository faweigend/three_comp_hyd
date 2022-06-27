import logging

from threecomphyd.agents.two_comp_hyd_agent import TwoCompHydAgent
from threecomphyd.animator.two_comp_hyd_animation import TwoCompHydAnimation

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # create two component hydraulic agent with example configuration
    agent = TwoCompHydAgent(10, w_p=28000, m_u=240, phi=0.5, psi=0.3)

    # run the interactive animation
    ani = TwoCompHydAnimation(agent)
    ani.run()
