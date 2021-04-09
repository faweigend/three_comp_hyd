import logging

from animate.three_comp_interactive_animation import ThreeCompInteractiveAnimation
from agents.three_comp_hyd_agent import ThreeCompHydAgent

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # example configuration
    params = [11532.526538727172,
              23240.257042239595,
              249.7641585019016,
              286.26673813946095,
              7.988078323028352,
              0.25486842730772163,
              0.26874299216869681,
              0.2141766056862277]

    # create three component hydraulic agent with example configuration
    agent = ThreeCompHydAgent(10, a_anf=params[0], a_ans=params[1], m_ae=params[2],
                              m_ans=params[3], m_anf=params[4], the=params[5], gam=params[6],
                              phi=params[7])

    # run the interactive animation
    ani = ThreeCompInteractiveAnimation(agent)
    ani.run()
