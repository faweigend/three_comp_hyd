import logging
import matplotlib as mpl
from pypermod.utility import PlotLayout

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

PlotLayout.set_rc_params()

p = [11220.356531171583,
     21516.06360371364,
     238.17395911833233,
     57.12368620643582,
     6.193546462332192,
     0.242533769054134,
     0.27055182889336115,
     0.23158433943054582]

demo_p = [3460, 6800, 200, 200, 6800000, 0.4, 0.23, 0.18]

agent = ThreeCompHydAgent(hz=10,
                          lf=p[0], ls=p[1],
                          m_u=p[2], m_ls=p[3], m_lf=p[4],
                          the=p[5], gam=p[6], phi=p[7])

# Three comp base vis
ThreeCompVisualisation(agent=agent,
                       detail_annotations=True,
                       black_and_white=False,
                       all_outlines=True)
