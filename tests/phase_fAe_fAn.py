from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

from tests.phase_fAe import test

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    conf = [15101.24769778409, 486209.27743067988, 252.71702367096787,
            363.2970828395908, 43.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]
    log = 0
    h2 = 0.825100624266976845
    g2 = 0

    # fAe_fAn has four possible ends
    func = ODEThreeCompHydSimulator.fAe_fAn
    logging.info("Func {}".format(func))

    p = 600
    logging.info("h = 1")
    n_func = test(func, h2, g2, p, conf, log=log)
    assert n_func is None

    p = 0
    logging.info("h = 1 - gamma")
    n_func = test(func, h2, g2, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe_lAn

    p = 0
    conf = [15101.24769778409, 486209.27743067988, 252.71702367096787,
            363.2970828395908, 43.27073086773415, 0.14892228099402588,
            0.5524379644134216, 0.3580228306857272]
    logging.info("h = 1 - phi")
    n_func = test(func, h2, g2, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.lAe_fAn

    p = 200
    conf = [151010.24769778409, 486209.27743067988, 252.71702367096787,
            363.2970828395908, 43.27073086773415, 0.14892228099402588,
            0.5524379644134216, 0.3580228306857272]
    logging.info("time limit")
    n_func = test(func, h2, g2, p, conf, log=log, t_max=10)
    assert n_func is None
