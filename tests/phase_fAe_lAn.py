import math

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from scipy import optimize

import logging

import numpy as np
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

from tests import configurations
from tests.phase_fAe import test

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    conf = configurations.c

    log = 2
    h5 = 1.0 - conf[7]
    g5 = 0.1

    # fAe_lAn has four possible ends
    func = ODEThreeCompHydSimulator.fAe_lAn
    logging.info("Func {}".format(func))

    p = 500
    logging.info("h = 1 - gamma")
    n_func = test(func, h5, g5, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe_fAn

    p = 100
    h5 = 1.0 - conf[6]
    logging.info("h = 1 - phi")
    n_func = test(func, h5, g5, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.lAe_lAn

    p = 100
    h5 = 1.0 - conf[6]
    g5 = 0.6
    logging.info("h = g")
    n_func = test(func, h5, g5, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe_rAn

    p = 255
    h5 = 1.0 - conf[6]
    g5 = 0.6
    logging.info("time limit")
    n_func = test(func, h5, g5, p, conf, log=log)
    assert n_func is None
