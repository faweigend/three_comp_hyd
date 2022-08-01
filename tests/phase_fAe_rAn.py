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

    conf = [5604.966588001499, 54499.44673416602, 155.82060702780947,
            105.26777135234472, 28.623478621476917, 0.2314852496176266,
            0.15438323467786853, 0.5949904604992432]

    log = 2
    ht6 = 0.45
    gt6 = 0.5

    # fAe_rAn has four possible ends
    func = ODEThreeCompHydSimulator.fAe_rAn
    logging.info("Func {}".format(func))

    p = 0
    logging.info("h = 1 - phi")
    n_func = test(func, ht6, gt6, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.lAe_rAn

    p = 500
    logging.info("h = g")
    n_func = test(func, ht6, gt6, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe_lAn

    p = 120
    gt6 = 0.1
    conf = [5604.966588001499, 14499.44673416602, 155.82060702780947,
            105.26777135234472, 128.623478621476917, 0.6314852496176266,
            0.15438323467786853, 0.5949904604992432]
    logging.info("g = 0")
    n_func = test(func, ht6, gt6, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe
