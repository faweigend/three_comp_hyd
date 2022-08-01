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

    conf = [10101.24769778409, 80209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]

    log = 2
    ht3 = 0.2
    gt3 = 0

    # lAe_lAn has four possible ends
    func = ODEThreeCompHydSimulator.lAe_lAn
    logging.info("Func {}".format(func))

    p = 350
    logging.info("h = 1 - phi")
    n_func = test(func, ht3, gt3, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe_lAn

    p = 0
    ht3 = 0.5
    logging.info("h = g")
    n_func = test(func, ht3, gt3, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.lAe_rAn

    p = 150
    ht3 = 0.5
    logging.info("time limit")
    n_func = test(func, ht3, gt3, p, conf, log=log, t_max=1000)
    assert n_func is None

    p = 450
    conf = [10101.24769778409, 80209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.1580228306857272]
    logging.info("h = 1 - gamma")
    n_func = test(func, ht3, gt3, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.lAe_fAn
