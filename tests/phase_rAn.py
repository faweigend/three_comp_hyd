import logging
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

from tests.phase_fAe import test

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    conf = [15101.24769778409, 86209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 1.0]
    log = 2
    h = 0
    g = 0.3

    # lAe has four possible ends
    func = ODEThreeCompHydSimulator.rAn
    logging.info("Func {}".format(func))

    p = 0
    logging.info("g = 0")
    n_func = test(func, h, g, p, conf, t_max=5000, log=log)
    assert n_func is None
