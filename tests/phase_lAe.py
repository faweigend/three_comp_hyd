import logging
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

from tests.phase_fAe import test

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    conf = [15101.24769778409, 86209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.3524379644134216, 0.4580228306857272]

    h = 0.1

    # lAe has four possible ends
    func = ODEThreeCompHydSimulator.lAe
    logging.info("Func {}".format(func))

    p = 0
    logging.info("h = 0")
    n_func = test(func, h, 0, p, conf, t_max=500)
    assert n_func is None

    p = 350
    logging.info("h = 1- gamma")
    n_func = test(func, h, 0, p, conf)
    assert n_func == ODEThreeCompHydSimulator.lAe_lAn

    p = 350
    conf = [15101.24769778409, 86209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.44892228099402588,
            0.3524379644134216, 0.8580228306857272]
    logging.info("h = 1 - phi")
    n_func = test(func, h, 0, p, conf)
    assert n_func == ODEThreeCompHydSimulator.fAe

    p = 10
    logging.info("equilibrium")
    n_func = test(func, h, 0, p, conf)
    assert n_func is None
