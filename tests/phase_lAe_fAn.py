import logging
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator

from tests.phase_fAe import test

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    conf = [15101.24769778409, 86209.27743067988, 252.71702367096787,
            363.2970828395908, 38.27073086773415, 0.14892228099402588,
            0.4524379644134216, 0.2580228306857272]

    ht4 = 0.6475620355865783
    gt4 = 0.15679831105786776

    log = 0

    # lAe_fAn has three possible ends
    func = ODEThreeCompHydSimulator.lAe_fAn
    logging.info("Func {}".format(func))

    p = 350
    logging.info("h = 1 - phi")
    n_func = test(func, ht4, gt4, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.fAe_fAn

    p = 100
    logging.info("h = 1 - gamma")
    n_func = test(func, ht4, gt4, p, conf, log=log)
    assert n_func == ODEThreeCompHydSimulator.lAe_lAn

    p = 230
    gt4 = 0.35
    logging.info("time limit")
    n_func = test(func, ht4, gt4, p, conf, log=log, t_max=1000)
    assert n_func is None
