import logging
import os

from threecomphyd.evolutionary_fitter.pygmo_three_comp_fitter import PyGMOThreeCompFitter
from threecomphyd.evolutionary_fitter.three_comp_tools import prepare_caen_recovery_ratios, \
    prepare_standard_tte_measures

import threecomphyd.config as config

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # set the path to store the results into
    config.paths["data_storage"] = os.path.dirname(os.path.abspath(__file__)) + "/../data-storage/"

    # define (W', CP) combination in use. Insert your parameters here.
    comb = (
        18200,  # W'
        248  # CP
    )

    # define exhaustion trials
    ttes = prepare_standard_tte_measures(w_p=comb[0], cp=comb[1])

    # load desired recovery ratios to fit to
    recovery_measures = prepare_caen_recovery_ratios(w_p=comb[0], cp=comb[1])

    # fit a three component model to the agent
    fitter = PyGMOThreeCompFitter(ttes=ttes,
                                  recovery_measures=recovery_measures)

    # run 10 fittings and save results to
    # config.paths["data_storage"] + "THREE_COMP_PYGMO_FIT" folder
    for _ in range(10):
        # sets parameters according to best fitting determined by grid search
        # see Table 3 in Appendix
        fitter.fit_with_moead(gen=30,
                              cycles=40,
                              pop_size=64,
                              islands=7)
