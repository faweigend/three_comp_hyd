import logging

from threecomphyd.evolutionary_fitter.pygmo_three_comp_fitter import PyGMOThreeCompFitter
from threecomphyd.evolutionary_fitter.three_comp_tools import prepare_caen_recovery_ratios, prepare_standard_tte_measures

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # group averages from Caen et al. 2019
    caen = (
        18200, # W'
        248 # CP
    )

    # define wp cp combination in use
    comb = caen

    # define exhaustion trials
    ttes = prepare_standard_tte_measures(w_p=comb[0], cp=comb[1])

    # load desired recovery ratios to fit to
    recovery_measures = prepare_caen_recovery_ratios(w_p=comb[0], cp=comb[1])

    # fit a three component model to the agent
    fitter = PyGMOThreeCompFitter(ttes=ttes, recovery_measures=recovery_measures)

    # Grid search starts here
    islands = [7, 14, 21]
    for isl in islands:
        fitter.grid_search_algorithm_moead(islands=isl)
