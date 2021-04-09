import logging

from data_structure.constant_effort_measures import ConstantEffortMeasures
from data_structure.recovery_measures import RecoveryMeasures
from evolutionary_fitter.pygmo_three_comp_fitter import PyGMOThreeCompFitter


def prepare_tte_measures(w_p, cp):
    """
    creates TTE measures as a ConstantEffortMeasures object
    """
    # different TTE settings to check which one makes the model recreate the hyperbolic curve the best
    tte_t_setting = [120, 130, 140, 150, 170, 190, 210, 250, 310, 400, 600, 1200]  # setting_0
    tte_ts = tte_t_setting
    tte_ps = [(w_p + x * cp) / x for x in tte_ts]
    return ConstantEffortMeasures(times=tte_ts, measures=tte_ps,
                                  name="{}_{}_setting_0".format(w_p, cp))


def prepare_caen_recovery_ratios(w_p: float, cp: float):
    """
    creates recovery ratio data according to published data by Caen et al.
    https://insights.ovid.com/crossref?an=00005768-201908000-00022
    """

    # originally read from a csv. For demo purposes we moved the content into this script
    # sub, test, wb_power, r_power, r_time, r_percent
    caen_data = [['p4', 'cp33', 120, 55.0],
                 ['p4', 'cp33', 240, 61.0],
                 ['p4', 'cp33', 360, 70.5],
                 ['p4', 'cp66', 120, 49.0],
                 ['p4', 'cp66', 240, 55.0],
                 ['p4', 'cp66', 360, 58.0],
                 ['p8', 'cp33', 120, 42.0],
                 ['p8', 'cp33', 240, 52.0],
                 ['p8', 'cp33', 360, 59.5],
                 ['p8', 'cp66', 120, 38.0],
                 ['p8', 'cp66', 240, 37.5],
                 ['p8', 'cp66', 360, 50.0]]

    # fills available test data for listed subjects into here
    rms = RecoveryMeasures("caen")

    # estimate intensities
    p4 = round(cp + w_p / 240, 2)  # predicted exhaustion after 4 min
    p8 = round(cp + w_p / 480, 2)  # predicted exhaustion after 8 min
    cp33 = round(cp * 0.33, 2)
    cp66 = round(cp * 0.66, 2)

    # read recovery ratios
    for data_row in caen_data:
        # get intensities and times from labels
        p_power = p4 if 'p4' in data_row[0] else p8
        r_power = cp33 if 'cp33' in data_row[1] else cp66

        # create new test in conventional format
        rms.add_measure(p_power=p_power,
                        r_power=r_power,
                        r_time=int(data_row[2]),
                        recovery_percent=float(data_row[3]))
    return rms


if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # group averages from Caen et al. https://insights.ovid.com/crossref?an=00005768-201908000-00022
    caen = (18200, 248)

    # define wp cp combination in use
    comb = caen

    # define exhaustion trials
    ttes = prepare_tte_measures(w_p=comb[0], cp=comb[1])

    # load desired recovery ratios to fit to
    recovery_measures = prepare_caen_recovery_ratios(w_p=comb[0], cp=comb[1])

    # fit a three component model to the agent
    fitter = PyGMOThreeCompFitter(ttes=ttes, recovery_measures=recovery_measures)
    fitter.early_stopping = 1

    # Grid search starts here
    islands = [2]
    for isl in islands:
        fitter.grid_search_algorithm_moead(islands=isl)
