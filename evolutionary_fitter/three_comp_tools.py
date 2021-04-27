import math
import numpy as np

import three_comp_config
from agents.three_comp_hyd_agent import ThreeCompHydAgent
from data_structure.constant_effort_measures import ConstantEffortMeasures
from data_structure.recovery_measures import RecoveryMeasures
from simulate.three_comp_hyd_simulator import ThreeCompHydSimulator

# bounds for all parameters of the three comp hydraulic model
three_comp_parameter_limits = {
    "a_anf": [1000, 500000],
    "a_ans": [1000, 500000],
    "m_ae": [1, 5000],
    "m_ans": [1, 5000],
    "m_anf": [1, 5000],
    "theta": [0.01, 0.99],
    "gamma": [0.01, 0.99],
    "phi": [0.01, 0.99]
}


class MultiObjectiveThreeCompUDP:
    """
    A user defined multi-objective problem (UDP) to solve the three comp fitting with PyGMO
    """

    def __init__(self, ttes: ConstantEffortMeasures, recovery_measures: RecoveryMeasures):
        """
        init needs a couple parameters to be able to call the objective function
        :param ttes: standardised time to exhaustion measures to use
        :param recovery_measures: standardised recovery ratio measures to use
        """
        self.__hz = 1  # hz
        self.__limits = three_comp_parameter_limits  # limits
        self.__ttes = ttes  # tte_ps
        self.__recovery_measures = recovery_measures  # recovery_trials

        # transform limits dict into object that returns bounds in expected format
        self.__bounds = ([x[0] for x in self.__limits.values()],
                         [x[1] for x in self.__limits.values()])

    def create_educated_initial_guess(self, cp: float = 250.0, w_p: float = 200000.0):
        """
        creates a suitable initial guess configuration
        :param cp: critical power that usually corresponds to Ae flow
        :param w_p: W' that is correlated to capacities of tanks
        :return:
        """
        # set up an initial configuration with sensibly distributed values
        i_x = [
            np.random.normal(1, 0.4) * w_p * 0.3,  # size AnF is expected to be smaller than AnS
            np.random.normal(1, 0.4) * w_p,  # size AnS is expected to be larger and correlated to W'
            np.random.normal(1, 0.4) * cp,  # max flow from Ae should be related to CP
            np.random.normal(1, 0.4) * cp * 10,  # max flow from AnS is expected to be high
            np.random.normal(1, 0.4) * cp * 0.1,  # max recovery flow is expected to be low
            np.random.normal(1, 0.4) * 0.5,  # for a curvelinear expenditure dynamic the pipe has to be halfway or lower
            np.random.normal(1, 0.4) * 0.25,  # AnS needs a considerable height
            np.random.normal(1, 0.4) * 0.25,  # AnS needs a considerable height
        ]
        # ensure values are within limits
        for i, i_x_e in enumerate(i_x):
            # lower bound
            i_x[i] = max(self.__bounds[0][i], i_x_e)
            # upper bound
            i_x[i] = min(self.__bounds[1][i], i_x_e)
        return i_x

    def fitness(self, x):
        """
        multi-objective fitness function
        :param x: a configuration for the three comp model
        :return: [expenditure error, recovery error]
        """
        try:
            tte_nrmse, rec_nrmse = three_comp_two_objective_functions(obj_vars=x,
                                                                      hz=self.__hz,
                                                                      ttes=self.__ttes,
                                                                      recovery_measures=self.__recovery_measures)

        # UserWarnings indicate that exhaustion was not reached
        # or recovery was too long
        # or violation of tank size constraint on ANS
        except UserWarning:
            tte_nrmse, rec_nrmse = 100, 100

        # error measures as fitness and constraint
        return [tte_nrmse, rec_nrmse]

    def get_bounds(self):
        """
        bounds for config parameters
        :return: bounds in tuple format
        """
        return self.__bounds

    def get_additional_info(self):
        """
        some additional info regarding the parameters in use
        :return:
        """
        return {"phi_constraint": three_comp_config.three_comp_phi_constraint,
                "ttes": self.__ttes.as_dict(),
                "recs": self.__recovery_measures.name}

    def get_name(self):
        """
        simple name getter for PyGMO interface
        :return: name string
        """
        # adds the phi constraint information to the name
        return "MultiObjectiveThreeComp_{}".format(three_comp_config.three_comp_phi_constraint)

    def get_nobj(self):
        """
        :return: number of objectives
        """
        return 2


def three_comp_two_objective_functions(obj_vars, hz: int,
                                       ttes: ConstantEffortMeasures,
                                       recovery_measures: RecoveryMeasures):
    """
    Two objective functions for recovery and expenditure error
    that get all required params as arguments
    :param obj_vars: values that define the three comp agent [anf, ans, m_ae, m_anf, m_ans, theta, gamma, phi]
    :param hz: estimations per second for agent
    :param ttes: time to exhaustion tests to use
    :param recovery_measures: recovery trials to compare to
    :return: tte_nrmse and rec_nrmse values to minimise (the smaller the better)
    """

    # differences in exhaustion times determine fitness
    tte_se = []  # TTE standard errors
    ttes_exp = []  # TTEs that are expected (original)
    rec_se = []  # Recovery standard errors
    recs_exp = []  # Recovery ratios expected (original)

    three_comp_agent = ThreeCompHydAgent(hz=hz,
                                         a_anf=obj_vars[0],
                                         a_ans=obj_vars[1],
                                         m_ae=obj_vars[2],
                                         m_ans=obj_vars[3],
                                         m_anf=obj_vars[4],
                                         the=obj_vars[5],
                                         gam=obj_vars[6],
                                         phi=obj_vars[7])
    # compare tte times
    for tte_t, tte_p in ttes.iterate_pairs():
        # use the simulator
        tte = ThreeCompHydSimulator.do_a_tte(agent=three_comp_agent,
                                             p_exp=tte_p)
        # square time difference
        tte_se.append(pow(tte - tte_t, 2))
        ttes_exp.append(tte_t)

    # get NRMSE (Normalised Root Mean Squared Error)
    tte_nrmse = math.sqrt(sum(tte_se) / len(tte_se)) / np.mean(ttes_exp)

    # compare all available recovery ratio measures
    for p_exp, p_rec, t_rec, expected in recovery_measures.iterate_all_measures():
        # use the simulator
        achieved = ThreeCompHydSimulator.get_recovery_ratio_caen(three_comp_agent,
                                                                 p_exp=p_exp,
                                                                 p_rec=p_rec,
                                                                 t_rec=t_rec) * 0.01

        # add the squared difference
        rec_se.append(pow(expected - achieved, 2))
        recs_exp.append(expected)

    # get NRMSE
    rec_nrmse = math.sqrt(sum(rec_se) / len(rec_se)) / np.mean(recs_exp)

    # determine return value
    return tte_nrmse, rec_nrmse

def multi_to_single_objective(t, r):
    """
    transformation of multi-objective to single-objective
    the euclidean distance to the ideal state of 0,0
    :param t: tte_nrmse
    :param r: rec_nrmse
    """
    return math.sqrt(t ** 2 + r ** 2)

def three_comp_single_objective_function(obj_vars,
                                         hz,
                                         ttes: ConstantEffortMeasures,
                                         recovery_measures: RecoveryMeasures):
    """
    The function how it was used in the past
    :param obj_vars:
    :param hz:
    :param ttes:
    :param recovery_measures:
    :return: distance to ideal state of 0,0 and optional debug parameters
    """
    t_nrmse, r_nrmse = three_comp_two_objective_functions(obj_vars=obj_vars,
                                                          hz=hz,
                                                          ttes=ttes,
                                                          recovery_measures=recovery_measures)
    return multi_to_single_objective(t_nrmse, r_nrmse)


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
