import math

import numpy as np
from threecomphyd import config
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.data_structure.simple_rec_measures import SimpleRecMeasures
from threecomphyd.data_structure.simple_tte_measures import SimpleTTEMeasures
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator

# bounds for all parameters of the three comp hydraulic model
three_comp_parameter_limits = {
    "a_anf": [5000, 500000],
    "a_ans": [5000, 500000],
    "m_ae": [1, 2000],
    "m_ans": [1, 2000],
    "m_anf": [1, 2000],
    "theta": [0.01, 0.99],  # 0.0 and 1.0 are not possible because equations would divide by 0
    "gamma": [0.01, 0.99],
    "phi": [0.01, 0.99]
}


class MultiObjectiveThreeCompUDP:
    """
    A user defined multi-objective problem (UDP) to solve the three comp fitting with PyGMO
    """

    def __init__(self,
                 ttes: SimpleTTEMeasures,
                 recovery_measures: SimpleRecMeasures):
        """
        init needs a couple parameters to be able to call the objective function
        :param ttes: standardised time to exhaustion measures to use
        :param recovery_measures: standardised recovery ratio measures to use
        """

        self.__hz = 1  # hz
        self.__limits = three_comp_parameter_limits  # limits
        self.__ttes = ttes  # tte_ps
        self.__recs = recovery_measures  # recovery_trials

        # transform limits dict into object that returns bounds in expected format
        self.__bounds = ([x[0] for x in self.__limits.values()],
                         [x[1] for x in self.__limits.values()])

    def create_educated_initial_guess(self, cp: float = 250.0, w_p: float = 50000.0):
        """
        creates a suitable initial guess configuration
        :param cp: critical power that usually corresponds to Ae flow
        :param w_p: W' that is correlated to capacities of tanks
        :return: initial configuration guess as a list
        """
        # set up an initial configuration with sensibly distributed values
        i_x = [
            np.random.normal(1, 0.4) * w_p * 0.3,  # size AnF is expected to be smaller than AnS
            np.random.normal(1, 0.4) * w_p,  # size AnS is expected to be larger and correlated to W'
            np.random.normal(1, 0.4) * cp,  # max flow from Ae should be related to CP
            np.random.normal(1, 0.4) * cp,  # max flow from AnS
            np.random.normal(1, 0.4) * cp * 0.1,  # max recovery flow is expected to be low
            np.random.normal(1, 0.4) * 0.25,  # theta: top of AnS
            np.random.normal(1, 0.4) * 0.25,  # gamma: 1 - bottom of AnS
            np.random.normal(1, 0.4) * 0.5,  # phi: for a curvelinear expenditure the pipe should be halfway or lower
        ]

        # make sure AnS has a positive cross-sectional area
        while i_x[5] + i_x[6] > 0.99:
            i_x[5] = np.random.normal(1, 0.4) * 0.25  # theta
            i_x[6] = np.random.normal(1, 0.4) * 0.25  # gamma

        # ensure values are within limits
        for i in range(len(i_x)):
            i_x[i] = max(self.__bounds[0][i], i_x[i])  # lower bound
            i_x[i] = min(self.__bounds[1][i], i_x[i])  # upper bound
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
                                                                      recovery_measures=self.__recs)

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
        :return: additional info in a dictionary
        """
        return {
            "phi_constraint": config.three_comp_phi_constraint,
            "ttes": str(self.__ttes),
            "recs": str(self.__recs),
        }

    def get_name(self):
        """
        simple name getter for PyGMO interface
        :return: name string
        """
        # adds the phi constraint information to the name
        return "MultiObjectiveThreeComp_{}".format(config.three_comp_phi_constraint)

    def get_nobj(self):
        """
        :return: number of objectives
        """
        return 2


def three_comp_two_objective_functions(obj_vars, hz: int,
                                       ttes: SimpleTTEMeasures,
                                       recovery_measures: SimpleRecMeasures):
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
        try:
            tte = ThreeCompHydSimulator.tte(agent=three_comp_agent,
                                            p_work=tte_p)
        except UserWarning:
            tte = 5000
        # square time difference
        tte_se.append(pow(tte - tte_t, 2))
        ttes_exp.append(tte_t)

    # get NRMSE (Normalised Root Mean Squared Error)
    tte_nrmse = math.sqrt(sum(tte_se) / len(tte_se)) / np.mean(ttes_exp)

    # compare all available recovery ratio measures
    for p_exp, p_rec, t_rec, expected in recovery_measures.iterate_measures():
        # use the simulator
        try:
            achieved = ThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(three_comp_agent,
                                                                        p_exp=p_exp,
                                                                        p_rec=p_rec,
                                                                        t_rec=t_rec)
        except UserWarning:
            achieved = 200
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
                                         ttes: SimpleTTEMeasures,
                                         recovery_measures: SimpleRecMeasures):
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


def prepare_standard_tte_measures(w_p: float, cp: float):
    """
    creates TTE measures as a ConstantEffortMeasures object
    :param w_p: W' measure
    :param cp: CP measure
    :return SimpleTTEMeasures Object
    """
    # different TTE settings to check which one makes the model recreate the hyperbolic curve the best
    tte_ts = [120, 130, 140, 150, 170, 190, 210, 250, 310, 400, 600, 1200]  # setting_0
    # name according to current convention
    ttes = SimpleTTEMeasures("{}_{}_setting_0".format(w_p, cp))
    for t in tte_ts:
        # create pairs with two param CP fomula
        ttes.add_pair(t, round((w_p + t * cp) / t, 2))
    # return simple TTEs object
    return ttes


def prepare_caen_recovery_ratios(w_p: float, cp: float):
    """
    returns recovery ratio data according to published data by Caen et al. 2019
    :param w_p: W'
    :param cp: CP
    :return SimpleRecMeasures Object
    """
    # estimate intensities
    p4 = round(cp + w_p / 240, 2)  # predicted exhaustion after 4 min
    p8 = round(cp + w_p / 480, 2)  # predicted exhaustion after 8 min
    cp33 = round(cp * 0.33, 2)
    cp66 = round(cp * 0.66, 2)
    # sub, test, p_work, p_rec, r_time, recovery_percent
    caen_data = [[p4, cp33, 120, 55.0],
                 [p4, cp33, 240, 61.0],
                 [p4, cp33, 360, 70.5],
                 [p4, cp66, 120, 49.0],
                 [p4, cp66, 240, 55.0],
                 [p4, cp66, 360, 58.0],
                 [p8, cp33, 120, 42.0],
                 [p8, cp33, 240, 52.0],
                 [p8, cp33, 360, 59.5],
                 [p8, cp66, 120, 38.0],
                 [p8, cp66, 240, 37.5],
                 [p8, cp66, 360, 50.0]]
    # name indicates used measures
    recs = SimpleRecMeasures("caen")
    for p_work, p_rec, t_rec, r_percent in caen_data:
        recs.add_measure(p_work=p_work, p_rec=p_rec, r_time=t_rec, recovery_percent=r_percent)
    # return simple recs object
    return recs
