import logging
import itertools

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# published combined cp33 and cp66 measures with std dev
caen_combination = {
    "P4": [[120, 240, 360], [51.8, 57.7, 64.0], [2.8, 4.3, 5.8]],
    "P8": [[120, 240, 360], [40.1, 44.8, 54.8], [3.9, 3.0, 3.8]]
}


def multiple_exhaustion_comparison_overview(w_p: float, cp: float, ps: list):
    """
    Plots the expenditure energy dynamics of multiple three component hydraulic model configurations
    in comparison to the CP model.
    :param w_p: ground truth W' parameter
    :param cp: ground truth CP parameter
    :param ps: a list of three component hydraulic model configurations
    """

    hyd_color = "tab:green"
    two_p_color = "tab:blue"

    # fig sizes to make optimal use of space in paper
    fig = plt.figure(figsize=(8, 3.4))
    ax = fig.add_subplot(1, 1, 1)

    resolution = 1

    ts_ext = np.arange(120, 1801, 20 / resolution)
    ts = [120, 240, 360, 600, 900, 1800]
    powers = [((w_p + x * cp) / x) for x in ts]
    powers_ext = [((w_p + x * cp) / x) for x in ts_ext]

    # mark P4 and P8
    ax.get_xaxis().set_ticks(ts)
    ax.set_xticklabels([int(p / 60) for p in ts])
    ax.get_yaxis().set_ticks([])
    ax.set_yticklabels([])

    # small zoomed-in detail window
    insert_ax = ax.inset_axes([0.3, 0.40, 0.3, 0.45])
    detail_obs = resolution * 5
    detail_ts = [120, 150, 180, 210]
    detail_ps = []

    # plot three comp agents
    for p in ps:
        three_comp_agent = ThreeCompHydAgent(hz=1, a_anf=p[0], a_ans=p[1], m_ae=p[2], m_ans=p[3], m_anf=p[4],
                                             the=p[5], gam=p[6], phi=p[7])

        hyd_fitted_times_ext = [ThreeCompHydSimulator.simulate_tte_hydraulic_detail(three_comp_agent, x) for x in
                                powers_ext]
        hyd_powers_ext = powers_ext
        ax.plot(hyd_fitted_times_ext, hyd_powers_ext,
                linestyle='-', linewidth=1, color=hyd_color)

        insert_ax.plot(hyd_fitted_times_ext[:detail_obs], hyd_powers_ext[:detail_obs],
                       linestyle='-', linewidth=1, color=hyd_color)

    # plot CP curve
    insert_ax.plot(ts_ext[:detail_obs], powers_ext[:detail_obs],
                   linestyle='-', linewidth=2, label="critical power\nmodel", color=two_p_color)
    ax.plot(ts_ext, powers_ext,
            linestyle='-', linewidth=2, label="critical power\nmodel", color=two_p_color)

    # detailed view
    formatted = []
    for p in detail_ts:
        val = round((p / 60), 1)
        if val % 1 == 0:
            formatted.append(int(val))
        else:
            formatted.append(val)

    insert_ax.get_xaxis().set_ticks(detail_ts)
    insert_ax.set_xticklabels(formatted)
    insert_ax.get_yaxis().set_ticks(detail_ps)
    insert_ax.set_title("detail view")

    # label axis and lines
    ax.set_xlabel("time to exhaustion (min)")
    ax.set_ylabel("intensity (watt)", labelpad=10)

    # insert number of models only if more than 1 was plotted
    if len(ps) > 1:
        ax.plot([], linestyle='-', linewidth=1, color=hyd_color, label="hydraulic model ({})".format(len(ps)))
    else:
        ax.plot([], linestyle='-', linewidth=1, color=hyd_color, label="hydraulic model")
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.91)
    plt.show()
    plt.close(fig)


def multiple_caen_recovery_overview(w_p: float, cp: float, ps: list):
    """
    plots the energy recovery dynamics comparison of multiple three comonent hydraulic model configurations
    in comparison to observations by Caen et al.
    :param w_p: ground truth W'
    :param cp: ground truth CP
    :param ps: three component hydraulic model configurations
    """

    # caen observation and hydraulic model colors
    c_color = "tab:blue"
    hyd_color = "tab:green"

    # power level and recovery level estimations for the trials
    p_4 = (w_p + 240 * cp) / 240
    p_8 = (w_p + 480 * cp) / 480
    cp_33 = cp * 0.33
    cp_66 = cp * 0.66

    # create all the comparison trials
    exp_ps = [p_4, p_8]
    rec_ps = [cp_33, cp_66]
    rec_ts = [10, 20, 25, 30, 35, 40, 45, 50, 60, 70, 90, 110, 130, 150, 170, 240, 300, 360]

    fig = plt.figure(figsize=(8, 3.4))
    axes = []

    # using combined CP33 and CP66 measures
    # only two plots with combined recovery intensities
    axes.append(fig.add_subplot(1, 2, 1))
    axes.append(fig.add_subplot(1, 2, 2))

    # add three component model data
    for p in ps:
        # set up agent according to parameters set
        three_comp_agent = ThreeCompHydAgent(hz=1, a_anf=p[0], a_ans=p[1], m_ae=p[2], m_ans=p[3], m_anf=p[4],
                                             the=p[5], gam=p[6], phi=p[7])

        # data will be stored in here
        three_comp_data = []
        # let the created agend do all the trial combinations
        combs = list(itertools.product(exp_ps, rec_ps))
        for comb in combs:
            rec_times, rec_percent = [0], [0]
            for rt in rec_ts:
                ratio = ThreeCompHydSimulator.get_recovery_ratio_caen(three_comp_agent, comb[0], comb[1], rt)
                rec_times.append(rt)
                rec_percent.append(ratio)
            three_comp_data.append([rec_times, rec_percent])

        # sort into p4s and p8s
        p4s, p8s = [], []
        for i, comb in enumerate(combs):
            if comb[0] == exp_ps[0]:
                p4s.append(three_comp_data[i][1])
            else:
                p8s.append(three_comp_data[i][1])

        # get the means
        p4s = [(p4s[0][x] + p4s[1][x]) / 2 for x in range(len(p4s[0]))]
        p8s = [(p8s[0][x] + p8s[1][x]) / 2 for x in range(len(p8s[0]))]
        # plot into both available axes
        axes[0].plot(three_comp_data[0][0], p4s, linestyle='-', linewidth=1, color=hyd_color)
        axes[1].plot(three_comp_data[0][0], p8s, linestyle='-', linewidth=1, color=hyd_color)

    # combined data reported by Caen
    axes[0].errorbar(caen_combination["P4"][0],
                     caen_combination["P4"][1],
                     caen_combination["P4"][2],
                     label="Caen et al.",
                     linestyle='None',
                     marker='o',
                     capsize=3,
                     color=c_color)
    axes[0].set_title("P4")
    axes[1].errorbar(caen_combination["P8"][0],
                     caen_combination["P8"][1],
                     caen_combination["P8"][2],
                     label="Caen et al.",
                     linestyle='None',
                     marker='o',
                     capsize=3,
                     color=c_color)
    axes[1].set_title("P8")

    # insert number of models only if more than 1 was plotted
    if len(ps) > 1:
        axes[0].plot([], linestyle='-', linewidth=1, color=hyd_color, label="hydraulic model ({})".format(len(ps)))
    else:
        axes[0].plot([], linestyle='-', linewidth=1, color=hyd_color, label="hydraulic model")

    # format axis
    for ax in axes:
        ax.set_ylim(0, 70)
        ax.set_xlim(0, 370)
        ax.set_xticks([0, 120, 240, 360])
        ax.set_xticklabels([0, 2, 4, 6])
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.legend(loc='lower right')

    fig.text(0.5, 0.04, 'recovery duration (min)', ha='center')
    fig.text(0.01, 0.5, 'recovery (%)', va='center', rotation='vertical')
    plt.tight_layout()
    plt.subplots_adjust(left=0.09, bottom=0.20, top=0.91)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # general settings
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")
    rcParams['font.size'] = 12
    hz = 1

    # CP agent with group averages of Caen et al.
    w_p = 18200
    cp = 248

    # all configurations for three component hydraulic agents
    ps = [
        [
            18317.985677917935,
            129179.32465932379,
            247.90440593859648,
            85.99543495873674,
            8.8811555531768,
            0.774014008721324,
            0.13413758517048566,
            0.21316945230174328
        ],
        [
            18299.37774893322,
            159976.90023664298,
            247.83314832283853,
            84.19912288603444,
            9.237491794683404,
            0.7775401496189448,
            0.14789249329066642,
            0.21013146876025446
        ],
        [
            18555.355855624162,
            54081.718964449254,
            247.53332782046704,
            101.32108972440675,
            8.763644979167898,
            0.7352004659009965,
            0.03111645156674795,
            0.21584163543038992
        ],
        [
            18234.945230777033,
            136475.2736334105,
            248.0911312364717,
            85.73231293830037,
            9.170881107694296,
            0.7838076997298662,
            0.12852675715086678,
            0.19843515240195564
        ],
        [
            19080.218048988325,
            81659.2634694489,
            248.12390918259803,
            92.29525240175494,
            8.62536742352512,
            0.7813864868668488,
            0.0729463970340806,
            0.18935962845394638
        ],
        [
            18764.594226565343,
            59987.36294028655,
            247.37996960379456,
            101.79369713236133,
            9.058853131025726,
            0.7579688942793361,
            0.02524816298198033,
            0.20322143189936465
        ],
        [
            18786.84981302384,
            55707.6905771989,
            247.75277223898217,
            100.11527884033734,
            8.5408367027065,
            0.7442894711235575,
            0.0317825675687539,
            0.20688322369028503
        ],
        [
            18566.255853995925,
            80089.97565428763,
            247.75116000503044,
            92.32175626562861,
            8.501684410385646,
            0.7667333274594474,
            0.08337398127957903,
            0.2123027809945262
        ],
        [
            19310.69888836859,
            52643.91673959909,
            247.49449002246607,
            101.99068981659494,
            7.5984058727841965,
            0.7454843699999475,
            0.01411595969589113,
            0.20658572181747628
        ],
        [
            17498.130747953455,
            61501.40144904987,
            249.10139384341682,
            90.46391727854098,
            9.049762035456272,
            0.7021456802477695,
            0.10922107177115482,
            0.22049652492010913
        ]
    ]

    logging.info("Start Exhaustion Comparison")
    multiple_exhaustion_comparison_overview(w_p=w_p, cp=cp, ps=ps)
    logging.info("Start Recovery Comparison")
    multiple_caen_recovery_overview(w_p=w_p, cp=cp, ps=ps)
