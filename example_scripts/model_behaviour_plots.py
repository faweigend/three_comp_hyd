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
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(1, 1, 1)

    resolution = 1

    ts_ext = np.arange(120, 1801, 20 / resolution)
    ts = [120, 240, 360, 600, 900, 1800]
    powers_ext = [((w_p + x * cp) / x) for x in ts_ext]

    # mark P4 and P8
    ax.get_xaxis().set_ticks(ts)
    ax.set_xticklabels([int(p / 60) for p in ts])
    ax.get_yaxis().set_ticks([])
    ax.set_yticklabels([])

    # small zoomed-in detail window
    insert_ax = ax.inset_axes([0.18, 0.40, 0.3, 0.45])
    detail_obs = resolution * 5
    detail_ts = [120, 150, 180, 210]
    detail_ps = []

    # plot three comp agents
    for p in ps:
        three_comp_agent = ThreeCompHydAgent(hz=1, lf=p[0], ls=p[1], m_u=p[2], m_ls=p[3], m_lf=p[4],
                                             the=p[5], gam=p[6], phi=p[7])

        hyd_fitted_times_ext = [ThreeCompHydSimulator.tte(three_comp_agent, x) for x in
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
    ax.set_ylabel("power output (W)", labelpad=10)

    # insert number of models only if more than 1 was plotted
    if len(ps) > 1:
        ax.plot([], linestyle='-', linewidth=1, color=hyd_color,
                label="$\mathrm{hydraulic}_\mathrm{weig}$" + " ({})".format(len(ps)))
    else:
        ax.plot([], linestyle='-', linewidth=1, color=hyd_color, label="$\mathrm{hydraulic}_\mathrm{weig}$")
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(left=0.071, bottom=0.162, top=0.99, right=0.986)
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

    fig = plt.figure(figsize=(6, 3.4))
    # using combined CP33 and CP66 measures
    # only two plots with combined recovery intensities
    axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

    # add three component model data
    for p in ps:
        # set up agent according to parameters set
        three_comp_agent = ThreeCompHydAgent(hz=1, lf=p[0], ls=p[1], m_u=p[2], m_ls=p[3], m_lf=p[4],
                                             the=p[5], gam=p[6], phi=p[7])

        # data will be stored in here
        three_comp_data = []
        # let the created agent run all the trial combinations
        combs = list(itertools.product(exp_ps, rec_ps))
        for comb in combs:
            rec_times, rec_percent = [0], [0]
            for rt in rec_ts:
                ratio = ThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(three_comp_agent, comb[0], comb[1], rt)
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
                     label="Caen et al. 2019",
                     linestyle='None',
                     marker='o',
                     capsize=3,
                     color=c_color)
    axes[0].set_title("$P240$")
    axes[1].errorbar(caen_combination["P8"][0],
                     caen_combination["P8"][1],
                     caen_combination["P8"][2],
                     label="Caen et al. 2019",
                     linestyle='None',
                     marker='o',
                     capsize=3,
                     color=c_color)
    axes[1].set_title("$P480$")

    # insert number of models only if more than 1 was plotted
    if len(ps) > 1:
        axes[0].plot([], linestyle='-', linewidth=1, color=hyd_color,
                     label="$\mathrm{hydraulic}_\mathrm{weig}$" + " ({})".format(len(ps)))
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
    plt.subplots_adjust(left=0.12, bottom=0.20, top=0.91, right=0.99)

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
            19267.32481889428,
            53448.210634894436,
            247.71129903795293,
            105.16070293047284,
            15.666725105148853,
            0.7594031586210723,
            0.01174178885462004,
            0.21384965007797813
        ],
        [
            18042.056916563655,
            46718.177938027344,
            247.39628102450715,
            106.77042879166977,
            16.96027055119397,
            0.715113715965181,
            0.01777338005017555,
            0.24959503053279475
        ],
        [
            16663.27479753794,
            58847.492279822916,
            247.33216892510987,
            97.81632336744637,
            17.60675026641434,
            0.6982927653365388,
            0.09473837917127066,
            0.28511732156368097
        ],
        [
            18122.869259409636,
            67893.3143320925,
            248.05526835319083,
            90.11814418883185,
            16.70767452536087,
            0.7239627763005275,
            0.10788624159437807,
            0.24269812950697436
        ],
        [
            16442.047054013587,
            43977.34142455164,
            246.6580206034908,
            112.31940101973737,
            18.851235626075855,
            0.6825405103462707,
            0.011882463932316418,
            0.2859061602516494
        ],
        [
            18241.03024888177,
            52858.78758906142,
            247.23306533702817,
            103.15393367087151,
            16.753404619019577,
            0.7323264858183177,
            0.03138505655373579,
            0.2448593661774296
        ],
        [
            16851.79305167275,
            48348.71227226837,
            246.55295343504088,
            106.85431134985403,
            19.123861764063058,
            0.693498746836151,
            0.03083991609890696,
            0.28415132656652087
        ],
        [
            16350.59391699999,
            40391.18583934175,
            246.40648185859834,
            111.34355485406216,
            19.583788050143703,
            0.6498660573226038,
            0.01016029963401485,
            0.30008300685218803
        ],
        [
            16748.777297458924,
            41432.324234179905,
            246.68605949562686,
            108.72756892117448,
            19.34245943802004,
            0.6596009015402553,
            0.013654881063378194,
            0.2814663041365496
        ],
        [
            17687.258359719104,
            51859.254197641196,
            247.39818147041177,
            103.37418807409084,
            17.566017275093582,
            0.7251325338084225,
            0.03563558893277309,
            0.24824817650787334
        ]
    ]

    logging.info("Start Exhaustion Comparison")
    multiple_exhaustion_comparison_overview(w_p=w_p, cp=cp, ps=ps)
    logging.info("Start Recovery Comparison")
    multiple_caen_recovery_overview(w_p=w_p, cp=cp, ps=ps)
