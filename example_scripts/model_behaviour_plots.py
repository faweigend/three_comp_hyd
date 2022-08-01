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

    max_err = 0
    max_err_t = 0

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

        limit = 100
        if max(np.abs(np.array(ts_ext) - np.array(hyd_fitted_times_ext))[:limit]) > max_err:
            max_err = np.max(np.abs(np.array(ts_ext) - np.array(hyd_fitted_times_ext))[:limit])
            max_err_t = ts_ext[np.argmax(np.abs(np.array(ts_ext) - np.array(hyd_fitted_times_ext))[:limit])]

    print(max_err, max_err_t)

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
        [17456.501540851958, 46235.99782887891, 246.8754194575765, 105.0458949192332, 18.61813360834505,
         0.693542652379417, 0.01997874313126515, 0.2600493216687265],
        [17571.44900414276, 43402.54960764281, 246.97054379746947, 108.14610069719515, 17.1821984762747,
         0.6912752767454273, 0.010027987890364522, 0.2645509984088022],
        [18052.873627292814, 50179.25213290453, 247.47018172193998, 105.62464445309857, 16.23985284036382,
         0.7290616564998857, 0.028183446547478494, 0.25082072506146275],
        [16683.592287141386, 44887.58560272133, 246.63457734640542, 112.68272987053999, 18.073084252052254,
         0.6862224549739296, 0.018586310009923508, 0.2941636955314809],
        [16852.46273505409, 42909.778724502234, 246.81384076206803, 108.9815702415051, 18.8647120699937,
         0.6769843873369275, 0.015245173850835667, 0.2787578120456768],
        [16606.045920840497, 39679.597398728074, 246.8007548653167, 112.23416682274151, 18.476718183869735,
         0.6555286114289854, 0.010386190778171801, 0.28920487069782524],
        [16898.924891950985, 41130.49577887294, 246.86360534861495, 109.21040516611954, 18.828103302561843,
         0.6647849659185002, 0.013486135401909773, 0.2784037171471362],
        [17531.522855693423, 45987.13199861467, 246.83527483837145, 110.74973247039583, 18.39464657469822,
         0.7056095369263342, 0.01198807017906641, 0.26591486702753386],
        [16312.389121504952, 38982.0514065314, 246.68018865065818, 111.63159367625984, 19.378294909372325,
         0.6445812471142565, 0.010438963567674072, 0.29581431611381087],
        [16891.005497924867, 45520.8495750172, 246.842373015698, 115.53307266702876, 17.367498696562347,
         0.7039380779511493, 0.010040967408569901, 0.28745204966698823]
    ]

    logging.info("Start Exhaustion Comparison")
    multiple_exhaustion_comparison_overview(w_p=w_p, cp=cp, ps=ps)
    logging.info("Start Recovery Comparison")
    multiple_caen_recovery_overview(w_p=w_p, cp=cp, ps=ps)
