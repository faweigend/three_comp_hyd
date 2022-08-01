import itertools
import logging

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.simulator.ode_three_comp_hyd_simulator import ODEThreeCompHydSimulator
from threecomphyd.simulator.three_comp_hyd_simulator import ThreeCompHydSimulator
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

# All configurations with phi = 1.0
a = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.14892228099402588,  # m_anf, theta
     0.3524379644134216, 1.0]  # gamma, phi

b = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.0,  # m_anf, theta
     0.3524379644134216, 1.0]  # gamma, phi

c = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.5,  # m_anf, theta
     0.0, 1.0]  # gamma, phi

d = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.0,  # m_anf, theta
     0.0, 1.0]  # gamma, phi

# All configurations with phi = 0
e = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.3,  # m_anf, theta
     0.3, 0.0]  # gamma, phi

f = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.0,  # m_anf, theta
     0.3, 0.0]  # gamma, phi

g = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.6,  # m_anf, theta
     0.0, 0.0]  # gamma, phi

h = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.0,  # m_anf, theta
     0.0, 0.0]  # gamma, phi

# All configurations with phi <= gamma and 1>phi>0
i = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.0,  # m_anf, theta
     0.3524379644134216, 0.1580228306857272]  # gamma, phi

j = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.14892228099402588,  # m_anf, theta
     0.3524379644134216, 0.1580228306857272]  # gamma, phi

k = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.0,  # m_anf, theta
     0.1580228306857272, 0.1580228306857272]  # gamma, phi

l = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.14892228099402588,  # m_anf, theta
     0.2580228306857272, 0.2580228306857272]  # gamma, phi

# All configurations with phi > gamma and gamma < 1
m = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.0,  # m_anf, theta
     0.1580228306857272, 0.3580228306857272]  # gamma, phi

n = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     38.27073086773415, 0.14892228099402588,  # m_anf, theta
     0.1580228306857272, 0.3580228306857272]  # gamma, phi

o = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 1 - 0.6580228306857272,  # m_anf, theta
     0.1580228306857272, 0.6580228306857272]  # gamma, phi

p = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.64892228099402588,  # m_anf, theta
     0.1580228306857272, 0.6580228306857272]  # gamma, phi

# Configurations with gamma = 0
q = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.0,  # m_anf, theta
     0.0, 0.6580228306857272]  # gamma, phi

r = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.2,  # m_anf, theta
     0.0, 0.6580228306857272]  # gamma, phi

s = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.6,  # m_anf, theta
     0.0, 0.4]  # gamma, phi

t = [15101.24769778409, 86209.27743067988,  # anf, ans
     252.71702367096787, 363.2970828395908,  # m_ae, m_ans
     380.27073086773415, 0.6,  # m_anf, theta
     0.0, 0.6]  # gamma, phi

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    hz = 1000  # delta t
    eps = 0.1  # required precision
    t_max = 2000
    log_level = 0

    # setting combinations
    p_exps = [260, 681]
    rec_times = [10, 240, 3600]
    p_recs = [0, 247]
    configs = [a, b, c, d,
               e, f, g, h,
               m, n, o, p,
               q, r, s, t]

    combs = list(itertools.product(p_exps, rec_times, p_recs, configs))

    # iterate through all possible setting combinations
    for i, comb in enumerate(combs):
        logging.info("{}/{} Comb {}".format(i, len(combs), comb))

        # get settings from combination
        p_exp = comb[0]
        rec_time = comb[1]
        p_rec = comb[2]
        conf = comb[3]

        # create three component hydraulic agent with example configuration
        agent = ThreeCompHydAgent(hz=hz,
                                  lf=conf[0], ls=conf[1],
                                  m_u=conf[2], m_ls=conf[3],
                                  m_lf=conf[4], the=conf[5],
                                  gam=conf[6], phi=conf[7])

        # Start with first time to exhaustion bout
        tte, h_tte, g_tte = ODEThreeCompHydSimulator.constant_power_trial(p=p_exp,
                                                                          conf=conf,
                                                                          start_h=0,
                                                                          start_g=0,
                                                                          t_max=t_max)

        if log_level > 1:
            logging.info("TTE ODE {} with h {} and g {}".format(round(tte), h_tte, g_tte))

        # double-check time to exhaustion
        try:
            c_tte = ThreeCompHydSimulator.tte(agent=agent, p_work=p_exp, t_max=t_max)
        except UserWarning:
            c_tte = t_max
        assert abs(round(tte) - round(c_tte)) < eps, "TTE ODE {} and IT {} is off by {}".format(round(tte),
                                                                                                round(c_tte),
                                                                                                abs(tte - c_tte))

        # double-check h and g
        agent.reset()
        for _ in range(int(round(tte * hz))):
            agent.set_power(p_exp)
            agent.perform_one_step()
        g_diff = agent.get_g() - g_tte
        h_diff = agent.get_h() - h_tte
        assert abs(g_diff) < eps, "TTE g is off by {}".format(g_diff)
        assert abs(h_diff) < eps, "TTE h is off by {}".format(h_diff)

        # Recovery behaviour
        rec, h_rec, g_rec = ODEThreeCompHydSimulator.constant_power_trial(p=p_rec, conf=conf,
                                                                          start_h=h_tte, start_g=g_tte,
                                                                          t_max=rec_time)

        # double-check with discrete agent
        # agent.reset()
        agent.set_h(h_tte)
        agent.set_g(g_tte)
        for _ in range(int(round(rec_time * hz))):
            agent.set_power(p_rec)
            agent.perform_one_step()
        g_diff = agent.get_g() - g_rec
        h_diff = agent.get_h() - h_rec
        assert abs(g_diff) < eps, "REC g is off by {}. {} vs {}".format(g_diff, agent.get_g(), g_rec)
        assert abs(h_diff) < eps, "REC h is off by {}. {} vs {}".format(h_diff, agent.get_h(), h_rec)

        # Now a full recovery trial
        rec_t = ODEThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(conf=conf, p_exp=p_exp,
                                                                    p_rec=p_rec, t_rec=rec_time,
                                                                    t_max=t_max)
        try:
            c_rec_t = ThreeCompHydSimulator.get_recovery_ratio_wb1_wb2(agent=agent, p_work=p_exp,
                                                                       p_rec=p_rec, t_rec=rec_time,
                                                                       t_max=t_max)
        except UserWarning:
            c_rec_t = 200

        assert abs(rec_t - c_rec_t) < eps, "Rec trial ODE {} and IT {} are of by {}".format(rec_t, c_rec_t,
                                                                                            abs(rec_t - c_rec_t))

        logging.info("PASSED")
