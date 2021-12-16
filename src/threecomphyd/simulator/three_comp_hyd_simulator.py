import logging
import numpy as np

from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
import matplotlib.pyplot as plt


class ThreeCompHydSimulator:
    """
    Employs the ThreeComponentHydraulic model to simulate training sessions and
    TTE tests
    """

    step_limit = 5000

    @staticmethod
    def do_a_tte(agent: ThreeCompHydAgent, p_exp, step_function=None):
        """
        a normal time to exhaustion test
        :param agent:
        :param p_exp:
        :param step_function:
        :return:
        """
        agent.reset()

        if step_function is None:
            step_function = agent.perform_one_step

        # WB1 Exhaust...
        agent.set_power(p_exp)
        steps = 0
        while not agent.is_exhausted() and steps < ThreeCompHydSimulator.step_limit:
            step_function()
            steps += 1
        wb1_t = agent.get_time()

        if not agent.is_exhausted():
            raise UserWarning("exhaustion not reached!")

        return wb1_t

    @staticmethod
    def get_recovery_ratio_wb1_wb2(agent: ThreeCompHydAgent, p_exp, p_rec, t_rec):
        """
        Returns recovery ratio of given agent according to WB1 -> RB -> WB2 protocol.
        Recovery ratio estimations for given exp, rec intensity and time
        :param agent: three component hydraulic agent to use
        :param p_exp: work bout intensity
        :param p_rec: recovery bout intensity
        :param t_rec: recovery bout duration
        :return: ratio in percent
        """

        hz = agent.hz
        agent.reset()

        # WB1 Exhaust...
        agent.set_power(p_exp)
        steps = 0
        while not agent.is_exhausted() and steps < ThreeCompHydSimulator.step_limit:
            agent.perform_one_step()
            steps += 1
        wb1_t = agent.get_time()

        if not agent.is_exhausted():
            raise UserWarning("exhaustion not reached!")

        # Recover...
        agent.set_power(p_rec)
        for _ in range(0, int(t_rec * hz)):
            agent.perform_one_step()
        rec_t = agent.get_time()

        # WB2 Exhaust...
        agent.set_power(p_exp)
        steps = 0
        while not agent.is_exhausted() and steps < ThreeCompHydSimulator.step_limit:
            agent.perform_one_step()
            steps += 1
        wb2_t = agent.get_time()
        # return ratio of times as recovery ratio
        return ((wb2_t - rec_t) / wb1_t) * 100.0

    @staticmethod
    def simulate_course_detail(agent: ThreeCompHydAgent, powers: list, plot: bool = False):
        """
        simulates a whole course with given agent
        :param agent:
        :param powers:
        :param plot:
        :return all parameter values throughout the simulation
        """

        agent.reset()
        h, g, anf, ans, p_ae, p_an, m_flow, w_p_bal = [], [], [], [], [], [], [], []

        # let the agent simulate the list of power demands
        for step in powers:
            # log all the parameters
            h.append(agent.get_h())
            g.append(agent.get_g())
            anf.append(agent.get_fill_lf())
            ans.append(agent.get_fill_ls())
            p_ae.append(agent.get_p_u())
            p_an.append(agent.get_p_l())
            m_flow.append(agent.get_m_flow())
            w_p_bal.append(agent.get_w_p_ratio())

            # perform current power step
            agent.set_power(step)
            agent.perform_one_step()

        # plot results
        if plot is True:
            ThreeCompHydSimulator.plot_dynamics(t=np.arange(len(powers)),
                                                p=powers,
                                                anf=anf,
                                                ans=ans,
                                                p_ae=p_ae,
                                                p_an=p_an)

        # return parameters
        return h, g, anf, ans, p_ae, p_an, m_flow, w_p_bal

    @staticmethod
    def simulate_tte_hydraulic_detail(agent: ThreeCompHydAgent, power, plot=False):
        """
        returns the time the agent takes till exhaustion at given power
        """

        agent.reset()
        agent.set_power(power)

        t, p, anf, ans, p_ae, p_an, m_flow = [], [], [], [], [], [], []
        # perform steps until agent is exhausted
        steps = 0
        while agent.is_exhausted() is False and steps < 3000:
            t.append(agent.get_time())
            p.append(agent.perform_one_step())
            anf.append(agent.get_fill_lf())
            ans.append(agent.get_fill_ls())
            p_ae.append(agent.get_p_u())
            p_an.append(agent.get_p_l())
            m_flow.append(agent.get_m_flow())
            steps += 1

        if plot is True:
            ThreeCompHydSimulator.plot_dynamics(t, p, anf, ans, p_ae, p_an)

        return agent.get_time()

    @staticmethod
    def simulate_tte_with_recovery(agent: ThreeCompHydAgent, exp_p, rec_p, plot=False):
        """
        The time the agent takes till exhaustion at given power and time till recovery
        :param agent: agent instance to use
        :param exp_p: expenditure intensity
        :param rec_p: recovery intensity
        :param plot: plot parameter time series
        :returns: tte, ttr
        """

        agent.reset()
        t, p, anf, ans, p_h, p_g, m_flow = [], [], [], [], [], [], []

        # perform steps until agent is exhausted
        logging.info("start exhaustion")
        agent.set_power(exp_p)
        steps = 0
        while agent.is_exhausted() is False and steps < 10000:
            t.append(agent.get_time())
            p.append(agent.perform_one_step())
            anf.append(agent.get_fill_lf())
            ans.append(agent.get_fill_ls() * agent.height_ls + agent.theta)
            p_h.append(agent.get_p_u())
            p_g.append(agent.get_p_l())
            m_flow.append(agent.get_m_flow())
            steps += 1
        # save time
        tte = agent.get_time()

        # add recovery at 0
        logging.info("start recovery")
        agent.set_power(rec_p)
        steps = 0
        while agent.is_equilibrium() is False and steps < 20000:
            t.append(agent.get_time())
            p.append(agent.perform_one_step())
            anf.append(agent.get_fill_lf())
            ans.append(agent.get_fill_ls() * agent.height_ls + agent.theta)
            p_h.append(agent.get_p_u())
            p_g.append(agent.get_p_l())
            m_flow.append(agent.get_m_flow())
            steps += 1
        # save recovery time
        ttr = agent.get_time() - tte

        # plot the parameter overview if required
        if plot is True:
            ThreeCompHydSimulator.plot_dynamics(t, p, anf, ans, p_h, p_g)

        # return time till exhaustion and time till recovery
        return tte, ttr

    @staticmethod
    def plot_dynamics(t, p, anf, ans, p_ae, p_an):
        """
        Debugging plots to look at developed power curves
        """

        # set up plot
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)

        # plot liquid flows
        ax.plot(t, p, color='tab:blue', label="power")
        ax.plot(t, p_ae, color='tab:red', label="flow from Ae")
        ax.plot(t, p_an, color='tab:purple', label="flow from AnS to AnF")

        # plot tank fill levels
        ax2 = ax.twinx()
        ax2.plot(t, anf, color='tab:green', label="fill level AnF", linestyle="--")
        ax2.plot(t, ans, color='tab:orange', label="fill level AnS", linestyle="--")

        # label plot
        ax.set_xlabel("time (s)")
        ax.set_ylabel("flow and power in Watts")
        ax2.set_ylabel("fill level")

        # legends
        ax.legend(loc=1)
        ax2.legend(loc=4)

        # formant plot
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=-45)
        plt.tight_layout()
        plt.show()
