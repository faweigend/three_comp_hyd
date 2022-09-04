import datetime
import logging
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import mpl_toolkits.axes_grid1

from threecomphyd import config
from pypermod.agents.hyd_agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.animator.three_comp_extended_animation import ThreeCompExtendedAnimation
from threecomphyd.evolutionary_fitter.three_comp_tools import three_comp_parameter_limits, \
    three_comp_single_objective_function
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation

from pypermod.simulator.simulator_basis import SimulatorBasis
from pypermod.utility import PlotLayout
from pypermod.agents.wbal_agents.wbal_ode_agent_skiba import WbalODEAgentSkiba


class ThreeCompFitnessInvestigator:
    """
    An interactive visualisation to investigate the fitness function and how it evaluates behaviour of a
    three component hydraulic model configuration
    """

    def __init__(self, ttes, recovery_measures, load_config: list = None):
        """
        basic setup
        """

        # All parameters for fitness estimation
        # hz for agent simulations
        self.__hz = 1

        self.__recovery_trials = recovery_measures

        self.__load_config = load_config

        self.__ttes = ttes
        # parameter boundaries in use
        self.__limits = three_comp_parameter_limits
        # the objective function in use
        self.__fitness_function = three_comp_single_objective_function

        # visualisation parameters
        # set up matplotlib figure
        self.fig = plt.figure(figsize=(14, 8))
        ax1 = self.fig.add_subplot(2, 2, 1)

        # all the controls
        ax_anf = self.fig.add_subplot(2, 2, 2)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax_anf)
        ax_ans = divider.append_axes("bottom", size="80%", pad=0.05)
        ax_m_ae = divider.append_axes("bottom", size="80%", pad=0.05)
        ax_m_ans = divider.append_axes("bottom", size="80%", pad=0.05)
        ax_m_anf = divider.append_axes("bottom", size="80%", pad=0.05)
        ax_theta = divider.append_axes("bottom", size="80%", pad=0.05)
        ax_gamma = divider.append_axes("bottom", size="80%", pad=0.05)
        ax_phi = divider.append_axes("bottom", size="80%", pad=0.05)
        ax_rand = divider.append_axes("bottom", size="100%", pad=0.03)
        ax_reset = divider.append_axes("bottom", size="100%", pad=0.03)
        ax_estimate = divider.append_axes("bottom", size="100%", pad=0.03)
        ax_results = divider.append_axes("bottom", size="100%", pad=0.03)
        ax_animations = divider.append_axes("bottom", size="100%", pad=0.03)
        ax_load = divider.append_axes("bottom", size="100%", pad=0.03)

        # expenditure evaluation
        self.ax_expenditure_overview = self.fig.add_subplot(2, 2, 3)

        # recovery evaluation
        self.rec_grid = mpl_toolkits.axes_grid1.ImageGrid(self.fig, 224,  # similar to subplot(111)
                                                          nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                                                          axes_pad=0.6,  # pad between axes in inch.
                                                          aspect=False)

        # set up three comp agend and visualisation
        agent = ThreeCompHydAgent(1, lf=1000, ls=1000, m_u=10, m_ls=10, m_lf=10, the=0.1, gam=0.1,
                                  phi=0.1)
        self.vis = ThreeCompVisualisation(agent=agent, axis=ax1)
        # self.vis.init_layout()

        # slider
        self.slider_anf = Slider(ax_anf, 'LF', 1000, 500000, valstep=1000)
        self.slider_ans = Slider(ax_ans, 'LS', 1000, 500000, valstep=1000)
        self.slider_phi = Slider(ax_phi, 'phi', 0, 0.99, valstep=0.01)
        self.slider_theta = Slider(ax_theta, 'theta', 0, 0.99, valstep=0.01)
        self.slider_gamma = Slider(ax_gamma, 'gamma', 0, 0.99, valstep=0.01)
        self.slider_m_ae = Slider(ax_m_ae, 'M_U', 1, 5000, valstep=1)
        self.slider_m_ans = Slider(ax_m_ans, 'M_LS', 1, 5000, valstep=1)
        self.slider_m_anf = Slider(ax_m_anf, 'M_LF', 1, 5000, valstep=1)

        # buttons
        self.random_button = Button(ax_rand, 'Random', hovercolor='0.975')
        self.random_button.on_clicked(self.new_random)
        self.reset_button = Button(ax_reset, 'Reset', hovercolor='0.975')
        self.reset_button.on_clicked(self.reset)
        self.estimate_button = Button(ax_estimate, 'Estimate', hovercolor='0.975')
        self.estimate_button.on_clicked(self.estimate_fitness)
        self.__est_num = 0
        self.animations_button = Button(ax_animations, 'Create Animations', hovercolor='0.975')
        #self.animations_button.on_clicked(self.create_animations)
        self.load_button = Button(ax_load, 'load', hovercolor='0.975')
        self.load_button.on_clicked(self.load_values)

        # info
        ax_results.set_axis_off()
        self.text_results = ax_results.text(0, 0, "fitness: {}      run: {}".format(0, 0))

        self.cids = []
        # assign update function to all sliders
        for slider in [self.slider_anf, self.slider_ans,
                       self.slider_phi, self.slider_theta,
                       self.slider_gamma, self.slider_m_ae,
                       self.slider_m_ans, self.slider_m_anf]:
            self.cids.append(slider.on_changed(self.update_vis))

        # update slider visualisation and make plot ready
        self.set_random_positions()
        self.update_vis(0)

        plt.show()

    def load_values(self, event):
        """
        assigns vals to slider values
        """
        vals = self.__load_config

        anf = vals[0]
        ans = vals[1]
        m_ae = vals[2]
        m_ans = vals[3]
        m_anf = vals[4]
        theta = vals[5]
        gamma = vals[6]
        phi = vals[7]

        self.assign_slider_values(anf=anf, ans=ans, m_ae=m_ae, m_anf=m_anf, m_ans=m_ans,
                                  phi=phi, theta=theta, gamma=gamma)

        # just to check if constraint is violated
        agent = ThreeCompHydAgent(1, lf=self.slider_anf.val, ls=self.slider_ans.val,
                                  m_u=self.slider_m_ae.val, m_ls=self.slider_m_ans.val,
                                  m_lf=self.slider_m_anf.val, the=self.slider_theta.val,
                                  gam=self.slider_gamma.val, phi=self.slider_phi.val)

        self.update_vis(0)

    def assign_slider_values(self, anf, ans, m_ae, m_anf, m_ans, phi, theta, gamma):
        """
        sets slider values and defaults according to input
        :param anf:
        :param ans:
        :param phi:
        :param theta:
        :param gamma:
        :param m_ae:
        :param m_anf:
        :param m_ans:
        :return:
        """

        # deactivate all sliders
        for i, slider in enumerate([self.slider_anf, self.slider_ans,
                                    self.slider_phi, self.slider_theta,
                                    self.slider_gamma, self.slider_m_ae,
                                    self.slider_m_ans, self.slider_m_anf]):
            cid = self.cids[i]
            slider.disconnect(cid)

        self.slider_anf.set_val(anf)
        self.slider_ans.set_val(ans)
        self.slider_phi.set_val(phi)
        self.slider_theta.set_val(theta)
        self.slider_gamma.set_val(gamma)
        self.slider_m_ae.set_val(m_ae)
        self.slider_m_anf.set_val(m_anf)
        self.slider_m_ans.set_val(m_ans)

        # set slider defaults
        self.slider_anf.valinit = anf
        self.slider_ans.valinit = ans
        self.slider_phi.valinit = phi
        self.slider_theta.valinit = theta
        self.slider_gamma.valinit = gamma
        self.slider_m_ae.valinit = m_ae
        self.slider_m_anf.valinit = m_anf
        self.slider_m_ans.valinit = m_ans

        # update slider default vlines
        self.slider_anf.vline.set_xdata(anf)
        self.slider_ans.vline.set_xdata(ans)
        self.slider_phi.vline.set_xdata(phi)
        self.slider_theta.vline.set_xdata(theta)
        self.slider_gamma.vline.set_xdata(gamma)
        self.slider_m_ae.vline.set_xdata(m_ae)
        self.slider_m_anf.vline.set_xdata(m_anf)
        self.slider_m_ans.vline.set_xdata(m_ans)

        # activate all sliders again
        self.cids = []
        # assign update function to all sliders
        for slider in [self.slider_anf, self.slider_ans,
                       self.slider_phi, self.slider_theta,
                       self.slider_gamma, self.slider_m_ae,
                       self.slider_m_ans, self.slider_m_anf]:
            self.cids.append(slider.on_changed(self.update_vis))

    def set_random_positions(self):
        """
        random agent creation that ensures AnS has a positive height
        :return: agent
        """
        # sizes
        anf = int(random.uniform(self.slider_anf.valmin, self.slider_anf.valmax))  # AnF
        ans = int(random.uniform(self.slider_ans.valmin, self.slider_ans.valmax))  # AnS

        # maximal flows
        m_ae = random.uniform(self.slider_m_ae.valmin, self.slider_m_ae.valmax)
        m_ans = random.uniform(self.slider_m_ans.valmin, self.slider_m_ans.valmax)
        m_anf = random.uniform(self.slider_m_anf.valmin, self.slider_m_anf.valmax)

        # positions
        theta = random.uniform(self.slider_theta.valmin, self.slider_theta.valmax)

        # make sure parameter conditions are met according to set constraints
        if config.three_comp_phi_constraint is True:
            phi = random.uniform(self.slider_phi.valmin, 0.99 - theta)
            gamma = random.uniform(phi, 0.99 - theta)
        else:
            phi = random.uniform(self.slider_phi.valmin, self.slider_phi.valmax)
            gamma = random.uniform(self.slider_gamma.valmin, 0.99 - theta)

        # assign set values
        self.assign_slider_values(anf=anf, ans=ans, m_ae=m_ae, m_anf=m_anf, m_ans=m_ans,
                                  phi=phi, theta=theta, gamma=gamma)

    def update_vis(self, val):
        """
        update visualisation according to slider values
        :param val:
        """
        self.slider_gamma.valmax = 0.99 - self.slider_theta.val
        self.slider_gamma.ax.set_xlim(self.slider_gamma.valmin, self.slider_gamma.valmax)

        if config.three_comp_phi_constraint is True:
            # adjust the phi slider to not violate three comp constraints
            self.slider_phi.valmax = self.slider_gamma.val
            self.slider_phi.ax.set_xlim(self.slider_phi.valmin, self.slider_gamma.val)
            if self.slider_phi.val > self.slider_phi.valmax:
                self.slider_phi.set_val(self.slider_phi.valmax)

        self.slider_theta.valmax = 0.99 - self.slider_gamma.val
        self.slider_theta.ax.set_xlim(self.slider_theta.valmin, self.slider_theta.valmax)

        agent = ThreeCompHydAgent(1, lf=self.slider_anf.val, ls=self.slider_ans.val,
                                  m_u=self.slider_m_ae.val, m_ls=self.slider_m_ans.val,
                                  m_lf=self.slider_m_anf.val, the=self.slider_theta.val,
                                  gam=self.slider_gamma.val, phi=self.slider_phi.val)

        self.vis.update_basic_layout(agent)
        self.fig.canvas.draw_idle()

    def new_random(self, event):
        """
        create a random agend and update visualisation
        """
        self.set_random_positions()
        self.update_vis(0)

    def reset(self, event):
        """
        simple reset function for all sliders
        :param event:
        """
        # reset all sliders
        for slider in [self.slider_anf, self.slider_ans,
                       self.slider_phi, self.slider_theta,
                       self.slider_gamma, self.slider_m_ae,
                       self.slider_m_ans, self.slider_m_anf]:
            slider.reset()
        self.update_vis(0)
    #
    # def create_animations(self, event):
    #     """
    #     Creates animations for all fitness trials
    #     :param event:
    #     :return:
    #     """
    #     data = DataParser.load_caen_w_p_cp()
    #     hz = 1
    #     step_limit = int(3000 * hz)
    #
    #     # create the agents
    #     w_p = data['1']['w_p']
    #     cp = data['1']['cp']
    #     skiba_agent = WbalODEAgentSkiba(w_p=w_p, cp=cp, hz=hz)
    #     three_comp_agent = ThreeCompHydAgent(hz, lf=self.slider_anf.val, ls=self.slider_ans.val,
    #                                          m_u=self.slider_m_ae.val, m_ls=self.slider_m_ans.val,
    #                                          m_lf=self.slider_m_anf.val, the=self.slider_theta.val,
    #                                          gam=self.slider_gamma.val, phi=self.slider_phi.val)
    #
    #     # exhaustion trials
    #     ttes = np.arange(120, 1801, 360)
    #     # corresponding power
    #     tte_ps = [(skiba_agent.w_p + x * skiba_agent.cp) / x for x in ttes]
    #
    #     # recovery trials
    #     recovery_measures = DataParser.prepare_caen_recovery_ratios(w_p=w_p, cp=cp)
    #
    #     # compare tte times
    #     for j, tte_p in enumerate(tte_ps):
    #         logging.info("simulate TTE {} for {}".format(tte_p, three_comp_agent.get_name()))
    #         # first run a simple simulation to get the time
    #         three_comp_agent.reset()
    #         three_comp_agent.set_power(tte_p)
    #         steps = 0
    #         # perform exercise at p_work with eventual time limit t_rec
    #         while three_comp_agent.is_exhausted() is False and steps < step_limit:
    #             three_comp_agent.perform_one_step()
    #             steps += 1
    #         tte = three_comp_agent.get_time()
    #
    #         # now create the animation with estimated time
    #         three_comp_agent.reset()
    #         ani = ThreeCompExtendedAnimation(three_comp_agent, hz=10, controls=False, frames=int(tte))
    #         logging.info("create animation TTE {} for {}".format(tte_p, three_comp_agent.get_name()))
    #         ani.set_run_commands([0, tte], [tte_p, 0])
    #         name = "{}_tte_{}_{}.mp4".format(tte_p,
    #                                          three_comp_agent.get_name(),
    #                                          datetime.datetime.now())
    #         ani.save_to_file(name)
    #
    #     # compare recovery ratios
    #     for p_exp, p_rec, t_rec, expected in recovery_measures.iterate_measures():
    #
    #         # Perform WB1...
    #         three_comp_agent.reset()
    #         three_comp_agent.set_power(p_exp)
    #         steps = 0
    #         # perform exercise at p_work with eventual time limit t_exp
    #         while three_comp_agent.is_exhausted() is False and steps < step_limit:
    #             three_comp_agent.perform_one_step()
    #             steps += 1
    #         # record time result
    #         hyd_wb1_t = three_comp_agent.get_time()
    #
    #         if steps >= step_limit:
    #             # punish exceeded step count with a 200% recovery
    #             achieved = 2
    #         else:
    #             # Recover...
    #             three_comp_agent.set_power(p_rec)
    #             for _ in range(0, int(t_rec * hz)):
    #                 three_comp_agent.perform_one_step()
    #             rec_t = three_comp_agent.get_time()
    #
    #             # WB2 perform p_work until exhaustion...
    #             three_comp_agent.set_power(p_exp)
    #             steps = 0
    #             while three_comp_agent.is_exhausted() is False and steps < step_limit:
    #                 three_comp_agent.perform_one_step()
    #                 steps += 1
    #             hyd_wb2_t = three_comp_agent.get_time() - rec_t
    #
    #             # now create the animation with estimated time
    #             tte = hyd_wb1_t + int(t_rec * hz) + hyd_wb2_t
    #             three_comp_agent.reset()
    #             ani = ThreeCompExtendedAnimation(three_comp_agent, hz=10, controls=False, frames=int(tte))
    #             logging.info("create animation recovery trial {}-{}-{} for {}".format(p_exp, p_rec, t_rec,
    #                                                                                   three_comp_agent.get_name()))
    #             ani.set_run_commands([0, hyd_wb1_t, hyd_wb1_t + int(t_rec * hz), tte], [p_exp, p_rec, p_exp, 0])
    #             name = "{}-{}-{}_trial_{}_{}.mp4".format(p_exp, p_rec, t_rec, three_comp_agent.get_name(),
    #                                                      datetime.datetime.now())
    #             ani.save_to_file(name)
    #
    #     logging.info("animations done")

    def estimate_fitness(self, event):
        """
        :param event:
        """

        obj_vars = [self.slider_anf.val,
                    self.slider_ans.val,
                    self.slider_m_ae.val,
                    self.slider_m_ans.val,
                    self.slider_m_anf.val,
                    self.slider_theta.val,
                    self.slider_gamma.val,
                    self.slider_phi.val]

        # determine fitness
        fitness = self.__fitness_function(
            obj_vars=obj_vars,
            hz=self.__hz,
            ttes=self.__ttes,
            recovery_measures=self.__recovery_trials)

        self.__est_num += 1
        self.text_results.set_text("fitness: {}      run: {}".format(round(fitness, 5), self.__est_num))

        # create agent for simulations
        agent = ThreeCompHydAgent(hz=1,
                                  lf=obj_vars[0],
                                  ls=obj_vars[1],
                                  m_u=obj_vars[2],
                                  m_ls=obj_vars[3],
                                  m_lf=obj_vars[4],
                                  the=obj_vars[5],
                                  gam=obj_vars[6],
                                  phi=obj_vars[7])

        # plot ground truth for expenditure
        self.ax_expenditure_overview.clear()
        self.ax_expenditure_overview.plot(self.__ttes.times,
                                          self.__ttes.powers,
                                          label="two parameter model",
                                          color=PlotLayout.get_plot_color("ground_truth"))

        # get tte predictions from agent
        ttes = []
        for p in self.__ttes.powers:
            try:
                ttes.append(len(SimulatorBasis.get_tte_dynamics(agent=agent, p_work=p)))
            except UserWarning:
                ttes.append(SimulatorBasis.step_limit)
        self.ax_expenditure_overview.plot(ttes,
                                          self.__ttes.powers,
                                          linestyle='-',
                                          label="hydraulic model",
                                          color=PlotLayout.get_plot_color("ThreeCompHydAgent"))

        # label axis and lines
        self.ax_expenditure_overview.set_xlabel("time to exhaustion (s)")
        self.ax_expenditure_overview.set_ylabel("power output (W)")
        self.ax_expenditure_overview.legend()
        self.ax_expenditure_overview.set_title(
            "expenditure " + r'${} \downarrow {}$' + "                 recovery" + r'${} \rightarrow {}$')
        # self.ax_expenditure_overview.set_xlim(0, self.__ttes.times[-1])

        # now the recovery part
        all_combs = list(self.__recovery_trials.get_all_wb_rb_combinations())
        max_t = self.__recovery_trials.get_max_t_rec() + 1

        # one axis per p_work p_rec combination
        for i, ax in enumerate(self.rec_grid):
            ax.clear()

            # if less combs are available than axis to plot on
            if i >= len(all_combs):
                continue

            # get all ground truth values
            p_exp, p_rec = all_combs[i]
            times, obs = self.__recovery_trials.get_all_obs_for_wb_rb_combination(p_work=p_exp, p_rec=p_rec)
            ax.scatter(times, obs, color=PlotLayout.get_plot_color("ground_truth"))

            # simulate and plot agent behaviour
            rts = np.arange(0, max_t, 30)
            recs = []
            for rt in rts:
                try:
                    ratio = SimulatorBasis.get_recovery_ratio_wb1_wb2(agent, p_work=p_exp, p_rec=p_rec, t_rec=rt)
                    recs.append(ratio)
                except UserWarning:
                    recs.append(0)
            ax.plot(rts, recs, color=PlotLayout.get_plot_color(agent.get_name()))

            # add labels to axis
            ax.set_title(r'${} \rightarrow {}$'.format(p_exp, p_rec))
            ax.set_ylabel("recovery (%)")
            ax.set_xlabel("recovery duration (s)")
            ax.set_ylim((-5, 105))
