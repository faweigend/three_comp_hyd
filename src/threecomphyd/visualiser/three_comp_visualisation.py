import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent

from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams


class ThreeCompVisualisation:
    """
    Basis to visualise power flow within the hydraulics model as an animation or simulation
    """

    def __init__(self, agent: ThreeCompHydAgent,
                 axis: plt.axis = None,
                 animated: bool = False,
                 detail_annotations: bool = False,
                 basic_annotations: bool = True,
                 black_and_white: bool = False):
        """
        Whole visualisation setup using given agent's parameters
        :param agent: The agent to be visualised
        :param axis: If set, the visualisation will be drawn using the provided axis object.
        Useful for animations or to display multiple models in one plot
        :param animated: If true the visualisation is set up to deal with frame updates.
        See animation script for more details.
        :param detail_annotations: If true, tank distances and sizes are annotated as well.
        :param basic_annotations: If true, U, LF, and LS are visible
        :param black_and_white: If true, the visualisation is in black and white
        """
        # matplotlib fontsize
        rcParams['font.size'] = 10

        # plot if no axis was assigned
        if axis is None:
            fig = plt.figure(figsize=(8, 5))
            self._ax1 = fig.add_subplot(1, 1, 1)
        else:
            fig = None
            self._ax1 = axis

        if black_and_white:
            self.__u_color = (0.7, 0.7, 0.7)
            self.__lf_color = (0.5, 0.5, 0.5)
            self.__ls_color = (0.3, 0.3, 0.3)
            self.__ann_color = (0, 0, 0)
            self.__p_color = (0.5, 0.5, 0.5)

        elif not black_and_white:
            self.__u_color = "tab:cyan"
            self.__lf_color = "tab:orange"
            self.__ls_color = "tab:red"
            self.__ann_color = "tab:blue"
            self.__p_color = "tab:green"

        # basic parameters for setup
        self._animated = animated
        self.__detail_annotations = detail_annotations
        self._agent = agent
        self.__offset = 0.2

        # U tank with three stripes
        self.__width_u = 0.3
        self._u = None
        self._u1 = None
        self._u2 = None
        self._r1 = None  # line marking flow from U to LF
        self._ann_u = None  # U annotation

        # LF tank
        self._lf = None
        self._h = None  # fill state
        self._ann_lf = None  # annotation

        # LS tank
        self._ls = None
        self._g = None
        self._ann_ls = None  # annotation LS
        self._r2 = None  # line marking flow from LS to LF

        # finish the basic layout
        self.__set_basic_layout()
        self.update_basic_layout(agent)

        # now the animation components
        if self._animated:
            # U flow
            self._arr_u_flow = None
            self._ann_u_flow = None

            # flow out of tap
            self._arr_power_flow = None
            self._ann_power_flow = None

            # LS flow (R2)
            self._arr_r2_l_pos = None
            self._arr_r2_flow = None
            self._ann_r2_flow = None

            # time information annotation
            self._ann_time = None

            self.__set_animation_layout()
            self._ax1.add_artist(self._ann_time)

        # basic annotations are U, LF, and LS
        if not basic_annotations:
            self.hide_basic_annotations()

        # add layout for detailed annotations
        # detail annotation add greek letters for distances and positions
        if self.__detail_annotations:
            self.__set_detailed_annotations_layout()
            self._ax1.set_xlim(0, 1.05)
            self._ax1.set_ylim(0, 1.2)
        else:
            self._ax1.set_xlim(0, 1.0)
            self._ax1.set_ylim(0, 1.2)

        if self.__detail_annotations and self._animated:
            raise UserWarning("Detailed annotations and animation cannot be combined")

        self._ax1.set_axis_off()

        # display plot if no axis object was assigned
        if fig is not None:
            plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
            plt.show()
            plt.close(fig)

    def __set_detailed_annotations_layout(self):
        """
        Adds components required for a detailed
        annotations view with denoted positions and distances
        """

        u_width = self.__width_u
        ls_left = self._ls.get_x()
        ls_width = self._ls.get_width()
        lf_left = self._lf.get_x()
        lf_width = self._lf.get_width()
        ls_height = self._ls.get_height()

        # some offset to the bottom
        offset = self.__offset
        phi_o = self._agent.phi + offset
        gamma_o = self._agent.gamma + offset

        rcParams['text.usetex'] = True

        self._ann_u_flow = Text(text="$M_{U}$", ha='right', fontsize="xx-large", x=u_width + 0.09,
                                y=phi_o - 0.08)
        ann_p_ae = Text(text="$p_{U}$", ha='right', fontsize="xx-large", x=u_width + 0.07,
                        y=phi_o + 0.03)
        self._arr_u_flow = FancyArrowPatch((u_width, phi_o),
                                           (u_width + 0.1, phi_o),
                                           arrowstyle='-|>',
                                           mutation_scale=30,
                                           lw=2,
                                           color=self.__u_color)
        self._ax1.annotate('$\phi$',
                           xy=(u_width / 2, phi_o),
                           xytext=(u_width / 2, (phi_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\phi$',
                           xy=(u_width / 2, offset),
                           xytext=(u_width / 2, (phi_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._ann_power_flow = Text(text="$p$", ha='center', fontsize="xx-large", x=self._ann_lf.get_position()[0],
                                    y=offset - 0.06)
        self._arr_power_flow = FancyArrowPatch((self._ann_lf.get_position()[0], offset - 0.078),
                                               (self._ann_lf.get_position()[0], 0.0),
                                               arrowstyle='-|>',
                                               mutation_scale=30,
                                               lw=2,
                                               color=self.__p_color)

        self._ax1.annotate('$h$',
                           xy=(self._ann_lf.get_position()[0] + 0.07, 1 + offset),
                           xytext=(self._ann_lf.get_position()[0] + 0.07, 1 + offset - 0.30),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$h$',
                           xy=(self._ann_lf.get_position()[0] + 0.07, 1 + offset - 0.55),
                           xytext=(self._ann_lf.get_position()[0] + 0.07, 1 + offset - 0.30),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._h.update(dict(xy=(lf_left, offset),
                            width=lf_width,
                            height=1 - 0.55,
                            color=self.__lf_color))

        self._ax1.annotate('$g$',
                           xy=(ls_left + ls_width + 0.02, ls_height + gamma_o),
                           xytext=(ls_left + ls_width + 0.02, ls_height * 0.61 + gamma_o),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$g$',
                           xy=(ls_left + ls_width + 0.02, ls_height * 0.3 + gamma_o),
                           xytext=(ls_left + ls_width + 0.02, ls_height * 0.61 + gamma_o),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._g.update(dict(xy=(ls_left, gamma_o),
                            width=ls_width,
                            height=ls_height * 0.3,
                            color=self.__ls_color))

        ann_p_an = Text(text="$p_{L}$", ha='left', usetex=True, fontsize="xx-large", x=ls_left - 0.06,
                        y=gamma_o + 0.11)
        ann_arr_flow = Text(text="$M_{LS}$", ha='left', usetex=True, fontsize="xx-large", x=ls_left - 0.09,
                            y=gamma_o + 0.03)
        self._ann_r2_flow = Text(text="$M_{LF}$", ha='left', usetex=True, fontsize="xx-large", x=ls_left - 0.04,
                                 y=gamma_o - 0.07)

        self._arr_r2_flow = FancyArrowPatch((ls_left - 0.1, gamma_o),
                                            (ls_left, gamma_o),
                                            arrowstyle='<|-|>',
                                            mutation_scale=30,
                                            lw=2,
                                            color=self.__ls_color)

        self._ax1.annotate('$\\theta$',
                           xy=(ls_left + ls_width / 2, 1 + offset),
                           xytext=(
                               ls_left + ls_width / 2,
                               1 - (1 - (ls_height + gamma_o - offset)) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\\theta$',
                           xy=(ls_left + ls_width / 2, ls_height + gamma_o),
                           xytext=(
                               ls_left + ls_width / 2,
                               1 - (1 - (ls_height + gamma_o - offset)) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._ax1.annotate('$\\gamma$',
                           xy=(ls_left + ls_width / 2, offset),
                           xytext=(ls_left + ls_width / 2, (gamma_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\\gamma$',
                           xy=(ls_left + ls_width / 2, gamma_o),
                           xytext=(ls_left + ls_width / 2, (gamma_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._ax1.annotate('$1$',
                           xy=(1.05, 0 + offset),
                           xytext=(1.05, 0.5 + offset),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$1$',
                           xy=(1.05, 1 + offset),
                           xytext=(1.05, 0.5 + offset),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.add_artist(ann_arr_flow)
        # self._ax1.add_artist(ann_p_an)
        # self._ax1.add_artist(ann_p_ae)
        self._ax1.axhline(offset, linestyle='--', color=self.__ann_color)
        self._ax1.axhline(1 + offset - 0.001, linestyle='--', color=self.__ann_color)
        self._ax1.add_artist(self._ann_power_flow)
        self._ax1.add_artist(self._arr_power_flow)
        self._ax1.add_artist(self._arr_u_flow)
        self._ax1.add_artist(self._ann_u_flow)
        self._ax1.add_artist(self._arr_r2_flow)
        self._ax1.add_artist(self._ann_r2_flow)

    def __set_animation_layout(self):
        """
        Adds layout components that are required for an animation
        """

        offset = self.__offset
        o_width = self.__width_u
        phi_o = self._agent.phi + offset
        gamma_o = self._agent.gamma + offset

        # U flow (R1)
        self._arr_u_flow = FancyArrowPatch((o_width, phi_o),
                                           (o_width + 0.1, phi_o),
                                           arrowstyle='simple',
                                           mutation_scale=0,
                                           ec='white',
                                           fc=self.__u_color)
        self._ann_u_flow = Text(text="flow: ", ha='right', fontsize="large", x=o_width, y=phi_o - 0.05)

        # Tap flow (Power)
        self._arr_power_flow = FancyArrowPatch((self._ann_lf.get_position()[0], offset - 0.05),
                                               (self._ann_lf.get_position()[0], 0.0),
                                               arrowstyle='simple',
                                               mutation_scale=0,
                                               ec='white',
                                               color=self.__p_color)
        self._ann_power_flow = Text(text="flow: ", ha='center', fontsize="large", x=self._ann_lf.get_position()[0],
                                    y=offset - 0.05)

        # LS flow (R2)
        self._arr_r2_l_pos = [(self._ls.get_x(), gamma_o),
                              (self._ls.get_x() - 0.1, gamma_o)]
        self._arr_r2_flow = FancyArrowPatch(self._arr_r2_l_pos[0],
                                            self._arr_r2_l_pos[1],
                                            arrowstyle='simple',
                                            mutation_scale=0,
                                            ec='white',
                                            color=self.__ls_color)
        self._ann_r2_flow = Text(text="flow: ", ha='left', fontsize="large", x=self._ls.get_x(),
                                 y=gamma_o - 0.05)

        # information annotation
        self._ann_time = Text(x=1, y=0.9 + offset, ha="right")

        self._ax1.add_artist(self._ann_power_flow)
        self._ax1.add_artist(self._arr_power_flow)
        self._ax1.add_artist(self._arr_u_flow)
        self._ax1.add_artist(self._ann_u_flow)
        self._ax1.add_artist(self._arr_r2_flow)
        self._ax1.add_artist(self._ann_r2_flow)

    def __set_basic_layout(self):
        """
        updates position estimations and layout
        """

        # get sizes from agent
        lf = self._agent.lf
        ls = self._agent.ls
        ls_height = self._agent.height_ls

        # u_left is 0
        u_width = self.__width_u

        # determine width with size ratio retained
        lf_left = u_width + 0.1
        lf_width = ((lf * ls_height) * (1 - lf_left - 0.1)) / ls
        ls_left = lf_left + lf_width + 0.1
        ls_width = 1 - ls_left

        # some offset to the bottom
        offset = self.__offset
        phi_o = self._agent.phi + offset
        gamma_o = self._agent.gamma + offset

        # S tank
        self._u = Rectangle((0.0, phi_o), 0.05, 1 - self._agent.phi, color=self.__u_color, alpha=0.3)
        self._u1 = Rectangle((0.05, phi_o), 0.05, 1 - self._agent.phi, color=self.__u_color, alpha=0.6)
        self._u2 = Rectangle((0.1, phi_o), u_width - 0.1, 1 - self._agent.phi, color=self.__u_color)
        self._r1 = Line2D([u_width, u_width + 0.1],
                          [phi_o, phi_o],
                          color=self.__u_color)
        self._ann_u = Text(text="$U$", ha='center', fontsize="xx-large",
                           x=u_width / 2,
                           y=((1 - self._agent.phi) / 2) + phi_o - 0.02)

        # LF vessel
        self._lf = Rectangle((lf_left, offset), lf_width, 1, fill=False, ec="black")
        self._h = Rectangle((lf_left, offset), lf_width, 1, color=self.__lf_color)
        self._ann_lf = Text(text="$LF$", ha='center', fontsize="xx-large",
                            x=lf_left + (lf_width / 2),
                            y=offset + 0.5 - 0.02)

        # LS vessel
        self._ls = Rectangle((ls_left, gamma_o), ls_width, ls_height, fill=False, ec="black")
        self._g = Rectangle((ls_left, gamma_o), ls_width, ls_height, color=self.__ls_color)
        self._r2 = Line2D([ls_left, ls_left - 0.1],
                          [gamma_o, gamma_o],
                          color=self.__ls_color)
        self._ann_ls = Text(text="$LS$", ha='center', fontsize="xx-large",
                            x=ls_left + (ls_width / 2),
                            y=gamma_o + (ls_height / 2) - 0.02)

        # the basic layout
        self._ax1.add_line(self._r1)
        self._ax1.add_line(self._r2)
        self._ax1.add_artist(self._u)
        self._ax1.add_artist(self._u1)
        self._ax1.add_artist(self._u2)
        self._ax1.add_artist(self._lf)
        self._ax1.add_artist(self._ls)
        self._ax1.add_artist(self._h)
        self._ax1.add_artist(self._g)
        self._ax1.add_artist(self._ann_u)
        self._ax1.add_artist(self._ann_lf)
        self._ax1.add_artist(self._ann_ls)

    def update_basic_layout(self, agent):
        """
        updates tank positions and sizes according to new agent
        :param agent: agent to be visualised
        """

        self._agent = agent

        # get sizes from agent
        lf = agent.lf
        ls = agent.ls
        ls_heigh = agent.height_ls

        # o_left is 0
        u_width = self.__width_u

        # determine width with size ratio retained
        lf_left = u_width + 0.1
        lf_width = ((lf * ls_heigh) * (1 - lf_left - 0.1)) / (ls + lf * ls_heigh)
        ls_left = lf_left + lf_width + 0.1
        ls_width = 1 - ls_left

        # some offset to the bottom
        offset = self.__offset
        phi_o = agent.phi + offset
        gamma_o = agent.gamma + offset

        # S tank
        self._u.set_bounds(0.0, phi_o, 0.05, 1 - self._agent.phi)
        self._u1.set_bounds(0.05, phi_o, 0.05, 1 - self._agent.phi)
        self._u2.set_bounds(0.1, phi_o, u_width - 0.1, 1 - self._agent.phi)
        self._r1.set_xdata([u_width, u_width + 0.1])
        self._r1.set_ydata([phi_o, phi_o])
        self._ann_u.set_position(xy=(u_width / 2, ((1 - self._agent.phi) / 2) + phi_o - 0.02))

        # LF vessel
        self._lf.set_bounds(lf_left, offset, lf_width, 1)
        self._h.set_bounds(lf_left, offset, lf_width, 1)
        self._ann_lf.set_position(xy=(lf_left + (lf_width / 2),
                                      offset + 0.5 - 0.02))

        # LS vessel
        self._ls.set_bounds(ls_left, gamma_o, ls_width, ls_heigh)
        self._g.set_bounds(ls_left, gamma_o, ls_width, ls_heigh)
        self._r2.set_xdata([ls_left, ls_left - 0.1])
        self._r2.set_ydata([gamma_o, gamma_o])
        self._ann_ls.set_position(xy=(ls_left + (ls_width / 2), gamma_o + (ls_heigh / 2) - 0.02))

        # update levels
        self._h.set_height(1 - self._agent.get_h())
        self._g.set_height(self._agent.height_ls - self._agent.get_g())

    def hide_basic_annotations(self):
        """
        Simply hides the S, LF, and LS text
        """
        self._ann_u.set_text("")
        self._ann_lf.set_text("")
        self._ann_ls.set_text("")

    def update_animation_data(self, frame_number):
        """
        For animations and simulations.
        The function to call at each frame.
        :param frame_number: frame number has to be taken because of parent class method
        :return: an iterable of artists
        """

        if not self._animated:
            raise UserWarning("Animation flag has to be enabled in order to use this function")

        # perform one step
        cur_time = self._agent.get_time()
        power = self._agent.perform_one_step()

        # draw some information
        self._ann_time.set_text("agent \n time: {}".format(int(cur_time)))

        # power arrow
        self._ann_power_flow.set_text("power: {}".format(round(power)))
        self._arr_power_flow.set_mutation_scale(math.log(power + 1) * 10)

        # oxygen arrow
        p_u = round(self._agent.get_p_u() * self._agent.hz, 1)
        max_str = "(MAX)" if p_u == self._agent.m_u else ""
        self._ann_u_flow.set_text("flow: {} {}".format(p_u, max_str))
        self._arr_u_flow.set_mutation_scale(math.log(p_u + 1) * 10)

        # lactate arrow
        p_g = round(self._agent.get_p_l() * self._agent.hz, 1)
        if p_g < 0:
            max_str = "(MAX)" if p_g == self._agent.m_lf else ""
            self._arr_r2_flow.set_positions(self._arr_r2_l_pos[1], self._arr_r2_l_pos[0])
        else:
            max_str = "(MAX)" if p_g == self._agent.m_ls else ""
            self._arr_r2_flow.set_positions(self._arr_r2_l_pos[0], self._arr_r2_l_pos[1])
        self._ann_r2_flow.set_text("flow: {} {}".format(p_g, max_str))
        self._arr_r2_flow.set_mutation_scale(math.log(abs(p_g) + 1) * 10)

        # update levels
        self._h.set_height(1 - self._agent.get_h())
        self._g.set_height(self._agent.height_ls - self._agent.get_g())

        # list of artists to be drawn
        return [self._ann_time,
                self._ann_power_flow,
                self._arr_power_flow,
                self._g,
                self._h]
