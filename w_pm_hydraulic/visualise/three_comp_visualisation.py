import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from w_pm_hydraulic.agents.three_comp_hyd_agent import ThreeCompHydAgent

from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams


class ThreeCompVisualisation:
    """
    basis to visualise power flow within the hydraulics model as an animation or simulation
    """

    def __init__(self, agent: ThreeCompHydAgent,
                 axis: plt.axis = None,
                 animated: bool = False,
                 detail_annotations: bool = False,
                 black_and_white: bool = False):
        """
        Whole visualisation setup using given agent's parameters
        """
        # matplotlib fontsize
        rcParams['font.size'] = 12

        # plot if no axis was assigned
        if axis is None:
            fig = plt.figure(figsize=(8, 5))
            self._ax1 = fig.add_subplot(1, 1, 1)
        else:
            fig = None
            self._ax1 = axis

        if black_and_white is True:
            self.__ae_color = (0.7, 0.7, 0.7)
            self.__anf_color = (0.5, 0.5, 0.5)
            self.__ans_color = (0.3, 0.3, 0.3)
            self.__ann_color = (0, 0, 0)
            self.__p_color = (0.5, 0.5, 0.5)

        elif black_and_white is False:
            self.__ae_color = "tab:cyan"
            self.__anf_color = "tab:orange"
            self.__ans_color = "tab:red"
            self.__ann_color = "tab:blue"
            self.__p_color = "tab:green"

        # basic parameters for setup
        self._animated = animated
        self.__detail_annotations = detail_annotations
        self._agent = agent
        self.__offset = 0.2

        # Ae tank with three stripes
        self.__width_o = 0.3
        self._o = None
        self._o1 = None
        self._o2 = None
        self._r1 = None  # line marking flow from Ae to AnF
        self._ann_o = None  # Ae annotation

        # AnF tank
        self._a_anf = None
        self._h = None  # fill state
        self._ann_anf = None  # annotation

        # AnS tank
        self._a_ans = None
        self._g = None
        self._ann_ans = None  # annotation AnS
        self._r2 = None  # line marking flow from AnS to AnF

        # finish the basic layout
        self.__set_basic_layout()

        # now the animation components
        if self._animated is True:
            # Ae flow
            self._arr_o_flow = None
            self._ann_o_flow = None

            # flow out of tap
            self._arr_power_flow = None
            self._ann_power_flow = None

            # AnS flow (R2)
            self._arr_r2_l_pos = None
            self._arr_r2_flow = None
            self._ann_r2_flow = None

            # time information annotation
            self._ann_time = None

            self.__set_animation_layout()
            self._ax1.add_artist(self._ann_time)

        # add layout for detailed annotations
        # detail annotation add greek letters for distances and positions
        if self.__detail_annotations is True:
            self.__set_detailed_annotations_layout()
            self._ax1.set_xlim(0, 1.05)
            self._ax1.set_ylim(0, 1.2)
        else:
            self._ax1.set_xlim(0, 1.0)
            self._ax1.set_ylim(0, 1.2)

        if self.__detail_annotations is True and self._animated is True:
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

        o_width = self.__width_o
        ans_left = self._a_ans.get_x()
        ans_width = self._a_ans.get_width()
        anf_left = self._a_anf.get_x()
        anf_width = self._a_anf.get_width()
        ans_height = self._a_ans.get_height()

        # some offset to the bottom
        offset = self.__offset
        phi_o = self._agent.phi + offset
        gamma_o = self._agent.gamma + offset

        font = FontProperties()
        font.set_family('serif')
        font.set_name('Times New Roman')
        rcParams['text.usetex'] = True

        self._ann_o_flow = Text(text="$m^{Ae}$", ha='right', fontsize="xx-large", x=o_width + 0.09,
                                y=phi_o - 0.08)
        ann_p_ae = Text(text="$p^{Ae}$", ha='right', fontsize="xx-large", x=o_width + 0.07,
                        y=phi_o + 0.03)
        self._arr_o_flow = FancyArrowPatch((o_width, phi_o),
                                           (o_width + 0.1, phi_o),
                                           arrowstyle='-|>',
                                           mutation_scale=30,
                                           lw=2,
                                           color=self.__ae_color)
        self._ax1.annotate('$\phi$',
                           xy=(o_width / 2, phi_o),
                           xytext=(o_width / 2, (phi_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\phi$',
                           xy=(o_width / 2, offset),
                           xytext=(o_width / 2, (phi_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._ann_power_flow = Text(text="$p$", ha='center', fontsize="xx-large", x=self._ann_anf.get_position()[0],
                                    y=offset - 0.06)
        self._arr_power_flow = FancyArrowPatch((self._ann_anf.get_position()[0], offset - 0.078),
                                               (self._ann_anf.get_position()[0], 0.0),
                                               arrowstyle='-|>',
                                               mutation_scale=30,
                                               lw=2,
                                               color=self.__p_color)

        self._ax1.annotate('$h$',
                           xy=(self._ann_anf.get_position()[0] + 0.07, 1 + offset),
                           xytext=(self._ann_anf.get_position()[0] + 0.07, 1 + offset - 0.30),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$h$',
                           xy=(self._ann_anf.get_position()[0] + 0.07, 1 + offset - 0.55),
                           xytext=(self._ann_anf.get_position()[0] + 0.07, 1 + offset - 0.30),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._h.update(dict(xy=(anf_left, offset),
                            width=anf_width,
                            height=1 - 0.55,
                            color=self.__anf_color))

        self._ax1.annotate('$g$',
                           xy=(ans_left + ans_width + 0.02, ans_height + gamma_o),
                           xytext=(ans_left + ans_width + 0.02, ans_height * 0.61 + gamma_o),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$g$',
                           xy=(ans_left + ans_width + 0.02, ans_height * 0.3 + gamma_o),
                           xytext=(ans_left + ans_width + 0.02, ans_height * 0.61 + gamma_o),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._g.update(dict(xy=(ans_left, gamma_o),
                            width=ans_width,
                            height=ans_height * 0.3,
                            color=self.__ans_color))

        ann_p_an = Text(text="$p^{An}$", ha='left', usetex=True, fontsize="xx-large", x=ans_left - 0.06,
                        y=gamma_o + 0.11)
        ann_arr_flow = Text(text="$m^{AnS}$", ha='left', usetex=True, fontsize="xx-large", x=ans_left - 0.09,
                            y=gamma_o + 0.03)
        self._ann_r2_flow = Text(text="$m^{AnF}$", ha='left', usetex=True, fontsize="xx-large", x=ans_left - 0.04,
                                 y=gamma_o - 0.07)

        self._arr_r2_flow = FancyArrowPatch((ans_left - 0.1, gamma_o),
                                            (ans_left, gamma_o),
                                            arrowstyle='<|-|>',
                                            mutation_scale=30,
                                            lw=2,
                                            color=self.__ans_color)

        self._ax1.annotate('$\\theta$',
                           xy=(ans_left + ans_width / 2, 1 + offset),
                           xytext=(
                               ans_left + ans_width / 2,
                               1 - (1 - (ans_height + gamma_o - offset)) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\\theta$',
                           xy=(ans_left + ans_width / 2, ans_height + gamma_o),
                           xytext=(
                               ans_left + ans_width / 2,
                               1 - (1 - (ans_height + gamma_o - offset)) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._ax1.annotate('$\\gamma$',
                           xy=(ans_left + ans_width / 2, offset),
                           xytext=(ans_left + ans_width / 2, (gamma_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\\gamma$',
                           xy=(ans_left + ans_width / 2, gamma_o),
                           xytext=(ans_left + ans_width / 2, (gamma_o - offset) / 2 + offset - 0.015),
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
        self._ax1.add_artist(ann_p_an)
        self._ax1.add_artist(ann_p_ae)
        self._ax1.axhline(offset, linestyle='--', color=self.__ann_color)
        self._ax1.axhline(1 + offset - 0.001, linestyle='--', color=self.__ann_color)
        self._ax1.add_artist(self._ann_power_flow)
        self._ax1.add_artist(self._arr_power_flow)
        self._ax1.add_artist(self._arr_o_flow)
        self._ax1.add_artist(self._ann_o_flow)
        self._ax1.add_artist(self._arr_r2_flow)
        self._ax1.add_artist(self._ann_r2_flow)

    def __set_animation_layout(self):
        """
        Adds layout components that are required for an animation
        :return:
        """

        offset = self.__offset
        o_width = self.__width_o
        phi_o = self._agent.phi + offset
        gamma_o = self._agent.gamma + offset

        # Ae flow (R1)
        self._arr_o_flow = FancyArrowPatch((o_width, phi_o),
                                           (o_width + 0.1, phi_o),
                                           arrowstyle='simple',
                                           mutation_scale=0,
                                           ec='white',
                                           fc=self.__ae_color)
        self._ann_o_flow = Text(text="flow: ", ha='right', fontsize="large", x=o_width, y=phi_o - 0.05)

        # Tap flow (Power)
        self._arr_power_flow = FancyArrowPatch((self._ann_anf.get_position()[0], offset - 0.05),
                                               (self._ann_anf.get_position()[0], 0.0),
                                               arrowstyle='simple',
                                               mutation_scale=0,
                                               ec='white',
                                               color=self.__p_color)
        self._ann_power_flow = Text(text="flow: ", ha='center', fontsize="large", x=self._ann_anf.get_position()[0],
                                    y=offset - 0.05)

        # AnS flow (R2)
        self._arr_r2_l_pos = [(self._a_ans.get_x(), gamma_o),
                              (self._a_ans.get_x() - 0.1, gamma_o)]
        self._arr_r2_flow = FancyArrowPatch(self._arr_r2_l_pos[0],
                                            self._arr_r2_l_pos[1],
                                            arrowstyle='simple',
                                            mutation_scale=0,
                                            ec='white',
                                            color=self.__ans_color)
        self._ann_r2_flow = Text(text="flow: ", ha='left', fontsize="large", x=self._a_ans.get_x(),
                                 y=gamma_o - 0.05)

        # information annotation
        self._ann_time = Text(x=1, y=0.9 + offset, ha="right")

        self._ax1.add_artist(self._ann_power_flow)
        self._ax1.add_artist(self._arr_power_flow)
        self._ax1.add_artist(self._arr_o_flow)
        self._ax1.add_artist(self._ann_o_flow)
        self._ax1.add_artist(self._arr_r2_flow)
        self._ax1.add_artist(self._ann_r2_flow)

    def __set_basic_layout(self):
        """
        updates position estimations and layout
        :return:
        """

        # get sizes from agent
        a_anf = self._agent.a_anf
        a_ans = self._agent.a_ans
        ans_height = self._agent.height_ans

        # o_left is 0
        o_width = self.__width_o

        # determine width with size ratio retained
        anf_left = o_width + 0.1
        anf_width = ((a_anf * ans_height) * (1 - anf_left - 0.1)) / (a_ans + a_anf * ans_height)
        ans_left = anf_left + anf_width + 0.1
        ans_width = 1 - ans_left

        # some offset to the bottom
        offset = self.__offset
        phi_o = self._agent.phi + offset
        gamma_o = self._agent.gamma + offset

        # Ae tank
        self._o = Rectangle((0.0, phi_o), 0.05, 1 - self._agent.phi, color=self.__ae_color, alpha=0.3)
        self._o1 = Rectangle((0.05, phi_o), 0.05, 1 - self._agent.phi, color=self.__ae_color, alpha=0.6)
        self._o2 = Rectangle((0.1, phi_o), o_width - 0.1, 1 - self._agent.phi, color=self.__ae_color)
        self._ann_o = Text(text="$Ae$", ha='center', fontsize="xx-large",
                           x=o_width / 2,
                           y=((1 - self._agent.phi) / 2) + phi_o - 0.02)
        self._r1 = Line2D([o_width, o_width + 0.1],
                          [phi_o, phi_o],
                          color=self.__ae_color)

        # AnF vessel
        self._a_anf = Rectangle((anf_left, offset), anf_width, 1, fill=False, ec="black")
        self._h = Rectangle((anf_left, offset), anf_width, 1, color=self.__anf_color)
        self._ann_anf = Text(text="$AnF$", ha='center', fontsize="xx-large",
                             x=anf_left + (anf_width / 2),
                             y=offset + 0.5 - 0.02)

        # AnS vessel
        self._a_ans = Rectangle((ans_left, gamma_o), ans_width, ans_height, fill=False, ec="black")
        self._g = Rectangle((ans_left, gamma_o), ans_width, ans_height, color=self.__ans_color)
        self._ann_ans = Text(text="$AnS$", ha='center', fontsize="xx-large",
                             x=ans_left + (ans_width / 2),
                             y=gamma_o + (ans_height / 2) - 0.02)
        self._r2 = Line2D([ans_left, ans_left - 0.1],
                          [gamma_o, gamma_o],
                          color=self.__ans_color)

        # the basic layout
        self._ax1.add_line(self._r1)
        self._ax1.add_line(self._r2)
        self._ax1.add_artist(self._o)
        self._ax1.add_artist(self._o1)
        self._ax1.add_artist(self._o2)
        self._ax1.add_artist(self._a_anf)
        self._ax1.add_artist(self._a_ans)
        self._ax1.add_artist(self._h)
        self._ax1.add_artist(self._g)
        self._ax1.add_artist(self._ann_o)
        self._ax1.add_artist(self._ann_anf)
        self._ax1.add_artist(self._ann_ans)

    def update_basic_layout(self, agent):
        """
        updates tank positions and sizes according to new agent
        :param agent:
        :return:
        """

        self._agent = agent

        # get sizes from agent
        a_anf = agent.a_anf
        a_ans = agent.a_ans
        ans_height = agent.height_ans

        # o_left is 0
        o_width = self.__width_o

        # determine width with size ratio retained
        anf_left = o_width + 0.1
        anf_width = ((a_anf * ans_height) * (1 - anf_left - 0.1)) / (a_ans + a_anf * ans_height)
        ans_left = anf_left + anf_width + 0.1
        ans_width = 1 - ans_left

        # some offset to the bottom
        offset = self.__offset
        phi_o = agent.phi + offset
        gamma_o = agent.gamma + offset

        # Ae tank
        self._o.set_bounds(0.0, phi_o, 0.05, 1 - self._agent.phi)
        self._o1.set_bounds(0.05, phi_o, 0.05, 1 - self._agent.phi)
        self._o2.set_bounds(0.1, phi_o, o_width - 0.1, 1 - self._agent.phi)
        self._ann_o.set_position(xy=(o_width / 2, ((1 - self._agent.phi) / 2) + phi_o - 0.02))
        self._r1.set_xdata([o_width, o_width + 0.1])
        self._r1.set_ydata([phi_o, phi_o])

        # AnF vessel
        self._a_anf.set_bounds(anf_left, offset, anf_width, 1)
        self._h.set_bounds(anf_left, offset, anf_width, 1)
        self._ann_anf.set_position(xy=(anf_left + (anf_width / 2),
                                       offset + 0.5 - 0.02))

        # AnS vessel
        self._a_ans.set_bounds(ans_left, gamma_o, ans_width, ans_height)
        self._g.set_bounds(ans_left, gamma_o, ans_width, ans_height)
        self._ann_ans.set_position(xy=(ans_left + (ans_width / 2), gamma_o + (ans_height / 2) - 0.02))
        self._r2.set_xdata([ans_left, ans_left - 0.1])
        self._r2.set_ydata([gamma_o, gamma_o])

    def update_animation_data(self, frame_number):
        """
        For animations and simulations.
        The function to call at each frame.
        :param frame_number: frame number
        :return: an iterable of artists
        """

        if self._animated is False:
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
        p_o = round(self._agent.get_p_ae(), 1)
        max_str = "(MAX)" if p_o == self._agent.m_ae else ""
        self._ann_o_flow.set_text("flow: {} {}".format(p_o, max_str))
        self._arr_o_flow.set_mutation_scale(math.log(p_o + 1) * 10)

        # lactate arrow
        p_g = round(self._agent.get_p_an(), 1)
        if p_g < 0:
            max_str = "(MAX)" if p_g == self._agent.m_anf else ""
            self._arr_r2_flow.set_positions(self._arr_r2_l_pos[1], self._arr_r2_l_pos[0])
        else:
            max_str = "(MAX)" if p_g == self._agent.m_ans else ""
            self._arr_r2_flow.set_positions(self._arr_r2_l_pos[0], self._arr_r2_l_pos[1])
        self._ann_r2_flow.set_text("flow: {} {}".format(p_g, max_str))
        self._arr_r2_flow.set_mutation_scale(math.log(abs(p_g) + 1) * 10)

        # update levels
        self._h.set_height(1 - self._agent.get_h())
        self._g.set_height(self._agent.height_ans - self._agent.get_g())

        # list of artists to be drawn
        return [self._ann_time,
                self._ann_power_flow,
                self._arr_power_flow,
                self._g,
                self._h]
