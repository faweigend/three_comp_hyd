import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from three_comp_hyd.agents.three_comp_hyd_agent import ThreeCompHydAgent

from matplotlib.text import Text
from matplotlib.patches import Rectangle, Polygon
from matplotlib.patches import FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams


class MortonThreeCompVisualisation:
    """
    basis to visualise power flow within the hydraulics model as an animation or simulation
    """

    def __init__(self, detail_annotations: bool = True, black_and_white: bool = False):
        """
        Whole visualisation setup using given agent's parameters
        """

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

        self.__detail_annotations = detail_annotations

        # plot if no axis was assigned
        fig = plt.figure(figsize=(8, 5))
        self._ax1 = fig.add_subplot(1, 1, 1)

        self._agent = ThreeCompHydAgent(hz=10,
                                        a_anf=3460,
                                        a_ans=6800,
                                        m_ae=200,
                                        m_ans=200,
                                        m_anf=6800000,
                                        the=0.4,
                                        gam=0.23,
                                        phi=0.18)

        a_p = self._agent.a_anf
        a_g = self._agent.a_ans
        g_height = self._agent.height_ans

        # o_left is 0
        o_width = 0.3

        p_left = o_width + 0.1
        # determine width with size ratio retained
        p_width = ((a_p * g_height) * (1 - p_left - 0.1)) / (a_g + a_p * g_height)

        g_left = p_left + p_width + 0.1
        g_width = 1 - g_left

        # some offset to the bottom
        offset = 0.2
        phi_o = self._agent.phi + offset
        lambda_o = self._agent.gamma + offset

        # oxygen vessel
        self._o = Rectangle((0.0, phi_o), 0.05, 1 - self._agent.phi, color=self.__ae_color, alpha=0.3)
        self._o1 = Rectangle((0.05, phi_o), 0.05, 1 - self._agent.phi, color=self.__ae_color, alpha=0.6)
        self._o2 = Rectangle((0.1, phi_o), o_width - 0.1, 1 - self._agent.phi, color=self.__ae_color)
        self._ann_o = Text(text="$O$", ha='center', fontsize="xx-large",
                           x=o_width / 2,
                           y=((1 - self._agent.phi) / 2) + phi_o - 0.02)
        self._arr_o_flow = FancyArrowPatch((o_width, phi_o),
                                           (o_width + 0.1, phi_o),
                                           arrowstyle='simple',
                                           mutation_scale=0,
                                           ec='white',
                                           fc=self.__ae_color)
        self._ann_o_flow = Text(text="flow: ", ha='right', fontsize="large", x=o_width, y=phi_o - 0.05)
        self._r1 = Line2D([o_width, o_width + 0.1],
                          [phi_o, phi_o],
                          color=self.__ae_color)

        # middle vessel
        self._a_p = Rectangle((p_left, offset), p_width, 1, fill=False, ec="black")
        self._h = Rectangle((p_left, offset), p_width, 1, color=self.__anf_color)
        self._ann_p = Text(text="$P$", ha='center', fontsize="xx-large",
                           x=p_left + (p_width / 2),
                           y=offset + 0.5 - 0.02)
        self._arr_power_flow = FancyArrowPatch((self._ann_p.get_position()[0], offset - 0.05),
                                               (self._ann_p.get_position()[0], 0.0),
                                               arrowstyle='simple',
                                               mutation_scale=0,
                                               ec='white',
                                               color=self.__p_color)
        self._ann_power_flow = Text(text="flow: ", ha='center', fontsize="large", x=self._ann_p.get_position()[0],
                                    y=offset - 0.05)

        # lactate vessel
        # self._a_g = Rectangle((g_left, lambda_o), g_width, g_height, fill=False, ec="black")
        self._ax1.add_patch(Polygon([(g_left + g_width - 0.03, 1.2), (g_left + g_width - 0.03, lambda_o + g_height),
                                     (g_left, lambda_o + g_height), (g_left, lambda_o),
                                     (g_left + g_width, lambda_o), (g_left + g_width, 1.2)], closed=False, fill=False,
                                    ec="black"))
        self._ann_b = Text(text="$B$", ha='center', fontsize="xx-large",
                           x=g_left + g_width - 0.05,
                           y=1.0)

        self._g = Rectangle((g_left, lambda_o), g_width, g_height, color=self.__ans_color)
        self._ann_g = Text(text="$G$", ha='center', fontsize="xx-large",
                           x=g_left + (g_width / 2),
                           y=lambda_o + (g_height / 2) - 0.02)

        self._arr_g_l_pos = [(g_left, lambda_o), (g_left - 0.1, lambda_o)]
        self._arr_g_flow = FancyArrowPatch(self._arr_g_l_pos[0],
                                           self._arr_g_l_pos[1],
                                           arrowstyle='simple',
                                           mutation_scale=0,
                                           ec='white',
                                           color=self.__ans_color)
        self._ann_g_flow = Text(text="flow: ", ha='left', fontsize="large", x=g_left, y=lambda_o - 0.05)
        self._r2 = Line2D([g_left, g_left - 0.1],
                          [lambda_o, lambda_o],
                          color=self.__ans_color)

        # information annotation
        self._ann_time = Text(x=1, y=0.9 + offset, ha="right")

        font = FontProperties()
        font.set_family('serif')
        font.set_name('Times New Roman')
        rcParams['text.usetex'] = True

        self._ann_o_flow = Text(text="$R1$",
                                ha='right',
                                fontsize="xx-large",
                                x=o_width + 0.09,
                                y=phi_o - 0.08)

        self._arr_o_flow = FancyArrowPatch((o_width, phi_o),
                                           (o_width + 0.1, phi_o),
                                           arrowstyle='-|>',
                                           mutation_scale=30,
                                           lw=2,
                                           color=self.__ae_color)

        self._ann_power_flow = Text(text="$T$",
                                    ha='center',
                                    fontsize="xx-large",
                                    x=self._ann_p.get_position()[0],
                                    y=offset - 0.06)

        self._arr_power_flow = FancyArrowPatch((self._ann_p.get_position()[0], offset - 0.078),
                                               (self._ann_p.get_position()[0], 0.0),
                                               arrowstyle='-|>',
                                               mutation_scale=30,
                                               lw=2,
                                               color=self.__p_color)

        self._h = Rectangle((p_left, offset), p_width, 1 - 0.55, color=self.__anf_color)

        self._g = Rectangle((g_left, lambda_o), g_width, g_height * 0.3, color=self.__ans_color)

        ann_arr_flow = Text(text="$R2$", ha='left', usetex=True, fontsize="xx-large", x=g_left - 0.09,
                            y=lambda_o + 0.03)
        self._ann_g_flow = Text(text="$R3$", ha='left', usetex=True, fontsize="xx-large", x=g_left - 0.04,
                                y=lambda_o - 0.07)

        self._arr_g_flow = FancyArrowPatch((g_left - 0.1, lambda_o),
                                           (g_left, lambda_o),
                                           arrowstyle='<|-|>',
                                           mutation_scale=30,
                                           lw=2,
                                           color=self.__ans_color)

        if self.__detail_annotations is True:
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
            self._ax1.annotate('$h$',
                               xy=(self._ann_p.get_position()[0] + 0.07, 1 + offset),
                               xytext=(self._ann_p.get_position()[0] + 0.07, 1 + offset - 0.30),
                               ha='center',
                               fontsize="xx-large",
                               arrowprops=dict(arrowstyle='-|>',
                                               ls='-',
                                               fc=self.__ann_color)
                               )
            self._ax1.annotate('$h$',
                               xy=(self._ann_p.get_position()[0] + 0.07, 1 + offset - 0.55),
                               xytext=(self._ann_p.get_position()[0] + 0.07, 1 + offset - 0.30),
                               ha='center',
                               fontsize="xx-large",
                               arrowprops=dict(arrowstyle='-|>',
                                               ls='-',
                                               fc=self.__ann_color)
                               )
            self._ax1.annotate('$g$',
                               xy=(g_left + g_width + 0.02, g_height + lambda_o),
                               xytext=(g_left + g_width + 0.02, g_height * 0.61 + lambda_o),
                               ha='center',
                               fontsize="xx-large",
                               arrowprops=dict(arrowstyle='-|>',
                                               ls='-',
                                               fc=self.__ann_color)
                               )
            self._ax1.annotate('$g$',
                               xy=(g_left + g_width + 0.02, g_height * 0.3 + lambda_o),
                               xytext=(g_left + g_width + 0.02, g_height * 0.61 + lambda_o),
                               ha='center',
                               fontsize="xx-large",
                               arrowprops=dict(arrowstyle='-|>',
                                               ls='-',
                                               fc=self.__ann_color)
                               )

            self._ax1.annotate('$\\theta$',
                               xy=(g_left + g_width / 2, 1 + offset),
                               xytext=(
                                   g_left + g_width / 2, 1 - (1 - (g_height + lambda_o - offset)) / 2 + offset - 0.015),
                               ha='center',
                               fontsize="xx-large",
                               arrowprops=dict(arrowstyle='-|>',
                                               ls='-',
                                               fc=self.__ann_color)
                               )
            self._ax1.annotate('$\\theta$',
                               xy=(g_left + g_width / 2, g_height + lambda_o),
                               xytext=(
                                   g_left + g_width / 2, 1 - (1 - (g_height + lambda_o - offset)) / 2 + offset - 0.015),
                               ha='center',
                               fontsize="xx-large",
                               arrowprops=dict(arrowstyle='-|>',
                                               ls='-',
                                               fc=self.__ann_color)
                               )

            self._ax1.annotate('$\\gamma$',
                               xy=(g_left + g_width / 2, offset),
                               xytext=(g_left + g_width / 2, (lambda_o - offset) / 2 + offset - 0.015),
                               ha='center',
                               fontsize="xx-large",
                               arrowprops=dict(arrowstyle='-|>',
                                               ls='-',
                                               fc=self.__ann_color)
                               )
            self._ax1.annotate('$\\gamma$',
                               xy=(g_left + g_width / 2, lambda_o),
                               xytext=(g_left + g_width / 2, (lambda_o - offset) / 2 + offset - 0.015),
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
            self._ax1.axhline(offset, linestyle='--', color=self.__ann_color)
            self._ax1.axhline(1 + offset, linestyle='--', color=self.__ann_color)

        self._init_layout()
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        plt.show()
        plt.close(fig)

    def _init_layout(self):
        """
        format axis and add descriptions
        :return: list of artists to draw
        """
        self._ax1.add_line(self._r1)
        self._ax1.add_line(self._r2)
        self._ax1.add_artist(self._o)
        self._ax1.add_artist(self._o1)
        self._ax1.add_artist(self._o2)
        self._ax1.add_artist(self._a_p)
        self._ax1.add_artist(self._h)
        self._ax1.add_artist(self._g)
        self._ax1.add_artist(self._ann_o)
        self._ax1.add_artist(self._ann_p)
        self._ax1.add_artist(self._ann_g)
        self._ax1.add_artist(self._arr_power_flow)
        self._ax1.add_artist(self._arr_o_flow)
        self._ax1.add_artist(self._arr_g_flow)

        if self.__detail_annotations is True:
            self._ax1.add_artist(self._ann_power_flow)
            self._ax1.add_artist(self._ann_b)
            self._ax1.add_artist(self._ann_o_flow)
            self._ax1.add_artist(self._ann_g_flow)

        self._ax1.set_xlim(0, 1.05)
        self._ax1.set_ylim(0, 1.2)

        self._ax1.set_axis_off()
        self._ax1.add_artist(self._ann_time)

        return []
