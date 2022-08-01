import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams
from threecomphyd.agents.two_comp_hyd_agent import TwoCompHydAgent


class TwoCompVisualisation:
    """
    Basis to visualise power flow within the two tank hydraulics model as an animation or simulation
    """

    def __init__(self, agent: TwoCompHydAgent,
                 axis: plt.axis = None,
                 animated: bool = False,
                 detail_annotations: bool = False,
                 basic_annotations: bool = True,
                 black_and_white: bool = False,
                 all_outlines: bool = True):
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
        :param all_outlines: If true, adds outlines around liquid flow arrows and half of U
        """

        self.__all_outlines = all_outlines

        # plot if no axis was assigned
        if axis is None:
            fig = plt.figure(figsize=(4, 4.2))
            self._ax1 = fig.add_subplot(1, 1, 1)
        else:
            fig = None
            self._ax1 = axis

        if black_and_white:
            self.__ae_color = (0.7, 0.7, 0.7)
            self.__w_p_color = (0.5, 0.5, 0.5)
            self.__ann_color = (0, 0, 0)
            self.__p_color = (0.5, 0.5, 0.5)

        elif not black_and_white:
            self.__ae_color = "tab:cyan"
            self.__w_p_color = "tab:orange"
            self.__ann_color = "tab:blue"
            self.__p_color = "tab:green"

        # basic parameters for setup
        self._animated = animated
        self.__detail_annotations = detail_annotations
        self._agent = agent
        self.__offset = 0.2

        self.__width_ae = 0.5  # tank with three stripes
        self.__width_w_p = 0.4
        self._ae = None
        self._ae1 = None
        self._ae2 = None
        self._r1 = None  # line marking flow from Ae to W'
        self._ann_ae = None  # Ae annotation

        # An tank
        self._an = None
        self._h = None  # fill state
        self._ann_an = None  # annotation

        # finish the basic layout
        self.__set_basic_layout()
        self.update_basic_layout(agent)

        # now the animation components
        if self._animated:
            # R1 flow
            self._arr_r1_flow = None
            self._ann_r1_flow = None

            # flow out of tap
            self._arr_power_flow = None
            self._ann_power_flow = None

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
            self._ann_r1_max_flow = None
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

        ae_width = self.__width_ae
        an_left = self._an.get_x()
        an_width = self._an.get_width()

        # some offset to the bottom
        offset = self.__offset
        phi_o = self._agent.phi + offset

        rcParams['text.usetex'] = True

        self._ann_r1_max_flow = Text(text="$CP$", ha='right', fontsize="xx-large", x=ae_width + 0.09,
                                     y=phi_o - 0.08)
        self._ann_r1_flow = Text(text="$p_{Ae}$", ha='right', fontsize="xx-large", x=ae_width + 0.09,
                                 y=phi_o + 0.06)
        self._arr_r1_flow = FancyArrowPatch((ae_width, phi_o),
                                            (ae_width + 0.1, phi_o),
                                            arrowstyle='-|>',
                                            mutation_scale=30,
                                            color=self.__ae_color)
        self._ax1.annotate('$\phi$',
                           xy=(ae_width / 2, phi_o),
                           xytext=(ae_width / 2, (phi_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\phi$',
                           xy=(ae_width / 2, offset),
                           xytext=(ae_width / 2, (phi_o - offset) / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._ax1.annotate('$\psi$',
                           xy=(ae_width + 0.1 + self.__width_w_p / 2,
                               1 - self._agent.psi + offset),
                           xytext=(ae_width + 0.1 + self.__width_w_p / 2,
                                   1 - self._agent.psi / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$\psi$',
                           xy=(ae_width + 0.1 + self.__width_w_p / 2,
                               1 + offset),
                           xytext=(ae_width + 0.1 + self.__width_w_p / 2,
                                   1 - self._agent.psi / 2 + offset - 0.015),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )

        self._ann_power_flow = Text(text="$p$", ha='center', fontsize="xx-large", x=self._ann_an.get_position()[0],
                                    y=offset - 0.06)
        self._arr_power_flow = FancyArrowPatch((self._ann_an.get_position()[0], offset - 0.078),
                                               (self._ann_an.get_position()[0], 0.0),
                                               arrowstyle='-|>',
                                               mutation_scale=30,
                                               color=self.__p_color)

        self._h.update(dict(xy=(an_left, offset),
                            width=an_width,
                            height=0.15,
                            color=self.__w_p_color))

        self._ax1.annotate('$h$',
                           xy=(self._ann_an.get_position()[0] + 0.07,
                               1 - self._agent.psi + offset),
                           xytext=(self._ann_an.get_position()[0] + 0.07,
                                   (1 - self._agent.psi - self._h.get_height()) / 2 + self._h.get_height() + offset),
                           ha='center',
                           fontsize="xx-large",
                           arrowprops=dict(arrowstyle='-|>',
                                           ls='-',
                                           fc=self.__ann_color)
                           )
        self._ax1.annotate('$h$',
                           xy=(self._ann_an.get_position()[0] + 0.07,
                               self._h.get_height() + offset),
                           xytext=(self._ann_an.get_position()[0] + 0.07,
                                   (1 - self._agent.psi - self._h.get_height()) / 2 + self._h.get_height() + offset),
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

        if self.__all_outlines:
            self._arr_power_flow.set_edgecolor("black")
            self._arr_r1_flow.set_edgecolor("black")

        self._ax1.axhline(offset, linestyle='--', color=self.__ann_color)
        self._ax1.axhline(1 + offset - 0.001, linestyle='--', color=self.__ann_color)
        self._ax1.add_artist(self._ann_power_flow)
        self._ax1.add_artist(self._arr_power_flow)
        self._ax1.add_artist(self._arr_r1_flow)
        self._ax1.add_artist(self._ann_r1_flow)
        self._ax1.add_artist(self._ann_r1_max_flow)

    def __set_animation_layout(self):
        """
        Adds layout components that are required for an animation
        """

        offset = self.__offset
        o_width = self.__width_ae
        phi_o = self._agent.phi + offset

        # Ae flow (R1)
        self._arr_r1_flow = FancyArrowPatch((o_width, phi_o),
                                            (o_width + 0.1, phi_o),
                                            arrowstyle='simple',
                                            mutation_scale=0,
                                            fc=self.__ae_color)
        self._ann_r1_flow = Text(text="flow: ", ha='right', fontsize="large", x=o_width, y=phi_o - 0.05)

        # Tap flow (Power)
        self._arr_power_flow = FancyArrowPatch((self._ann_an.get_position()[0], offset - 0.05),
                                               (self._ann_an.get_position()[0], 0.0),
                                               arrowstyle='simple',
                                               mutation_scale=0,
                                               color=self.__p_color)
        self._ann_power_flow = Text(text="flow: ", ha='center', fontsize="large", x=self._ann_an.get_position()[0],
                                    y=offset - 0.05)

        # information annotation
        self._ann_time = Text(x=1, y=0.9 + offset, ha="right")

        if self.__all_outlines:
            self._arr_power_flow.set_edgecolor("black")
            self._arr_r1_flow.set_edgecolor("black")

        self._ax1.add_artist(self._ann_power_flow)
        self._ax1.add_artist(self._arr_power_flow)
        self._ax1.add_artist(self._arr_r1_flow)
        self._ax1.add_artist(self._ann_r1_flow)

    def __set_basic_layout(self):
        """
        updates position estimations and layout
        """

        # u_left is 0
        u_width = self.__width_ae

        # some offset to the bottom
        offset = self.__offset
        phi_o = self._agent.phi + offset

        # Ae tank
        self._ae = Rectangle((0.0, phi_o), 0.1, 1 + self._agent.phi, color=self.__ae_color, alpha=0.3)
        self._ae1 = Rectangle((0.1, phi_o), 0.1, 1 + self._agent.phi, color=self.__ae_color, alpha=0.6)
        self._ae2 = Rectangle((0.2, phi_o), 0.3, 1 + self._agent.phi, color=self.__ae_color)
        self._r1 = Line2D([u_width, u_width + 0.1],
                          [phi_o, phi_o],
                          color=self.__ae_color)
        self._ann_ae = Text(text="Ae", ha='center', fontsize="xx-large",
                            x=0.25,
                            y=phi_o + ((1 - self._agent.phi) / 2))

        # W' tank
        self._an = Rectangle((self.__width_ae + 0.1, offset), self.__width_w_p, 1 - self._agent.psi, fill=False,
                             ec="black")
        self._h = Rectangle((self.__width_ae + 0.1, offset), self.__width_w_p, 1 - self._agent.psi,
                            color=self.__w_p_color)
        self._ann_an = Text(text="An", ha='center', fontsize="xx-large",
                            x=self.__width_ae + 0.1 + self.__width_w_p / 2,
                            y=offset + (1 - self._agent.psi) / 2)

        # the basic layout
        self._ax1.add_line(self._r1)
        self._ax1.add_artist(self._ae)
        self._ax1.add_artist(self._ae1)
        self._ax1.add_artist(self._ae2)
        self._ax1.add_artist(self._an)
        self._ax1.add_artist(self._h)
        self._ax1.add_artist(self._ann_ae)
        self._ax1.add_artist(self._ann_an)

    def update_basic_layout(self, agent: TwoCompHydAgent):
        """
        updates tank positions and sizes according to new agent
        :param agent: agent to be visualised
        """

        self._agent = agent

        # o_left is 0
        width_ae = self.__width_ae

        # some offset to the bottom
        offset = self.__offset
        phi_o = agent.phi + offset

        # Ae tank
        self._ae.set_bounds(0.0, phi_o, 0.05, 1 - self._agent.phi)
        self._ae1.set_bounds(0.05, phi_o, 0.05, 1 - self._agent.phi)
        self._ae2.set_bounds(0.1, phi_o, width_ae - 0.1, 1 - self._agent.phi)
        self._r1.set_xdata([width_ae, width_ae + 0.1])
        self._r1.set_ydata([phi_o, phi_o])
        self._ann_ae.set_position(xy=(width_ae / 2, ((1 - self._agent.phi) / 2) + phi_o - 0.02))

        # An tank
        self._an.set_bounds(self.__width_ae + 0.1,
                            offset,
                            self.__width_w_p,
                            1 - agent.psi)
        self._h.set_bounds(self.__width_ae + 0.1,
                           offset, self.__width_w_p, 1 - agent.psi)
        self._ann_an.set_position(xy=(
            self.__width_ae + 0.1 + self.__width_w_p / 2,
            ((1 - agent.psi) / 2) + offset - 0.02))

        # update levels
        self._h.set_height(1 - self._agent.psi - self._agent.get_h())

    def hide_basic_annotations(self):
        """
        Simply hides the S, LF, and LS text
        """
        self._ann_ae.set_text("")
        self._ann_an.set_text("")

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

        p_o = round(self._agent.get_p_ae())

        # oxygen arrow
        max_str = "(CP)" if p_o == self._agent.cp else ""
        self._ann_r1_flow.set_text("flow: {} {}".format(p_o * self._agent.hz, max_str))
        self._arr_r1_flow.set_mutation_scale(math.log(p_o + 1) * 10)

        # update levels
        self._h.set_height(1 - self._agent.psi - self._agent.get_h())

        # list of artists to be drawn
        return [self._ann_time,
                self._ann_power_flow,
                self._arr_power_flow,
                self._h]
