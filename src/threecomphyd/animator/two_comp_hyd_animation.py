import math

import matplotlib.pyplot as plt
from threecomphyd.agents.two_comp_hyd_agent import TwoCompHydAgent

from threecomphyd.animator.interactive_animation import InteractiveAnimation

from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D


class TwoCompHydAnimation(InteractiveAnimation):
    """
    creates an animation to visualise power flow within the two component
    hydraulics model with exponential expenditure and recovery
    """

    def __init__(self, agent: TwoCompHydAgent):
        """
        Whole animation setup using given agent
        """

        # figure layout
        fig = plt.figure(figsize=(10, 6))
        self.__ax1 = fig.add_subplot(1, 1, 1)

        self._agent = agent

        offset = 0.2
        phi = self._agent.phi + offset
        psi = self._agent.psi

        # oxygen vessel
        self.__o = Rectangle((0.0, phi), 0.1, 1 + offset - phi, fc="tab:cyan", alpha=0.3)
        self.__o1 = Rectangle((0.1, phi), 0.1, 1 + offset - phi, color='tab:cyan', alpha=0.6)
        self.__o2 = Rectangle((0.2, phi), 0.3, 1 + offset - phi, color='tab:cyan')
        self.__ann_o = Text(text="Ae", ha='center', fontsize="xx-large",
                            x=0.25,
                            y=phi + ((1 + offset - phi) / 2))
        self.__arr_o_flow = FancyArrowPatch((0.5, phi),
                                            (0.6, phi),
                                            arrowstyle='simple',
                                            mutation_scale=0,
                                            ec='white',
                                            fc='tab:cyan')
        self.__ann_o_flow = Text(text="flow: ", ha='right', fontsize="large",
                                 x=0.58, y=phi - 0.1)
        self.__r1 = Line2D([0.5, 0.6], [phi, phi], color="tab:cyan")

        # anaerobic vessel
        self.__a_p = Rectangle((0.6, offset), 0.2, 1 - psi + offset, fill=False, ec="black")
        self.__h = Rectangle((0.6, offset), 0.2, 1 - psi + offset, fc='tab:orange')
        self.__ann_p = Text(text="W\'", ha='center', fontsize="xx-large",
                            x=0.7,
                            y=offset + (1 - psi + offset) / 2)
        self.__arr_power_flow = FancyArrowPatch((self.__ann_p.get_position()[0], offset),
                                                (self.__ann_p.get_position()[0], offset - 0.15),
                                                arrowstyle='simple',
                                                mutation_scale=0,
                                                ec='white',
                                                fc='tab:green')
        self.__ann_power_flow = Text(text="flow: ", ha='center', fontsize="large",
                                     x=self.__ann_p.get_position()[0] + 0.1, y=offset - 0.1)

        # information annotation
        self.__ann_time = Text(x=1, y=1, ha="right")

        super().__init__(figure=fig, agent=agent)

    def _init_layout(self):
        """
        format axis and add descriptions
        :return: list of artists to draw
        """
        self.__ax1.add_artist(self.__o)
        self.__ax1.add_artist(self.__o1)
        self.__ax1.add_artist(self.__o2)
        self.__ax1.add_line(self.__r1)
        self.__ax1.add_artist(self.__a_p)
        self.__ax1.add_artist(self.__h)
        self.__ax1.add_artist(self.__ann_o)
        self.__ax1.add_artist(self.__ann_p)
        self.__ax1.add_artist(self.__ann_power_flow)
        self.__ax1.add_artist(self.__arr_power_flow)
        self.__ax1.add_artist(self.__arr_o_flow)
        self.__ax1.add_artist(self.__ann_o_flow)
        self.__ax1.set_xlim(0, 1)
        self.__ax1.set_ylim(0, 1.2)
        self.__ax1.set_axis_off()

        self.__ax1.add_artist(self.__ann_time)

        return []

    def _update_data(self, frame_number):
        """
        The function to call at each frame.
        :param frame_number: frame number
        :return: an iterable of artists
        """

        # perform one step
        cur_time = self._agent.get_time()
        power = self._agent.perform_one_step()

        # draw some information
        self.__ann_time.set_text("agent \n time: {}".format(int(cur_time)))

        # power arrow
        self.__ann_power_flow.set_text("power: {}".format(round(power)))
        self.__arr_power_flow.set_mutation_scale(math.log(power + 1) * 10)

        p_o = round(self._agent.get_p_o())

        # oxygen arrow
        max_str = "(CP)" if p_o == self._agent.m_u else ""
        self.__ann_o_flow.set_text("flow: {} {}".format(p_o, max_str))
        self.__arr_o_flow.set_mutation_scale(math.log(p_o + 1) * 10)

        # update levels
        self.__h.set_height(1 - self._agent.get_h())

        # list of artists to be drawn
        return [self.__ann_time,
                self.__ann_power_flow,
                self.__arr_power_flow,
                self.__h]
