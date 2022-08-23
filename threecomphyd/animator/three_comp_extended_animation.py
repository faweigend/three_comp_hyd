import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent
from threecomphyd.animator.base_animation import BaseAnimation
from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation


class ThreeCompExtendedAnimation(BaseAnimation, ThreeCompVisualisation):
    """
    creates an animation to visualise power flow within the hydraulics model
    """

    def __init__(self, agent: ThreeCompHydAgent, hz: int = 10, controls: bool = False, frames: int = None):
        """
        Whole animation setup using given agent
        """

        # figure layout
        fig = plt.figure(figsize=(12, 6))
        self._ax1 = fig.add_subplot(1, 2, 1)
        self._ax2 = fig.add_subplot(1, 2, 2)

        ThreeCompVisualisation.__init__(self, axis=self._ax1, agent=agent, animated=True)

        # this is the indicator dot for the test run
        self.__dot = Line2D([], [], color='red', marker='o', markeredgecolor='r')

        self.__command_power = []
        self.__command_times = []

        super().__init__(figure=fig, hz=hz, controls=controls, frames=frames)

    def set_run_commands(self, times_data, power_data):
        """
        let's the animation run a test
        :param times_data:
        :param power_data:
        :return:
        """

        self.__command_times = times_data
        self.__command_power = power_data

        plot_times = []
        plot_power = []
        for i, t in enumerate(times_data):
            if i > 0:
                plot_times.append(t)
                plot_power.append(power_data[i - 1])
            plot_times.append(t)
            plot_power.append(power_data[i])

        self._ax2.clear()
        self._ax2.plot(plot_times, plot_power)
        self._ax2.add_line(self.__dot)
        self._ax2.set_ylabel("power(W)")
        self._ax2.set_xlabel("time(s)")

    def _init_layout(self):
        """
        format axis and add descriptions
        :return: list of artists to draw
        """
        pass

    def _update_data(self, frame_number):
        """
        The function to call at each frame.
        :param frame_number: frame number
        :return: an iterable of artists
        """

        cur_time = self._agent.get_time()
        # Iterate through times
        if len(self.__command_times) > 0 and cur_time > self.__command_times[0]:
            p = self.__command_power.pop(0)
            self.__command_times.pop(0)
            self._agent.set_power(p)
            # update dot power position
            self.__dot.set_ydata(p)
        # update dot time position
        self.__dot.set_xdata(cur_time)

        ThreeCompVisualisation.update_animation_data(self, frame_number)
