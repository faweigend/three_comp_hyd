from abc import abstractmethod

from w_pm_hydraulic.animate.base_animation import BaseAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets


class InteractiveAnimation(BaseAnimation):
    """
    An animation that adds buttons to affect the simulation during playtime
    """

    def __init__(self, figure, agent):
        """
        simple constructor
        :param figure:
        :param agent:
        """
        self._agent = agent

        # animation start, stop, step forward control buttons
        p_control_ax = figure.add_axes([0.65, 0.92, 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(p_control_ax)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        pfax = divider.append_axes("right", size="100%", pad=0.05)
        qfax = divider.append_axes("right", size="100%", pad=0.05)
        rfax = divider.append_axes("right", size="100%", pad=0.05)

        self.__button_mm = matplotlib.widgets.Button(p_control_ax, label="-50")
        self.__button_mm.on_clicked(self.__power_mm)
        self.__button_m = matplotlib.widgets.Button(ofax, label="-10")
        self.__button_m.on_clicked(self.__power_m)
        self.__button_p = matplotlib.widgets.Button(pfax, label="+10")
        self.__button_p.on_clicked(self.__power_p)
        self.__button_pp = matplotlib.widgets.Button(qfax, label="+50")
        self.__button_pp.on_clicked(self.__power_pp)
        self.__button_r = matplotlib.widgets.Button(rfax, label="$\u21BA$")
        self.__button_r.on_clicked(self.__reset)

        BaseAnimation.__init__(self, figure=figure, hz=self._agent.hz)

    def __reset(self, event=None):
        """
        resets the agent
        :param event:
        :return:
        """
        self._agent.reset()

    def __power_p(self, event=None):
        """
        increases agent's power output
        :param event:
        """
        p = self._agent.get_power()
        self._agent.set_power(p + 10)

    def __power_pp(self, event=None):
        """
        increases agent's power output
        :param event:
        """
        p = self._agent.get_power()
        self._agent.set_power(p + 50)

    def __power_m(self, event=None):
        """
        decreases agent's power output
        :param event:
        """
        p = self._agent.get_power()
        # clamp at 0
        self._agent.set_power(max(0, p - 10))

    def __power_mm(self, event=None):
        """
        decreases agent's power output
        :param event:
        """
        p = self._agent.get_power()
        # clamp at 0
        self._agent.set_power(max(0, p - 50))

    @abstractmethod
    def _init_layout(self):
        """
        format axis and add descriptions
        :return: list of artists to draw
        """

    @abstractmethod
    def _update_data(self, frame_number):
        """
        The function to call at each frame.
        :param frame_number: frame number
        :return: an iterable of artists
        """
