import matplotlib.pyplot as plt

from threecomphyd.agents.two_comp_hyd_agent import TwoCompHydAgent
from threecomphyd.animator.interactive_animation import InteractiveAnimation
from threecomphyd.visualiser.two_comp_visualisation import TwoCompVisualisation


class TwoCompHydAnimation(InteractiveAnimation, TwoCompVisualisation):
    """
    creates an animation to visualise power flow within the two component
    hydraulics model
    """

    def __init__(self, agent: TwoCompHydAgent):
        """
        Whole animation setup using given agent
        """

        # figure layout
        self._fig = plt.figure(figsize=(10, 6))
        ax1 = self._fig.add_subplot(1, 1, 1)

        # Three comp base vis
        TwoCompVisualisation.__init__(self, axis=ax1, agent=agent, animated=True)

        # Power control sim
        InteractiveAnimation.__init__(self, figure=self._fig, agent=agent)

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
        TwoCompVisualisation.update_animation_data(self, frame_number=frame_number)
