import matplotlib.pyplot as plt

from threecomphyd.visualiser.three_comp_visualisation import ThreeCompVisualisation
from threecomphyd.animator.interactive_animation import InteractiveAnimation
from threecomphyd.agents.three_comp_hyd_agent import ThreeCompHydAgent


class ThreeCompInteractiveAnimation(InteractiveAnimation, ThreeCompVisualisation):
    """
    creates an animation to visualise power flow within the three component hydraulics model
    """

    def __init__(self, agent: ThreeCompHydAgent):
        """
        Whole animation setup using given agent
        """

        self._fig = plt.figure(figsize=(10, 6))
        ax1 = self._fig.add_subplot(1, 1, 1)

        # Three comp base vis
        ThreeCompVisualisation.__init__(self, axis=ax1, agent=agent, animated=True, basic_annotations=True)

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
        ThreeCompVisualisation.update_animation_data(self, frame_number=frame_number)
