import logging
import os
from abc import abstractmethod

import mpl_toolkits.axes_grid1
import matplotlib.widgets
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib.text import Text

from threecomphyd import config


class BaseAnimation(animation.FuncAnimation):
    """
    The basic animation structure to allow start, stop and step forward controls.
    A second clock is drawn next to the controls and updated within the frame_generator.

    This is an abstract class. It provides two abstract methods for initiation and the update step per frame.
    Subclasses have to make sure to call this class' constructor.
    Run the whole thing with the run function.
    """

    def __init__(self, figure, hz: int, controls: bool = True, frames: int = None):
        """
        Animation controls setup
        """
        # drawing loop management
        self.__frame_count = 0
        self._seconds = 0
        self.__runs = True
        self._hz = hz
        self._name = "base_animation"

        # logging counter to issue an update every 1000 frames
        self.__prog_log = 1000

        # animation start, stop, step forward control buttons
        playerax = figure.add_axes([0.125, 0.92, 0.32, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)

        # manage control buttons
        if controls:
            ofax = divider.append_axes("right", size="100%", pad=0.05)
            fax = divider.append_axes("right", size="100%", pad=0.05)
            tax = divider.append_axes("right", size="100%", pad=0.05)
            self.__button_stop = matplotlib.widgets.Button(playerax, label=u'$\u25A0$')
            self.__button_stop.on_clicked(self._stop_run)
            self.__button_oneforward = matplotlib.widgets.Button(ofax, label=u'$\u29D0$')
            self.__button_oneforward.on_clicked(self._oneforward)
            self.__button_forward = matplotlib.widgets.Button(fax, label=u'$\u25B6$')
            self.__button_forward.on_clicked(self.__forward)
        else:
            playerax.set_axis_off()
            tax = playerax

        # visible second clock. Text is set in frame generator
        self.__timer = Text(0.5, 0, ha="center")
        tax.add_artist(self.__timer)
        tax.set_axis_off()

        # provide an infinite generator if no frame number is given
        frames = self.__frame_generator() if frames is None else frames

        # create the animation object
        super().__init__(figure,
                         interval=int(1000 / hz),
                         frames=frames,
                         blit=False,
                         init_func=self._init_layout,
                         func=self._update_data)

    def reset_stats(self):
        """
        resets internal stats
        :return:
        """
        self.__frame_count = 0
        self._seconds = 0

    def __frame_generator(self):
        """
        simple iterator that keeps the interruptible but endless loop running while maintaining the
        correct framecount
        :return: frame number
        """
        while self.__runs is True:
            self.__frame_count += 1
            self._seconds = self.__frame_count / self._hz
            self.__timer.set_text("t: {}".format(int(self._seconds)))

            # log a regular update
            self.__prog_log -= 1
            if self.__prog_log < 0:
                self.__prog_log = 1000
                logging.info("{} seconds rendered".format(self._seconds))

            yield self.__frame_count

    def _stop_run(self, event=None):
        """
        stops the loop
        :param event:
        """
        self.__runs = False
        self.event_source.stop()

    def __forward(self, event=None):
        """
        starts the loop
        :param event:
        """
        self.__runs = True
        self.event_source.start()

    def _oneforward(self, event=None):
        """
        manually draws next frame
        :param event:
        """
        self.__frame_count += 1
        super()._draw_next_frame(self.__frame_count, self._blit)

    def save_time_span_to_file(self, name: str, frames: int):
        """
        Saves time span of given length (seconds) to file with given name
        :param name:
        :param frames:
        :return:
        """
        self.save_count = frames
        dirpath = os.path.join(config.paths["data_storage"], "animations")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        # Set up formatting for the movie files
        self.save(os.path.join(dirpath, name + ".mp4"), fps=self._hz)
        self._stop_run()

    def save_to_file(self, name: str):
        """
        Saves time span of given length (seconds) to file with given name
        :param name:
        :param frames:
        :return:
        """
        dirpath = os.path.join(config.paths["data_storage"], "animations")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        # Set up formatting for the movie files
        self.save(os.path.join(dirpath, name + ".mp4"), fps=self._hz)
        self._stop_run()

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

    @staticmethod
    def run():
        """
        display the animation
        :return:
        """
        plt.show()
