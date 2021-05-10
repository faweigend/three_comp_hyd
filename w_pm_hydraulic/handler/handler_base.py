import json
import logging
import os

import config


class HandlerBase:
    """
    Handler simply rely on a save, load and update functionality
    """

    def __init__(self):
        """
        default constructor with a path addition to be overwritten by inheriting classes
        """
        self._dir_path_addition = "HandlerBase"

    def _update_json(self, data: dict, file: str):
        """
        updates existing metadata
        :param data: data to store into json
        :param file: filename
        """
        path = os.path.join(self._get_dir_path(), file)
        try:
            to_update = self._load_json(path)
        except FileNotFoundError:
            # simply save a new file if given path doesn't exist
            self._save_json(data, file)
            return

        # update loaded dict and overwrite
        to_update.update(data)
        self._save_json(data, file)

    def _save_json(self, data: dict, file: str, indent: bool = True):
        """
        saves metadata into standard json format
        :param data: data to store into json
        :param file: filename
        """
        path = os.path.join(self._get_dir_path(), file)
        # create directories if they don't exist
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        try:
            logging.info("Saving json to: {}".format(path))
            with open("{}.json".format(path), 'w') as fp:
                if indent is True:
                    json.dump(data, fp, indent=4)
                else:
                    json.dump(data, fp)
        except TypeError:
            logging.warning(data)
            logging.warning("Metadata is not in a format that can be converted to json: {}".format(path))

    def _load_json(self, file: str):
        """
        load and parse json into dict
        :param file: filename
        :return: dict in structure of read json
        """
        path = os.path.join(self._get_dir_path(), "{}.json".format(file))
        with open(path, 'rb') as fp:
            return json.load(fp)

    def _get_dir_path(self):
        """
        Simple getter for the data storage path
        :return: path
        """
        return os.path.join(config.paths["data_storage"], self._dir_path_addition)
