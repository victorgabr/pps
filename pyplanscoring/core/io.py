import bz2
import json
import pickle


def load(filename):
    """
        Loads a Calibration Object into a file using gzip and Pickle
    :param filename: Calibration filemane *.fco
    :return: object
    """
    with bz2.BZ2File(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save(obj, filename, protocol=-1):
    """
        Saves  Object into a file using gzip and Pickle
    :param obj: Calibration Object
    :param filename: Filename *.fco
    :param protocol: cPickle protocol
    """
    # TODO check if compression is necessary
    with bz2.BZ2File(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def save_dvh_json(dvh_data_dict, file_path_name):
    """
        Helper function to save dvh_data into JSON file
    :param dvh_data_dict:
    :param file_path_name:
    """

    with open(file_path_name, 'w', encoding='utf-8') as json_file:
        json.dump(dvh_data_dict, json_file, ensure_ascii=False)


def load_dvh_json(file_path_name):
    """

    :param file_path_name:
    :return:
    """

    with open(file_path_name, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def normalize_data(dvh_data_dict):
    pass


class IOHandler:

    def __init__(self, dvh_data_dict, header_info=None):
        """
            Class to encapsulate IO methods for DVH data storage
            It receives a PyPlanScoring DVH data dictionary
        :param dvh_data_dict: PyPlanScoring DVH data dictionary
        """
        self._header = None
        self._dvh_data = {}

        # setters
        self.dvh_data = dvh_data_dict
        self.header = header_info

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        # TODO define appropriate header
        self._header = value

    @property
    def dvh_data(self):
        return self._dvh_data

    @dvh_data.setter
    def dvh_data(self, value):
        # TODO add object checks
        self._dvh_data = dict(value)

    def to_dvh_file(self, file_path_name):
        """
            Save pickle *.dvh file
        :param file_path_name:
        """
        save(self.dvh_data, file_path_name)

    def read_dvh_file(self, file_path_name):
        """
            Loads pickle *.dvh file
        :param file_path_name:
        """
        self.dvh_data = load(file_path_name)

        return self.dvh_data

    def to_json_file(self, file_path_name):
        """
            Saves serialized dvh data into *.jdvh json file
        :param file_path_name:
        """
        save_dvh_json(self.dvh_data, file_path_name)

    def read_json_file(self, file_path_name):
        """
            Saves serialized dvh data into *.jdvh json file
        :param file_path_name:
        """
        self.dvh_data = load_dvh_json(file_path_name)
        return self.dvh_data
