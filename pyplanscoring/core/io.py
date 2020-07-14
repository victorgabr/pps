import bz2
import json
import os
import pickle

from pydicom.valuerep import IS

from .dicom_reader import PyDicomParser


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
        json.dump(dvh_data_dict,
                  json_file,
                  ensure_ascii=False)


def load_dvh_json(file_path_name):
    """

    :param file_path_name:
    :return:
    """

    with open(file_path_name, 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        # add pydicom key type (int)
        json_dict = {IS(k): v for k, v in json_dict.items()}
        return json_dict


def normalize_data(dvh_data_dict):
    pass


class IOHandler:

    def __init__(self, dvh_data_dict=None, header_info=None):
        """
            Class to encapsulate IO methods for DVH data storage
            It receives a PyPlanScoring DVH data dictionary
        :param dvh_data_dict: PyPlanScoring DVH data dictionary
        """
        if dvh_data_dict is None:
            dvh_data_dict = {}
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


def get_participant_folder_data(root_path):
    """
        Provide all participant required files (RP,RS an RD DICOM FILES)
    :param root_path: participant folder
    :return: Pandas DataFrame containing path to files

    """
    filtered_files = parse_patient_dicom_folder(root_path)

    missing_files = [key for key, value in filtered_files.items() if value is False]

    if not missing_files:
        return filtered_files, True
    else:
        return missing_files, False


def GetSOPClassUID(ds):
    """Determine the SOP Class UID of the current file."""

    if "SOPClassUID" not in ds:
        return None
    if ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.2":
        return "rtdose"
    elif ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3":
        return "rtss"
    elif ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.5":
        return "rtplan"
    elif ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2":
        return "ct"
    else:
        return None


def parse_patient_dicom_folder(pt_folder):
    """
        Method parse an exported patient folder
    :param pt_folder: path to the patient folder
    :return: dictionary with rtdose, rtss keys.
    """
    dicom_files = [os.path.join(pt_folder, f) for f in os.listdir(pt_folder)]
    filtered_files = {"rtdose": [], "rtss": [], "rtplan": [], "ct": []}
    for dicom_file in dicom_files:
        if os.path.isfile(dicom_file):
            try:
                ds = pydicom.read_file(dicom_file, defer_size=100, force=True)
                rt_type = GetSOPClassUID(ds)
                if rt_type == "rtdose":
                    filtered_files["rtdose"] = dicom_file
                if rt_type == "rtss":
                    filtered_files["rtss"] = dicom_file
                if rt_type == "rtplan":
                    filtered_files["rtplan"] = dicom_file
                if rt_type == "ct":
                    filtered_files["ct"].append(dicom_file)
            except IOError:
                print("{os.path.split(dicom_file)[-1]} is not dicom")

    return filtered_files

