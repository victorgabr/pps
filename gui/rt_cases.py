"""
    Module cases

"""

import abc
import difflib

import os
import string

from constraints.metrics import PlanningItem
from core.dicom_reader import PyDicomParser


class StringMatcher:

    def match(self, test_string, list_of_strings):
        """
            Helper method to match string to a list of strings
        :param test_string:
        :param list_of_strings:
        :return:
        """
        # normalize structure names
        test_strig_normlized = self.normalize_string(test_string)
        list_of_strings_normalzed = [self.normalize_string(s) for s in list_of_strings]

        # map normalized and original strings
        structure_names_map = dict(zip(list_of_strings_normalzed, list_of_strings))
        matches = difflib.get_close_matches(test_strig_normlized, list_of_strings_normalzed, n=1)

        return matches, structure_names_map

    @staticmethod
    def normalize_string(s):
        for p in string.punctuation:
            s = s.replace(p, '')

        return s.upper().strip()


class RTCase:
    def __init__(self, name, case_id, structures, metrics_df):
        self._case_id = case_id
        self._name = name
        self._stuctures = structures
        self._metrics = metrics_df

    @property
    def metrics(self):
        return self._metrics

    @property
    def structures(self):
        return self._stuctures

    @property
    def name(self):
        return self._name

    @property
    def case_id(self):
        return self._case_id

    def get_structure(self, structure_name, matcher=StringMatcher):
        """
             Gets a structure (if it exists from the structure set reference
        :param structure_name:
        :param matcher:  Helper class to match strings
        :return: Structure

        """
        structure_names = [self.structures[k]['name'] for k in self.structures.keys()]

        match, names_map = matcher().match(structure_name, structure_names)
        if match:
            original_name = names_map[match[0]]
            for k in self.structures.keys():
                if original_name == self.structures[k]['name']:
                    return self.structures[k]
        else:
            return "Structure %s not found" % structure_name


# planning item
rs_file_path = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\gui\RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm'
rs_dcm = PyDicomParser(filename=rs_file_path)
rs_dcm.GetSOPInstanceUID()

s_info = rs_dcm.GetStructureInfo()
