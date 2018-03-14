"""
Classes to DVH metrics
Author: Victor Alves

based on:
https://rexcardan.github.io/ESAPIX/
"""
import difflib
import string

import numpy as np
import pandas as pd

from constraints.query import QueryExtensions, PyQueryExtensions
from core.types import DoseValuePresentation, DoseValue, DoseUnit, DVHData, DICOMType


# TODO refactor to use PyDicomParser class and PyStructure
class PlanningItem:
    """
        Planning items extensions
    """

    def __init__(self, rp_dcm=None, rs_dcm=None, rd_dcm=None):
        self._rp_dcm = rp_dcm
        self._rs_dcm = rs_dcm
        self._rd_dcm = rd_dcm

        # TODO REFACTOR THIS LATER
        try:
            self._plan = rp_dcm.GetPlan()
            self._dose_data = rd_dcm.GetDoseData()
            self._dvhs = rd_dcm.GetDVHs()
            # TODO remove get structures call inside the constructor
            self._structures = rs_dcm.GetStructures()
        except:
            self._plan = {}
            self._dose_data = {}
            self._dvhs = {}
            self._structures = {}

    @property
    def dvh_data(self):
        return self._dvhs

    @dvh_data.setter
    def dvh_data(self, value):
        # swap, keys - values
        """
            set calculated dvh data by pyplanscoring
        :param value: dvh dictionary
        """
        self._dvhs = {v['key']: value[k] for k, v in value.items()}

    @property
    def plan(self):
        return self._plan

    @property
    def dose_data(self):
        return self._dose_data

    @property
    def approval_status(self):
        txt = self._rp_dcm.ds.ApprovalStatus if 'ApprovalStatus' in self._rp_dcm.ds else ''
        return txt

    @property
    def beams(self):
        return self._plan['beams'] if 'beams' in self._plan else {}

    @property
    def dose_value_presentation(self):
        dv = self._rd_dcm.ds.DoseUnits if 'DoseUnits' in self._rd_dcm.ds else ''
        if not dv:  # pragma: no cover
            return DoseValuePresentation.Unknown
        if dv == 'GY':
            return DoseValuePresentation.Absolute
        else:  # pragma: no cover
            return DoseValuePresentation.Relative

    @property
    def total_prescribed_dose(self):
        return DoseValue(self._plan['rxdose'], DoseUnit.Gy)

    @property
    def treatment_orientation(self):
        # TODO implement DICOM IMAGE ORIENTATION TAG
        return np.array(self._dose_data['orientation'], dtype=float)

    @property
    def structures(self):
        if self.dvh_data:
            for k in self._structures.keys():
                self._structures[k]['cdvh'] = self.dvh_data[k] if k in self.dvh_data else {}
                self._structures[k]['volume'] = self.dvh_data[k]['data'][0] if k in self.dvh_data else None
            return self._structures
        else:  # pragma: no cover
            return self.get_structures()

    def get_structures(self):
        """
             Returns the structures from the planning item. Removes the need to cast to plan or plan sum.
        :param plan: PlanningItem
        :return: the referenced structure set - Dict
        """
        return self._rs_dcm.GetStructures()

    def contains_structure(self, struct_id):
        """
                Returns true if the planning item references a structure set with the input structure id AND the structure is
                contoured. Also allows a regex
                expression to match to structure id.

        :param struct_id: the structure id to match
        :return: Returns true if the planning item references a structure set with the input structure id
                AND the structure is contoured.
        :rtype: bool and Structure
        """
        structure_names = [self.structures[k]['name'] for k in self.structures.keys()]

        # normalize structure names
        struct_id = self.normalize_string(struct_id)
        norm_struc_names = [self.normalize_string(s) for s in structure_names]

        # map normalized and original strings
        structure_names_map = dict(zip(norm_struc_names, structure_names))

        matches = difflib.get_close_matches(struct_id, norm_struc_names, n=1)

        return matches, structure_names_map

    @staticmethod
    def normalize_string(s):
        for p in string.punctuation:
            s = s.replace(p, '')

        return s.upper().strip()

    def get_structure(self, struct_id):
        """
             Gets a structure (if it exists from the structure set references by the planning item
        :param struct_id:
        :return: Structure
        """
        match, names_map = self.contains_structure(struct_id)
        if match:
            original_name = names_map[match[0]]
            for k in self.structures.keys():
                if original_name == self.structures[k]['name']:
                    return self.structures[k]
        else:
            return "Structure %s not found" % struct_id

    @property
    def creation_date_time(self):
        """
            # TODO implement pydicom interface
        :return: Creation datetime
        """
        return self._plan['date']

    def get_dvh_cumulative_data(self, structure, dose_presentation, volume_presentation=None):
        """
            Get CDVH data from DICOM-RTDOSE file
        :param structure: Structure
        :param dose_presentation: DoseValuePresentation
        :param volume_presentation: VolumePresentation
        :return: DVHData
        """

        struc_dict = self.get_structure(structure)
        if struc_dict['cdvh']:
            dvh = DVHData(struc_dict['cdvh'])
            if dose_presentation == self.dose_value_presentation:
                return dvh
            if dose_presentation == DoseValuePresentation.Relative:
                dvh.to_relative_dose(self.total_prescribed_dose)
                return dvh

    def get_dose_at_volume(self, ss, volume, v_pres, d_pres):
        """
             Finds the dose at a certain volume input of a structure
        :param ss: Structure - the structure to analyze
        :param volume: the volume (cc or %)
        :param v_pres: VolumePresentation - the units of the input volume
        :param d_pres: DoseValuePresentation - the dose value presentation you want returned
        :return: DoseValue
        """

        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_dose_at_volume(volume)

    def get_dose_compliment_at_volume(self, ss, volume, v_pres, d_pres):
        """
            Return the compliment dose (coldspot) for a given volume.
            This is equivalent to taking the total volume of the
            object and subtracting the input volume

        :param ss: Structure - the structure to analyze
        :param volume: the volume to sample
        :param v_pres: VolumePresentation - the units of the input volume
        :param d_pres: DoseValuePresentation - the dose value presentation you want returned
        :return: DoseValue
        """
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_dose_compliment(volume)

    def get_volume_at_dose(self, ss, dv, v_pres):
        """
             Returns the volume of the input structure at a given input dose
        :param ss: Structure - the structure to analyze
        :param dv: DoseValue
        :param v_pres: VolumePresentation - the units of the input volume
        :return: the volume at the requested presentation
        """
        d_pres = dv.get_presentation()
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_volume_at_dose(dv, v_pres)

    def get_compliment_volume_at_dose(self, ss, dv, v_pres):
        """
             Returns the compliment volume of the input structure at a given input dose
        :param ss: Structure - the structure to analyze
        :param dv: DoseValue
        :param v_pres: VolumePresentation - the units of the input volume
        :return: the volume at the requested presentation
        """
        d_pres = dv.get_presentation()
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_compliment_volume_at_dose(dv, v_pres)

    def execute_query(self, mayo_format_query, ss):
        """
        :param pi: PlanningItem
        :param mayo_format_query: String Mayo query
        :param ss: Structure string
        :return: Query result
        """
        query = QueryExtensions()
        query.read(mayo_format_query)
        return query.run_query(query, self, ss)


class MetricType:
    MIN = "min"
    MAX = "max"
    INSIDE = "inside"


class ConstrainMetric:
    def __init__(self, structure_name, query, metric_type, target, max_score):
        self._criteria = None
        self._target = target
        self._metric_type = metric_type
        self._max_score = max_score
        self._structure_name = structure_name
        self._query = query
        self._query_result = None

    @property
    def query_result(self):
        return self._query_result

    @property
    def structure_name(self):
        return self._structure_name

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, value):
        self._query = value

    def metric_function(self, pi):
        # TODO add binary metric?
        constraint_value = pi.execute_query(self.query, self.structure_name)
        self._query_result = float(constraint_value)
        if self.metric_type == MetricType.MAX:
            score_points = [self.max_score, 0]
            return np.interp(self.query_result, self.target, score_points)
        if self.metric_type == MetricType.MIN:
            score_points = [0, self.max_score]
            # interpolation x axis should be increasing
            target = self.target[::-1]
            return np.interp(self.query_result, target, score_points)

    @property
    def metric_type(self):
        return self._metric_type

    @property
    def max_score(self):
        return self._max_score

    @max_score.setter
    def max_score(self, value):
        self._max_score = value

    @property
    def target(self):
        """
            This property holds the constraint objective and limit
        :return: [constrain_objective, constraint_limit]
        """
        return self._target

    @target.setter
    def target(self, value):
        self._target = value


class PlanEvaluation:
    def __init__(self):
        self._criteria = None

    def read(self, file_path, sheet_name):
        self._criteria = pd.read_excel(file_path, sheet_name=sheet_name)
        return self._criteria

    @property
    def criteria(self):
        return self._criteria

    @criteria.setter
    def criteria(self, value):
        self._criteria = value

    def eval_plan(self, pi):
        report_data = self.criteria.copy()
        score_res = []
        constraint_result = []
        for row in self.criteria.iterrows():
            row_val = row[1].to_dict()
            structure_name = row_val['Structure Name']
            query = row_val['Query']
            metric_type = row_val['Metric Type']
            target = [row_val['Target'], row_val['Tolerance']]
            score = row_val['Score']
            # eval metrics
            cm = ConstrainMetric(structure_name, query, metric_type, target, score)
            sc = cm.metric_function(pi)
            score_res.append(sc)
            constraint_result.append(cm.query_result)
        report_data['Result'] = constraint_result
        report_data['Raw score'] = score_res

        return report_data


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


def string_matcher(test_string, list_of_strings):
    """
        Helper method to match string to a list of strings
    :param test_string:
    :param list_of_strings:
    :return:
    """

    def normalize_string(s):
        for p in string.punctuation:
            s = s.replace(p, '')

        return s.upper().strip()

    # normalize structure names
    test_strig_normlized = normalize_string(test_string)
    list_of_strings_normalzed = [normalize_string(s) for s in list_of_strings]

    # map normalized and original strings
    structure_names_map = dict(zip(list_of_strings_normalzed, list_of_strings))
    matches = difflib.get_close_matches(test_strig_normlized, list_of_strings_normalzed, n=1)

    return matches, structure_names_map


class RTCase:
    def __init__(self, name, case_id, structures, metrics_df):
        self._case_id = case_id
        self._name = name
        self._stuctures = structures
        self._metrics = metrics_df
        self._calc_structures_names = []

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
    def structure_names(self):
        return [self.structures[k]['name'] for k in self.structures.keys()]

    @property
    def calc_structure_names(self):
        snames = list(self.metrics['Structure Name'].unique())
        # add external to calculate CI
        external = self.get_external()
        snames.append(external['name'])

        return list(set(snames))

    @property
    def external_name(self):
        external = self.get_external()
        return external['name']

    @property
    def calc_structures(self):
        list_struct_dict = [self.get_structure(name) for name in self.calc_structure_names]
        return list_struct_dict

    @property
    def case_id(self):
        return self._case_id

    def get_structure(self, structure_name, matcher=string_matcher):
        """
             Gets a structure (if it exists from the structure set reference
        :param structure_name:
        :param matcher:  Helper class to match strings
        :return: PyStructure

        """
        match, names_map = matcher(structure_name, self.structure_names)
        if match:
            original_name = names_map[match[0]]
            for k in self.structures.keys():
                if original_name == self.structures[k]['name']:
                    return self.structures[k]
        else:
            raise ValueError('Structure %s not found' % structure_name)

    def get_external(self):
        external = None
        for k, v in self.structures.items():
            if v['RTROIType'] == DICOMType.EXTERNAL:
                external = v
                break

        if external is None:
            raise ValueError('External  not found')
        else:
            return external


class PyPlanningItem:

    def __init__(self, plan_dict, rt_case, dose_3d, dvh_calculator):
        self.plan_dict = plan_dict
        self.rt_case = rt_case
        self.dose_3d = dose_3d
        self.dvh_calculator = dvh_calculator
        self._dvh_data = {}

    @property
    def dvh_data(self):
        return self._dvh_data

    @property
    def external_name(self):
        return self.rt_case.external_name

    @property
    def total_prescribed_dose(self):
        return DoseValue(self.plan_dict['rxdose'], DoseUnit.Gy)

    def calculate_dvh(self):
        if not self._dvh_data:
            self._dvh_data = self.dvh_calculator.calculate_serial(self.dose_3d)

    def get_dvh_cumulative_data(self, structure, dose_presentation, volume_presentation=None):
        """
            Get CDVH data from DICOM-RTDOSE file
        :param structure: Structure
        :param dose_presentation: DoseValuePresentation
        :param volume_presentation: VolumePresentation
        :return: DVHData
        """
        if self._dvh_data:
            struc_dict = self.rt_case.get_structure(structure)
            for k, v in self._dvh_data.items():
                if struc_dict['name'] == v['name']:
                    dvh = DVHData(v)
                    if dose_presentation == DoseValuePresentation.Absolute:
                        return dvh
                    if dose_presentation == DoseValuePresentation.Relative:
                        dvh.to_relative_dose(self.total_prescribed_dose)
                        return dvh

    def get_dose_at_volume(self, ss, volume, v_pres, d_pres):
        """
             Finds the dose at a certain volume input of a structure
        :param ss: Structure - the structure to analyze
        :param volume: the volume (cc or %)
        :param v_pres: VolumePresentation - the units of the input volume
        :param d_pres: DoseValuePresentation - the dose value presentation you want returned
        :return: DoseValue
        """

        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_dose_at_volume(volume)

    def get_dose_compliment_at_volume(self, ss, volume, v_pres, d_pres):
        """
            Return the compliment dose (coldspot) for a given volume.
            This is equivalent to taking the total volume of the
            object and subtracting the input volume

        :param ss: Structure - the structure to analyze
        :param volume: the volume to sample
        :param v_pres: VolumePresentation - the units of the input volume
        :param d_pres: DoseValuePresentation - the dose value presentation you want returned
        :return: DoseValue
        """
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_dose_compliment(volume)

    def get_volume_at_dose(self, ss, dv, v_pres):
        """
             Returns the volume of the input structure at a given input dose
        :param ss: Structure - the structure to analyze
        :param dv: DoseValue
        :param v_pres: VolumePresentation - the units of the input volume
        :return: the volume at the requested presentation
        """
        d_pres = dv.get_presentation()
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        vol_at_dose = dvh.get_volume_at_dose(dv, v_pres)
        return vol_at_dose

    def get_compliment_volume_at_dose(self, ss, dv, v_pres):
        """
             Returns the compliment volume of the input structure at a given input dose
        :param ss: Structure - the structure to analyze
        :param dv: DoseValue
        :param v_pres: VolumePresentation - the units of the input volume
        :return: the volume at the requested presentation
        """
        d_pres = dv.get_presentation()
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_compliment_volume_at_dose(dv, v_pres)

    #
    # def get_ci(self, ss, dv, v_pres):
    #     """
    #         Helper method to calculate conformity index paddick
    #     :param ss: Structure name
    #     :param dv: Dose Value
    #     :param v_pres: Volume presentation
    #     :return:
    #     """
    #     d_pres = dv.get_presentation()
    #     target_dvh_data = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
    #     # target
    #     target_vol = target_dvh_data.volume
    #     target_volume_at_dose = self.get_volume_at_dose(ss, dv, v_pres)
    #     prescription_vol_isodose = self.get_volume_at_dose(self.external_name, dv, v_pres)
    #
    #     ci = (target_volume_at_dose * target_volume_at_dose) / (target_vol * prescription_vol_isodose)
    #
    #     return float(ci)

    def get_ci(self, ss, dv, v_pres):
        """
            Helper method to calculate conformity index  RTOG
        :param ss: Structure name
        :param dv: Dose Value
        :param v_pres: Volume presentation
        :return:
        """
        d_pres = dv.get_presentation()
        target_dvh_data = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        # target
        target_vol = target_dvh_data.volume

        prescription_vol_isodose = self.get_volume_at_dose(self.external_name, dv, v_pres)

        ci = prescription_vol_isodose / target_vol

        return float(ci)

    def get_gi(self, ss, dv, v_pres):
        """
            Helper method to calculate gradient index

            Calculates the Paddick gradient index (PMID 18503356) as Paddick GI = PIV_half/PIV
            PIV_half = Prescripition isodose volume at half by prescription isodose
            PIV = Prescripition isodose volume

        :param ss: Structure name
        :param dv: Dose Value
        :param v_pres: Volume presentation
        :return:
        """

        piv = self.get_volume_at_dose(self.external_name, dv, v_pres)
        piv_half = self.get_volume_at_dose(self.external_name, dv / 2.0, v_pres)

        gi = piv_half / piv

        return float(gi)

    def execute_query(self, mayo_format_query, ss):
        """
        :param pi: PlanningItem
        :param mayo_format_query: String Mayo query
        :param ss: Structure string
        :return: Query result
        """
        query = PyQueryExtensions()
        query.read(mayo_format_query)
        return query.run_query(query, self, ss)


class PyDVHItem:

    def __init__(self, dvh_data):
        """
            Helper class to encapsulate query on single DVHs
        :param dvh_data:
        """
        self._dvh_data = dvh_data

    @property
    def dvh_data(self):
        return self._dvh_data

    @property
    def volume(self):
        """
        :return: Total Volume in cc
        """
        return float(DVHData(self.dvh_data).volume)

    def get_dvh_cumulative_data(self, structure='', dose_presentation='', volume_presentation=None):
        """
            Get CDVH data from DICOM-RTDOSE file
        :param structure: Structure
        :param dose_presentation: DoseValuePresentation
        :param volume_presentation: VolumePresentation
        :return: DVHData
        """
        if self.dvh_data:
            return DVHData(self.dvh_data)

    def get_dose_at_volume(self, ss, volume, v_pres, d_pres):
        """
             Finds the dose at a certain volume input of a structure
        :param ss: Structure - the structure to analyze
        :param volume: the volume (cc or %)
        :param v_pres: VolumePresentation - the units of the input volume
        :param d_pres: DoseValuePresentation - the dose value presentation you want returned
        :return: DoseValue
        """

        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_dose_at_volume(volume)

    def get_dose_compliment_at_volume(self, ss, volume, v_pres, d_pres):
        """
            Return the compliment dose (coldspot) for a given volume.
            This is equivalent to taking the total volume of the
            object and subtracting the input volume

        :param ss: Structure - the structure to analyze
        :param volume: the volume to sample
        :param v_pres: VolumePresentation - the units of the input volume
        :param d_pres: DoseValuePresentation - the dose value presentation you want returned
        :return: DoseValue
        """
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_dose_compliment(volume)

    def get_volume_at_dose(self, ss, dv, v_pres):
        """
             Returns the volume of the input structure at a given input dose
        :param ss: Structure - the structure to analyze
        :param dv: DoseValue
        :param v_pres: VolumePresentation - the units of the input volume
        :return: the volume at the requested presentation
        """
        d_pres = dv.get_presentation()
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        vol_at_dose = dvh.get_volume_at_dose(dv, v_pres)
        return vol_at_dose

    def get_compliment_volume_at_dose(self, ss, dv, v_pres):
        """
             Returns the compliment volume of the input structure at a given input dose
        :param ss: Structure - the structure to analyze
        :param dv: DoseValue
        :param v_pres: VolumePresentation - the units of the input volume
        :return: the volume at the requested presentation
        """
        d_pres = dv.get_presentation()
        dvh = self.get_dvh_cumulative_data(ss, d_pres, v_pres)
        return dvh.get_compliment_volume_at_dose(dv, v_pres)

    def execute_query(self, mayo_format_query, ss=''):
        """
        :param pi: PlanningItem
        :param mayo_format_query: String Mayo query
        :param ss: Structure string
        :return: Query result
        """
        query = PyQueryExtensions()
        query.read(mayo_format_query)
        return query.run_query(query, self, ss)
