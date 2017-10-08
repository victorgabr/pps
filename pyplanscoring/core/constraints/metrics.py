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

from pyplanscoring.core.constraints.query import QueryExtensions
from pyplanscoring.core.constraints.types import DoseValuePresentation, DoseValue, DoseUnit, DVHData


class PlanningItem:
    """
        Planning items extensions
    """

    def __init__(self, rp_dcm, rs_dcm, rd_dcm):
        self.id = ''
        self._plan = rp_dcm.GetPlan()
        self._rp_dcm = rp_dcm
        self._rs_dcm = rs_dcm
        self._rd_dcm = rd_dcm
        self._dose_data = rd_dcm.GetDoseData()
        self._dvhs = rd_dcm.GetDVHs()
        self._structures = rs_dcm.GetStructures()

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
        # Todo not hard Coding cGy as default
        return DoseValue(self._plan['rxdose'], DoseUnit.cGy)

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
    MIN = 'min'
    MAX = 'max'


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
        constraint_value = pi.execute_query(self.query, self.structure_name)
        self._query_result = float(constraint_value)
        if self.metric_type == MetricType.MAX:
            score_points = [self.max_score, 0]
            return np.interp(self.query_result, self.target, score_points)
        if self.metric_type == MetricType.MIN:
            score_points = [0, self.max_score]
            return np.interp(self.query_result, self.target, score_points)

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

    def read(self, file_path):
        self._criteria = pd.read_excel(file_path)
        return self._criteria

    @property
    def criteria(self):
        return self._criteria

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
