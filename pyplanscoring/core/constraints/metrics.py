"""
Classes to DVH metrics

Author: Victor Alves

based on:
https://rexcardan.github.io/ESAPIX/

"""
import difflib

import numpy as np

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
        if self._dvhs:
            for k in self._structures.keys():
                self._structures[k]['cdvh'] = self._dvhs[k] if k in self._dvhs else {}
                self._structures[k]['volume'] = self._dvhs[k]['data'][0] if k in self._dvhs else None
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
        matches = difflib.get_close_matches(struct_id, structure_names, n=1)

        return True if matches else False

    def get_structure(self, struct_id):
        """
             Gets a structure (if it exists from the structure set references by the planning item
        :param struct_id:
        :return: Structure
        """
        if self.contains_structure(struct_id):
            for k in self.structures.keys():
                if struct_id == self.structures[k]['name']:
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
