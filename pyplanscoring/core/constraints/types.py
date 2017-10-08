"""
Classes to enumerate DVH types
based on:
https://rexcardan.github.io/ESAPIX/
"""

import numpy as np
import quantities as pq
from scipy import interpolate as itp


class DoseUnit:
    Gy = pq.Gy
    cGy = pq.UnitQuantity('cGy',
                          pq.centi * Gy,
                          symbol='cGy',
                          aliases=['cGy'])
    Percent = pq.percent
    Unknown = pq.dimensionless


class QuantityRegex:
    @staticmethod
    def string_to_quantity(arg):
        switch = {'CC': VolumePresentation.absolute_cm3,
                  'CM3': VolumePresentation.absolute_cm3,
                  'CGY': DoseUnit.cGy,
                  'GY': DoseUnit.Gy,
                  '%': DoseUnit.Percent,
                  'NA': DoseUnit.Unknown,
                  '': DoseUnit.Unknown}
        return switch.get(arg.upper(), DoseUnit.Unknown)


class QueryType:
    VOLUME_AT_DOSE = 0
    COMPLIMENT_VOLUME = 1
    DOSE_AT_VOLUME = 2
    DOSE_COMPLIMENT = 3
    MEAN_DOSE = 4
    MIN_DOSE = 5
    MAX_DOSE = 6
    CI = 7
    HI = 8


class Units:
    CC = 0
    PERC = 1
    GY = 2
    CGY = 3
    NA = 4


class DoseValuePresentation:
    Relative = 0
    Absolute = 1
    Unknown = 2


class Discriminator:
    LESS_THAN = 0
    LESS_THAN_OR_EQUAL = 1
    GREATER_THAN = 2
    GREATHER_THAN_OR_EQUAL = 3
    EQUAL = 4


# class VolumePresentation:
#     relative = 0
#     absolute_cm3 = 1

class VolumePresentation:
    relative = pq.percent
    absolute_cm3 = pq.cubic_centimeter
    Unknown = pq.dimensionless


class PriorityType:
    IDEAL = 0
    ACCEPTABLE = 1
    MINOR_DEVIATION = 2
    MAJOR_DEVIATION = 3
    GOAL = 4
    OPTIONAL = 5
    REPORT = 6
    PRIORITY_1 = 7
    PRIORITY_2 = 8


class ResultType:
    PASSED = 0
    ACTION_LEVEL_1 = 1
    ACTION_LEVEL_2 = 2
    ACTION_LEVEL_3 = 3
    NOT_APPLICABLE = 4
    NOT_APPLICABLE_MISSING_STRUCTURE = 5
    NOT_APPLICABLE_MISSING_DOSE = 6
    INCONCLUSIVE = 7


class TargetStat:
    CONFORMITY_INDEX_RTOG = 0
    CONFORMITY_INDEX_PADDICK = 1
    HOMOGENEITY_INDEX = 2
    VOXEL_BASED_HOMOGENEITY_INDEX = 3


class PatientOrientation:
    NoOrientation = 0
    HeadFirstSupine = 1
    HeadFirstProne = 2
    HeadFirstDecubitusRight = 3
    HeadFirstDecubitusLeft = 4
    FeetFirstSupine = 5
    FeetFirstProne = 6
    FeetFirstDecubitusRight = 7
    FeetFirstDecubitusLeft = 8
    Sitting = 9


class DICOMType:
    """
    Class that holds constant strings from the Eclipse Scripting API
    """
    PTV = "PTV"
    GTV = "GTV"
    CTV = "CTV"
    DOSE_REGION = "DOSE_REGION"
    NONE = ""
    CONSTRAST_AGENT = "CONSTRAST_AGENT"
    CAVITY = "CAVITY"
    AVOIDANCE = "AVOIDANCE"
    CONTROL = "CONTROL"
    FIXATION = "FIXATION"
    IRRAD_VOLUME = "IRRAD_VOLUME"
    ORGAN = "ORGAN"
    TREATED_VOLUME = "TREATED_VOLUME"
    EXTERNAL = "EXTERNAL"


class DVHData:
    def __init__(self, dvh):
        self._dose_format = None
        self._volume_format = None
        self.dvh = dvh
        self._volume = dvh['data'][0]
        self._dose_units = QuantityRegex.string_to_quantity(dvh['doseunits'])
        self._volume_units = QuantityRegex.string_to_quantity(dvh['volumeunits'])
        # set data according to the given units
        self._dose_axis_bkp = np.arange(len(dvh['data']) + 1) * dvh['scaling']
        self._dose_axis = np.arange(len(dvh['data']) + 1) * dvh['scaling'] * self._dose_units
        self._volume_axis = np.append(dvh['data'], 0) * self._volume_units
        self._curve_data = dvh['data']
        self._min_dose = dvh['min']
        self._mean_dose = dvh['mean']
        self._max_dose = dvh['max']
        self._bin_width = dvh['scaling']
        self.set_interpolation_data()
        self.set_volume_focused_data()

    def set_interpolation_data(self):
        # setting constrain interpolation functions
        self.fv = itp.interp1d(self.dose_axis, self.volume_pp, fill_value='extrapolate')  # pp
        self.fv_cc = itp.interp1d(self.dose_axis, self.volume_cc, fill_value='extrapolate')  # cc
        self.fd = itp.interp1d(self.volume_pp, self.dose_axis, fill_value='extrapolate')  # pp
        self.fd_cc = itp.interp1d(self.volume_cc, self.dose_axis, fill_value='extrapolate')  # cc

    def set_volume_focused_data(self):
        """
            Volume-Focused Format
            The use of a volume-focused DVH format facilitated the construction of a statistical representation
            of DVH curves and ensures the ability to represent DVH curves independently of Max[Gy] with a small,
            fixed set of points.

             ref. http://www.sciencedirect.com/science/article/pii/S2452109417300611
        """
        s0 = [0, 0.5]
        s1 = np.arange(1, 5, 1)
        s2 = np.arange(5, 96, 5)
        s3 = np.arange(96, 100, 1)
        s4 = [99.5, 100.0]
        volume_focused_format = np.concatenate((s0, s1, s2, s3, s4))[::-1]
        dose_focused_format = self.fd(volume_focused_format)
        self._volume_format = volume_focused_format
        self._dose_format = dose_focused_format

    @property
    def volume_focused_format(self):
        return self._volume_format

    @property
    def dose_focused_format(self):
        return self._dose_format

    @property
    def dose_axis(self):
        return self._dose_axis

    @dose_axis.setter
    def dose_axis(self, value):
        self._dose_axis = value

    @property
    def dose_unit(self):
        return self._dose_units

    @property
    def volume_cc(self):
        return self._volume_axis

    @property
    def curve_data(self):
        """
            implement DVHPoint[] from pyplanscoring results
        :return: Curve data array
        """
        return self._curve_data

    @property
    def volume_pp(self):
        return self.convert_to_relative_volume(self._volume_axis)

    @property
    def max_dose(self):
        """
        :return: class DoseValue max_dose
        """
        return DoseValue(self._max_dose, self.dose_unit)

    @property
    def mean_dose(self):
        """
        :return: class DoseValue mean_dose
        """
        return DoseValue(self._mean_dose, self.dose_unit)

    @property
    def min_dose(self):
        return DoseValue(self._min_dose, self.dose_unit)

    @property
    def bin_width(self):
        return self._bin_width

    @property
    def volume(self):
        return self._volume * self.volume_unit

    @property
    def volume_unit(self):
        return self._volume_units

    def get_volume_at_dose(self, dv, volume_unit):
        """
            Gets the volume that recieves the input dose
        :param volume_unit: VolumePresentation
        :param dvh: DVHPoints object - the dose volume histogram for this structure
        :param dv: DoseValue object - the dose value to sample the curve
        :return: volume_at_dose point
        """
        dose_presentation = dv.get_presentation()
        if dose_presentation == DoseValuePresentation.Absolute:
            if dv.unit != self.dose_unit:
                # rescale cGy to Gy or cGy to cGy...same unit result
                dv = dv.rescale(self.dose_unit)

        # If the max dose is less than the queried dose, then there is no volume at the queried dose (out of range)
        # If the min dose is greater than the queried dose, then 100% of the volume is at the queried dose
        if self.max_dose < dv or dv < self.min_dose:
            return 0 * self.volume_unit if self.max_dose < dv else self.volume

        if volume_unit == VolumePresentation.absolute_cm3:
            return float(self.fv_cc(dv.value)) * VolumePresentation.absolute_cm3
        elif volume_unit == VolumePresentation.relative:
            return float(self.fv(dv.value)) * VolumePresentation.relative

    def get_compliment_volume_at_dose(self, dv, volume_unit):
        """
            Gets the compliment volume (volume about a certain dose point) for the structure dvh
        :param volume_unit: VolumePresentation
        :param dv: DoseValue object - the dose value to sample the curve
        :return: volume_at_dose point
        """

        max_vol = 0
        if volume_unit == VolumePresentation.absolute_cm3:
            max_vol = self.volume_cc.max()
        elif volume_unit == VolumePresentation.relative:
            max_vol = self.volume_pp.max()

        normal_volume = self.get_volume_at_dose(dv, volume_unit)
        compliment_volume_at_dose = max_vol - normal_volume
        return compliment_volume_at_dose

    def get_dose_at_volume(self, volume):
        """
             Gets the dose value at the specified volume for the curve
        :param dvh: DVHPoints object - the dose volume histogram for this structure
        :param volume: the volume in the same units as the DVH curve
        :return: DoseValue object
        """
        if volume.units == VolumePresentation.relative:
            min_vol = self.volume_pp.min()
            max_vol = self.volume_pp.max()
            # Check for max point dose scenario
            if volume <= min_vol:
                return self.max_dose

            # Check dose to total volume scenario (min dose)
            if np.isclose(float(volume), float(max_vol)):
                return self.min_dose

            # Overvolume scenario, undefined
            if volume > max_vol:
                return None

            return DoseValue(float(self.fd(volume)), self.dose_unit)
        elif volume.units == VolumePresentation.absolute_cm3:
            min_vol = self.volume_cc.min()
            max_vol = self.volume_cc.max()
            # Check for max point dose scenario
            if volume <= min_vol:
                return self.max_dose

            # Check dose to total volume scenario (min dose)
            if np.isclose(float(volume), float(max_vol)):
                return self.min_dose

            # Overvolume scenario, undefined
            if volume > max_vol:
                return None

            return DoseValue(float(self.fd_cc(volume)), self.dose_unit)
        else:
            return ValueError('Wrong argument - Unknown volume units')

    def get_dose_compliment(self, volume):
        """
              Gets the compliment dose for the specified volume (the cold spot).
              Calculated by taking the total volume and subtracting the input volume.
        :param dvh: DVHPoints object - the dose volume histogram for this structure
        :param volume: the volume in the same units as the DVH curve
        :return: DoseValue object
        """

        if volume.units == self.volume_unit:
            max_vol = self.volume_cc.max()
            vol_of_interest = max_vol - volume
            return self.get_dose_at_volume(vol_of_interest)

        elif volume.units == VolumePresentation.relative:
            max_vol = self.volume_pp.max()
            vol_of_interest_rel = max_vol - volume
            return self.get_dose_at_volume(vol_of_interest_rel)

    @staticmethod
    def convert_to_relative_volume(curve_data):
        """
            If appropriate, converts the DVH curve into relative volume points instead of absolute volume
        :param curve_data: the input DVH
        :return: the dvh with relative volume points
        """
        rel_vol = curve_data / curve_data.max() * 100

        return rel_vol * VolumePresentation.relative

    def to_relative_dose(self, scaling_point):
        """
            If appropriate, converts the DVH curve into relative dose points instead of absolute dose
        :param dvh: the input DVH
        :param scaling_point: DoseValue object - the dose value which represents 100%, all doses will be scaled in reference to this
        :return: the dvh with relative dose points
        """
        # TODO add result depending on units desired
        dose_presentation = scaling_point.get_presentation()
        if dose_presentation == DoseValuePresentation.Absolute:
            # rescale to get same unit result
            if scaling_point.unit != self.dose_unit:
                scaling_point = scaling_point.rescale(self.dose_unit)

        dose_axis_norm = self._dose_axis_bkp * (100 / scaling_point.value)
        self._min_dose *= (100 / scaling_point.value)
        self._max_dose *= (100 / scaling_point.value)
        self._mean_dose *= (100 / scaling_point.value)
        self._dose_units = DoseUnit.Percent
        self.dose_axis = dose_axis_norm * self._dose_units
        self.set_interpolation_data()

    @staticmethod
    def convert_to_relative_dose(dvh, scaling_point):
        """
            If appropriate, converts the DVH curve into relative dose points instead of absolute dose
        :param dvh: the input DVH
        :param scaling_point: the dose value which represents 100%, all doses will be scaled in reference to this
        :return: the dvh with relative dose points
        """
        return NotImplementedError

    def merge_dvhs(self, dvhs):
        """
            Merges DVHData from multiple structures into one DVH by summing the volumes at each dose value
        :param dvhs: the multiple dvh curves to merge
        :return: the combined dvh from multiple structures
        """
        return NotImplementedError


class DoseValue:
    def __init__(self, dose_value, unit):
        """
            Class do encapsulate dose values and its quantities
            Default: cGy
        :param dose_value: Dose value
        :param unit: DoseUnit
        """
        self._value = dose_value
        self._dose = dose_value * unit
        self._unit = unit

    def get_presentation(self):
        if self.unit.name in ['gray', 'cGy']:
            return DoseValuePresentation.Absolute

        elif self.unit.name == 'percent':
            return DoseValuePresentation.Relative
        else:
            return DoseValuePresentation.Unknown

    @property
    def value(self):
        return self._value

    @property
    def dose(self):
        return self._dose

    @property
    def unit(self):
        return self._unit

    def rescale(self, dose_unit):
        """
        :param dose_unit: DoseUnit
        :return: Rescaled Dose
        """
        val = self.dose.rescale(dose_unit)
        return DoseValue(float(val), dose_unit)

    def get_dose(self, dose_unit):
        return self.rescale(dose_unit)

    def __float__(self):
        return self.value

    def __str__(self):
        dose_unit = self.unit.symbol
        dose = self.value
        dose_txt = ('%1.3f' % dose).rstrip('0').rstrip('.')
        return '%s %s' % (dose_txt, dose_unit)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        res = self.value * self.unit + other.value * other.unit
        return DoseValue(float(res.rescale(self.unit)), self.unit)

    def __sub__(self, other):
        res = self.value * self.unit - other.value * other.unit
        return DoseValue(float(res.rescale(self.unit)), self.unit)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return DoseValue(self.value * other, self.unit)
        if isinstance(other, DoseValue):
            a = self.dose.rescale(self.unit)
            b = other.dose.rescale(self.unit)
            c = a * b
            return c

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            if other != 0:
                return DoseValue(self.value / other, self.unit)
            else:
                raise ValueError('Division by zero')
        if isinstance(other, DoseValue):
            a = self.value * self.unit
            b = other.value * other.unit
            if b != 0:
                res = a / b
            else:
                raise ValueError('Division by zero dose')
            return DoseValue(float(res.rescale(pq.dimensionless)), pq.dimensionless)

    def __lt__(self, other):
        return self.dose < other.dose

    def __le__(self, other):
        return self.dose <= other.dose

    def __eq__(self, other):
        other = other.rescale(self.unit)
        return np.isclose(self.value, other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        return self.dose >= other.dose

    def __gt__(self, other):
        return self.dose >= other.dose

# TODO encapsulate 3D trilinear interpolation from RD_FILE
# class Dose3D:
#     @property
#     def dose_max_3d(self):
#         """
#         :return:  DoseValue class
#         """
#         return 'DoseValue'
#
#     @property
#     def dose_max_3d_location(self):
#         return 'VVector'
#
#     @property
#     def origin(self):
#         return
#
#     @property
#     def series(self):
#         return 'Series'
#
#     @property
#     def x_direction(self):
#         return 'VVector'
#
#     @property
#     def x_res(self):
#         return 'resolution in mm'
#
#     @property
#     def x_size(self):
#         return 'int num pixels x'
#
#     @property
#     def y_direction(self):
#         return 'VVector'
#
#     @property
#     def y_res(self):
#         return 'resolution in mm'
#
#     @property
#     def y_size(self):
#         return 'int num pixels y'
#
#     @property
#     def z_direction(self):
#         return 'VVector'
#
#     @property
#     def z_res(self):
#         return 'resolution in mm'
#
#     @property
#     def z_size(self):
#         return 'int num pixels x'
#
#     def get_dose_profile(self, start, stop):
#         """
#
#         :param start: VVector
#         :param stop: VVector
#         :return: DoseProfile class
#         """
#         return NotImplementedError
#
#     def get_dose_to_point(self, at):
#         """
#
#         :param at: VVector
#         :return: DoseValue class
#         """
#         return NotImplementedError
#
#     def get_voxels(self, plane_index):
#         """
#         :param plane_index:
#         :return:
#         """
#         return NotImplementedError
#
#     def set_voxels(self, plane_index):
#         """
#         :param plane_index:
#         :return:
#         """
#         return NotImplementedError
#
#     def voxel_to_dose_value(self, voxel_value):
#         return
#
# TODO encapsulate Structure in a class
# class StructureBase:
#     """
#         class to encapsulate structure contour data
#     """
#
#     def __init__(self, structure_dict):
#         self.structure_dict = structure_dict
#         self._is_high_resolution = False
#
#     @property
#     def point_cloud(self):
#         return self.planes2array(self.structure_dict['planes'])
#
#     @property
#     def center_point(self):
#         return np.median(self.point_cloud, axis=0)
#
#     @property
#     def color(self):
#         return self.structure_dict['color']
#
#     @property
#     def dicom_type(self):
#         return self.structure_dict['RTROIType']
#
#     @property
#     def is_high_resolution(self):
#         return self._is_high_resolution
#
#     @is_high_resolution.setter
#     def is_high_resolution(self, value):
#         self._is_high_resolution = value
#
#     @property
#     def mesh_geometry(self):
#         return NotImplementedError
#
#     @property
#     def roi_number(self):
#         return self.structure_dict['id']
#
#     @property
#     def volume(self):
#         return NotImplementedError
#
#     @property
#     def id(self):
#         return self.structure_dict['name']
#
#     def add_contour_on_image_plane(self, contour, z):
#         """
#
#         :param contour: VVector[]
#         :param z: plane z
#         """
#         return NotImplementedError
#
#     def to_high_resolution(self, z_grid_resolution):
#         # TODO implement z axis upsampling
#
#         return NotImplementedError
#
#     def get_contours_on_image_plane(self, z):
#         """
#         :param z: Image z plane - string e.g. 19.50
#         :return: VVector[][]
#         """
#         return self.structure_dict['planes'].get(z)
#
#     @staticmethod
#     def planes2array(s_planes):
#         """
#             Return all structure contour points as Point cloud array (x,y,z) points
#         :param s_planes: Structure planes dict
#         :return: points cloud contour points
#         """
#         zval = [z for z, sPlane in s_planes.items()]
#         zval.sort(key=float)
#         # sorted Z axis planes
#         structure_planes = []
#         zplanes = []
#         for z in zval:
#             plane_i = s_planes[z]
#             for i in range(len(plane_i)):
#                 polygon = np.asarray(plane_i[i]['contourData'])
#                 structure_planes.append(polygon)
#
#         return np.concatenate(structure_planes)
#
#     @staticmethod
#     def calculate_volume(structure_planes, grid_delta):
#         """Calculates the volume for the given structure.
#         :rtype: float
#         :param structure_planes: Structure planes dict
#         :param grid_delta: Voxel size (dx,dy,xz)
#         :return: Structure volume
#         """
#
#         return NotImplementedError
#
#     @staticmethod
#     def calculate_contour_areas(plane):
#         """Calculate the area of each contour for the given plane.
#            Additionally calculate_integrate and return the largest contour index."""
#
#         return NotImplementedError
