"""
Classes to enumerate DVH types
Copyright (c) 2017      Victor Gabriel Leandro Alves
based on:
https://rexcardan.github.io/ESAPIX/
"""
from copy import deepcopy

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
            if self.max_dose < dv:
                return 0 * volume_unit
            else:
                if volume_unit == VolumePresentation.absolute_cm3:
                    return self.volume
                elif volume_unit == VolumePresentation.relative:
                    return 100 * volume_unit

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


class StructureBase:
    """
        class to encapsulate structure contour data
    """

    def __init__(self, structure_dict, end_cap=None):
        self._end_cap = end_cap
        self._planes = None
        self._structure_dict = None
        self._contour_spacing = None
        self._planes = None
        self._is_high_resolution = False

        # setters original structure
        self.structure = structure_dict
        self._contour_spacing = self.structure['thickness']
        self.planes = self.structure['planes']

    @property
    def structure(self):
        return self._structure_dict

    @structure.setter
    def structure(self, value):
        if isinstance(value, dict):
            if self._end_cap:
                self._structure_dict = self.get_capped_structure(value, self._end_cap)
            else:
                self._structure_dict = value
        else:
            raise ValueError("Not a structure dict type")

    @property
    def contour_spacing(self):
        """
            Returns structure contour spacing (z grid) in mm
        :return: z-grid
        :rtype: float
        """
        return self._contour_spacing

    @property
    def planes(self):
        return self._planes

    @planes.setter
    def planes(self, value):
        if isinstance(value, dict):
            self._planes = value
        else:
            raise TypeError("Not a structure planes dict")

    @property
    def name(self):
        return self.structure['name']

    @property
    def point_cloud(self):
        return self.planes2array(self.planes)

    @property
    def center_point(self):
        return np.median(self.point_cloud, axis=0)

    @property
    def color(self):
        return self.structure['color']

    @property
    def dicom_type(self):
        return self.structure['RTROIType']

    @property
    def is_high_resolution(self):
        return self._is_high_resolution

    @is_high_resolution.setter
    def is_high_resolution(self, value):
        if isinstance(value, bool):
            self._is_high_resolution = value
        else:
            raise TypeError('Is High resolution must be boolean')

    @property
    def mesh_geometry(self):
        return NotImplementedError

    @property
    def roi_number(self):
        return self.structure['id']

    @property
    def volume(self):
        return NotImplementedError

    @property
    def id(self):
        return self.structure['name']

    def to_high_resolution(self, z_grid_resolution):
        """
        :param z_grid_resolution:
        :type z_grid_resolution: float
        """
        return NotImplementedError

    def get_contours_on_image_plane(self, z):
        """
        :param z: Image z plane - string e.g. 19.50
        :return: plane dict
        """
        if isinstance(z, str):
            return self.structure['planes'].get(z)
        else:
            raise TypeError('Structure plane key should be str')

    @staticmethod
    def planes2array(s_planes):
        """
            Return all structure contour points as Point cloud array (x,y,z) points
        :param s_planes: Structure planes dict
        :return: points cloud contour points
        """
        zval = [z for z, sPlane in s_planes.items()]
        zval.sort(key=float)
        # sorted Z axis planes
        structure_planes = []
        for z in zval:
            plane_i = s_planes[z]
            for i in range(len(plane_i)):
                polygon = np.asarray(plane_i[i]['contourData'])
                # assure correct z coordinate
                polygon[:, 2] = z
                structure_planes.append(polygon)

        return np.concatenate(structure_planes)

    @staticmethod
    def get_capped_structure(structure, shift=0):
        """
            Return structure planes dict end caped
        :param structure: Structure Dict
        :param shift: end cap shift - (millimeters)
        :return: Structure dict end-caped by shift
        """

        planes_dict = structure['planes']
        # is copy needed?
        structure_dict = deepcopy(structure)
        out_Dict = deepcopy(planes_dict)
        ordered_keys = [z for z in planes_dict.keys()]
        ordered_keys.sort(key=float)
        planes = np.array(ordered_keys, dtype=float)
        start_cap = (planes[0] - shift)
        start_cap_key = '%.2f' % start_cap
        start_cap_values = planes_dict[ordered_keys[0]]

        end_cap = (planes[-1] + shift)
        end_cap_key = '%.2f' % end_cap
        end_cap_values = planes_dict[ordered_keys[-1]]

        out_Dict.pop(ordered_keys[0])
        out_Dict.pop(ordered_keys[-1])
        # adding structure caps
        out_Dict[start_cap_key] = start_cap_values
        out_Dict[end_cap_key] = end_cap_values

        structure_dict['planes'] = out_Dict

        return structure_dict


class Dose3D:
    """
        Class to encapsulate Trilinear dose interpolation

    :param master: a master Tkinter widget (opt.)

    Example::

        app = Dose3D(values, grid, unit)
    """

    def __init__(self, values, grid, unit):
        """
        :param values: 3D dose matrix
        :type values: numpy.ndarray
        :param grid: (x_grid, y_grid, z_grid)
        :rype grid: Tuple
        :param unit: Dose Unit ex, Gy, cGy or %
        :type unit: UnitQuantity
        """
        self._values = None
        self._grid = None
        self._unit = None

        # setters
        self.values = values
        self.grid = grid
        self.unit = unit

        # setup regular grid inerpolator
        x_coord = np.arange(len(self.grid[0]))
        y_coord = np.arange(len(self.grid[1]))
        z_coord = np.arange(len(self.grid[2]))

        # mapped coordinates
        self._fx = itp.interp1d(self.grid[0], x_coord, fill_value='extrapolate')
        self._fy = itp.interp1d(self.grid[1], y_coord, fill_value='extrapolate')
        self._fz = itp.interp1d(self.grid[2], z_coord, fill_value='extrapolate')

        self._fx_mm = itp.interp1d(x_coord, self.grid[0], fill_value='extrapolate')
        self._fy_mm = itp.interp1d(y_coord, self.grid[1], fill_value='extrapolate')
        self._fz_mm = itp.interp1d(z_coord, self.grid[2], fill_value='extrapolate')

        # DICOM pixel array definition
        mapped_coords = (z_coord, y_coord, x_coord)
        self._dose_interp = itp.RegularGridInterpolator(mapped_coords, self.values, bounds_error=False, fill_value=None)

        # set up private variables
        self._x_coord = x_coord
        self._y_coord = y_coord
        self._x_coord = z_coord

    # properties


    @property
    def fx(self):
        return self._fx

    @property
    def fy(self):
        return self._fy

    @property
    def fz(self):
        return self._fz

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("Values should be type ndarray")
        if len(values.shape) != 3:
            txt = 'Values should be 3D - values shape is {}'.format(values.shape)
            raise TypeError(txt)
        self._values = values

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, values):
        if not isinstance(values, tuple):
            raise TypeError("Values should be type ndarray")
        if len(values) != 3:
            txt = 'Grid must be a tuple containing (x_grid, y_grid, z_grid)'
            raise TypeError(txt)
        self._grid = values

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if not isinstance(value, pq.unitquantity.UnitQuantity):
            raise TypeError("unit should be UnitQuantity class")

        self._unit = value

    @property
    def dose_max_3d(self):
        """
        :return:  DoseValue class
        """
        return DoseValue(self.values.max(), self.unit)

    @property
    def dose_max_location(self):
        """

        :return: (x,y,z) position in mm
        """
        index_max = self.values.argmax()
        # mapped_coords = (z_coord, y_coord, x_coord)
        vec_idx = np.unravel_index(index_max, self.values.shape)

        x_mm = self._fx_mm(vec_idx[2])
        y_mm = self._fy_mm(vec_idx[1])
        z_mm = self._fz_mm(vec_idx[0])

        return np.array((x_mm, y_mm, z_mm), dtype=float)

    @property
    def x_res(self):
        return abs(self.grid[0][0] - self.grid[0][1])

    @property
    def x_size(self):
        return len(self.grid[0])

    @property
    def y_res(self):
        return abs(self.grid[1][0] - self.grid[1][1])

    @property
    def y_size(self):
        return len(self.grid[1])

    @property
    def z_res(self):
        return abs(self.grid[2][0] - self.grid[2][1])

    @property
    def z_size(self):
        return len(self.grid[2])

    def get_z_dose_plane(self, z_pos, xy_lut=None):
        """
            Gets dose slice at position z

        :param z_pos: Slice position in mm
        :type z_pos: float
        :param xy_lut: x-y lookup table
        :type xy_lut: numpy.ndarray
        :return: 2D dose matrix at position z
        :rtype: numpy.ndarray
        """
        # convert mm to index coordinate
        zi = self.fz(z_pos)

        if xy_lut:
            # return interpolated dose plane at desired lookup table
            xi, yi = self.wrap_xy_coordinates(xy_lut)
            return self._dose_interp((zi, yi, xi))
        else:
            # return full xy dose plane
            xi, yi = self.wrap_xy_coordinates((self.grid[0], self.grid[1]))
            return self._dose_interp((zi, yi, xi))

    def wrap_xy_coordinates(self, xy_lut):
        """
            Wrap 3D structure and dose grid coordinates to regular ascending grid (x,y,z)
        :rtype: array,array,array,  string array
        :param structure_planes: Structure planes dict
        :param xy_lut: look up table (XY plane)
        :return: x,y
        """
        # sparse to save memory usage
        xx, yy = np.meshgrid(xy_lut[0], xy_lut[1], indexing='xy', sparse=True)
        x_c = self.fx(xx)
        y_c = self.fy(yy)

        return x_c, y_c

    def get_dose_to_point(self, at):
        """

        :param at: [x,y,z] position
        :return: DoseValue class
        """
        if not len(at) == 3:
            raise TypeError('Should be an array of size 3. (x,y,z) positions')

        xi = self.fx(at[0])
        yi = self.fy(at[1])
        zi = self.fz(at[2])
        dv = float(self._dose_interp((zi, yi, xi)))

        return DoseValue(dv, self.unit)
        #

    def get_dose_profile(self, start, stop):
        """

        :param start:Vector (x,y,z)
        :param stop: Vector (x,y,z)
        :return: DoseProfile class
        """
        # TODO ???
        return NotImplementedError

        # def get_voxels(self, plane_index):
        #     """
        #     :param plane_index:
        #     :return:
        #     """
        #     return NotImplementedError
        #
        # def set_voxels(self, plane_index):
        #     """
        #     :param plane_index:
        #     :return:
        #     """
        #     return NotImplementedError
        #
        # def voxel_to_dose_value(self, voxel_value):
        #     return
