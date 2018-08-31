"""
Classes to calculate DVH using up-sampling after reading formatted DICOM RT data.
Copyright (c) 2017      Victor Gabriel Leandro Alves
    references:
            http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.8.8.html
            http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1

"""
import configparser
import time

from joblib import Parallel, delayed

import numpy as np
from numpy import ma

from .dvhdoses import get_cdvh_numba, get_dvh_max, get_dvh_mean, get_dvh_min
from .geometry import calc_area, check_contour_inside, get_contour_mask_wn, get_oversampled_structure
from .types import DoseValue, StructureBase


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('Elapsed %r  %2.2f s' % (method.__name__, (te - ts)))
        return result

    return timed


class PyStructure(StructureBase):
    def __init__(self, structure_dict, end_cap=None):
        super().__init__(structure_dict, end_cap)

    @property
    def volume(self):
        """
            Gets structure volume in cc
        :return: structure volume
        :rtype: float
        """
        return self.calculate_volume(self.planes, self.contour_spacing)

    @property
    def z_axis_delta(self):
        ordered_keys = list(self.planes.keys())
        ordered_keys.sort(key=float)
        z_delta = np.diff(np.array(list(self.planes.keys()), dtype=float))
        return z_delta

    def calculate_volume(self, structure_planes, grid_delta):
        """
            Calculates the volume for the given structure.
            it considers end-capping or truncate last slice
            obs.
                It results an approximate volume.
                Structures such as rings may not have correct volume estimated by this method


        :param structure_planes: Structure planes dict
        :type structure_planes: dict
        :param grid_delta: Voxel size in mm (dx,dy,xz)
        :type grid_delta: float
        :return: Structure volume
        :rtype: float
        """
        ordered_keys = [z for z, sPlane in structure_planes.items()]
        ordered_keys.sort(key=float)

        # Store the total volume of the structure
        s_volume = 0
        n = 0
        for z in ordered_keys:
            sPlane = structure_planes[z]
            # calculate_integrate contour areas
            contours, largestIndex = self.calculate_contour_areas(sPlane)
            # See if the rest of the contours are within the largest contour
            area = contours[largestIndex]['area']
            # TODO fix it to calculate volumes of rings
            for i, contour in enumerate(contours):
                # Skip if this is the largest contour
                if not (i == largestIndex):
                    inside = check_contour_inside(
                        contour['data'], contours[largestIndex]['data'])
                    # If the contour is inside, subtract it from the total area
                    if inside:
                        area = area - contour['area']
                    # Otherwise it is outside, so add it to the total area
                    else:
                        area = area + contour['area']

            # If the plane is the first or last slice
            # only add half of the volume, otherwise add the full slice thickness (end cap)
            if (n == 0) or (n == len(structure_planes) - 1):
                if self._end_cap:
                    s_volume = float(s_volume) + float(area) * float(
                        grid_delta) * self._end_cap
                else:
                    s_volume = float(s_volume) + float(area) * float(grid_delta)
            else:
                s_volume = float(s_volume) + float(area) * float(grid_delta)
            # Increment the current plane number
            n += 1

        # Since DICOM uses millimeters, convert from mm^3 to cm^3
        volume = s_volume / 1000.0

        return volume

    def to_high_resolution(self, z_grid_resolution):
        """
            Interpolate z-axis contours

        :param z_grid_resolution:
        :type z_grid_resolution: float
        """
        if not self.is_high_resolution:
            if not np.isclose(z_grid_resolution, self.contour_spacing):
                structure = get_oversampled_structure(self.structure,
                                                      z_grid_resolution)
                self._structure_dict = structure
                self._planes = structure['planes']
                self._contour_spacing = z_grid_resolution
                # set high resolution structure
                self.is_high_resolution = True

    def get_plane_contours_areas(self, z):
        """
            Get the contours with calculated areas and the largest contour index
        :param z: slice position
        :rtype z: str
        :return: contours, largest_index
        """
        plane = self.get_contours_on_image_plane(z)
        contours, largest_index = self.calculate_contour_areas(plane)
        return contours, largest_index

    @staticmethod
    def calculate_contour_areas(plane):
        """Calculate the area of each contour for the given plane.
           Additionally calculate_integrate and return the largest contour index.
           :param plane: Contour Plane
           :type: Dict
           :return: contour area """

        # Calculate the area for each contour in the current plane
        contours = []
        largest = 0
        largest_index = 0
        for c, contour in enumerate(plane):
            # Create arrays for the x,y coordinate pair for the triangulation
            x = contour['contourData'][:, 0]
            y = contour['contourData'][:, 1]

            c_area = calc_area(x, y)

            # Remove the z coordinate from the xyz point tuple
            data = np.asarray(
                list(map(lambda x: x[0:2], contour['contourData'])))

            # Add the contour area and points to the list of contours
            contours.append({'area': c_area, 'data': data})

            # Determine which contour is the largest
            if c_area > largest:
                largest = c_area
                largest_index = c

        return contours, largest_index


class DVHCalculation:
    """
        class to encapsulate pyplanscoring upsampling and dvh calculation
    """

    def __init__(self, structure, dose, calc_grid=None):
        """
            Class to encapsulate PyPlanScoring DVH calculation methods
        :param structure: PyStructure instance
        :type structure: PyStructure
        :param dose: Dose3D instance
        :type dose: Dose3D
        :param calc_grid: (dx,dy,dz) up-sampling grid delta in mm
        :type calc_grid: tuple
        """
        self._structure = None
        self._dose = None
        self._calc_grid = None
        # setters
        self.structure = structure
        self.dose = dose
        self.calc_grid = calc_grid

        if calc_grid is not None:
            # To high resolution z axis
            self.structure.to_high_resolution(self.calc_grid[2])

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        self._structure = value

    # Create an empty array of bins to store the histogram in Gy
    @property
    def bin_size(self):
        return DoseValue(0.01, self.dose.unit)

    @property
    def n_bins(self):
        return int(float(self.dose.dose_max_3d / self.bin_size))

    @property
    def dose(self):
        return self._dose

    @dose.setter
    def dose(self, value):
        """

        :param value: Dose3D instance
        :type value: Dose3D
        """
        # if not isinstance(value, Dose3D):
        #     raise TypeError('Dose instance should be type Dose3D')

        self._dose = value

    def get_dose_plane(self, z, ctr_dose_lut):
        """
            Wrapper method to delegate the dose plane extraction.

        :param z: Plane position in mm
        :param ctr_dose_lut: Lookup table
        :return: Dose plane
        """
        return self.dose.get_z_dose_plane(float(z), ctr_dose_lut)

    @property
    def calc_grid(self):
        return self._calc_grid

    @calc_grid.setter
    def calc_grid(self, value):

        if value is None:
            self._calc_grid = (self.dose.x_res, self.dose.y_res,
                               self.structure.contour_spacing)
        elif len(value) != 3:
            raise ValueError(
                'Calculation grid should be size 3, (dx, dy, dz) mm')
        else:
            self._calc_grid = value

    # @timeit
    def calculate(self, verbose=False):
        """
            Calculate a DVH
        :param structure: Structure obj
        :type structure: PyStructure
        :param dose: Dose3D object
        :type dose: Dose3D class
        :param grid_delta: [dx,dy,dz] in mm
        :type grid_delta: np.ndarray
        :param verbose: Print or not verbose messages
        :type verbose: bool
        :return: dvh dict
        """
        if verbose:
            print('{} volume [cc]: {:0.1f}'.format(self.structure.name,
                                                   self.structure.volume))

        max_dose = float(self.dose.dose_max_3d)
        hist = np.zeros(self.n_bins)
        volume = 0
        # integrate DVH over all planes (z axis)
        for z in self.structure.planes.keys():
            # Get the contours with calculated areas and the largest contour index
            contours, largest_index = self.structure.get_plane_contours_areas(
                z)

            # Calculate the histogram for each contour
            hist_plane, volume_plane = self.calculate_plane_dvh(
                contours, max_dose, z)

            hist += hist_plane
            volume += volume_plane

        # generate dvh dictionary
        return self.prepare_dvh_data(volume, hist)

    def calculate_plane_dvh(self, contours, max_dose, z):

        # Get Grid and Dose plane for the largest contour
        plane_contour_points = np.vstack([c['data'] for c in contours])
        contour_dose_grid, ctr_dose_lut = self.get_contour_roi_grid(
            plane_contour_points, self.calc_grid)

        dose_plane = self.get_dose_plane(z, ctr_dose_lut)

        # pre allocate dose grid matrix
        grid = np.zeros(
            (len(ctr_dose_lut[1]), len(ctr_dose_lut[0])), dtype=np.uint8)
        for _, contour in enumerate(contours):
            # rasterized dose plane inside contour
            m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid,
                                    contour['data'])

            # using exclusive or operator to remove holes from each rasterized contour
            grid = np.logical_xor(m.astype(np.uint8), grid).astype(np.bool)

        hist_plane, volume_plane = self.calculate_contour_dvh(
            grid, dose_plane, self.n_bins, max_dose, self.calc_grid)

        return hist_plane, volume_plane

    def get_dose_grid_3d(self, grid_3d, delta_mm=(2, 2, 2)):
        """
         Generate a 3d mesh grid to create a polygon mask in dose coordinates
         adapted from Stack Overflow Answer from Joe Kington:
         http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/3655582
        Create vertex coordinates for each grid cell

        :param grid_3d: X,Y,Z grid coordinates (mm)
        :param delta_mm: Desired grid delta (dx,dy,dz) mm
        :return: dose_grid_points, up_dose_lut, grid_delta
        """
        xi = grid_3d[0]
        yi = grid_3d[1]
        zi = grid_3d[2]

        x_lut, x_delta = self.get_axis_grid(delta_mm[0], xi)
        y_lut, y_delta = self.get_axis_grid(delta_mm[1], yi)
        z_lut, z_delta = self.get_axis_grid(delta_mm[2], zi)

        xg, yg = np.meshgrid(x_lut, y_lut)
        xf, yf = xg.flatten(), yg.flatten()
        dose_grid_points = np.vstack((xf, yf)).T

        up_dose_lut = [x_lut, y_lut, z_lut]

        spacing = [x_delta, x_delta, z_delta]

        return dose_grid_points, up_dose_lut, spacing

    def get_contour_roi_grid(self, contour_points, delta_mm, fac=1):
        """
            Returns a boundary contour ROI/GRID
        :param contour_points:
        :param delta_mm: (dx,dy) in mm
        :param fac:  number of margin delta of ROI
        :return: contour_grid and contour lookup table (mesh)
        """
        x = contour_points[:, 0]
        y = contour_points[:, 1]
        x_min = x.min() - delta_mm[0] * fac
        x_max = x.max() + delta_mm[0] * fac
        y_min = y.min() - delta_mm[1] * fac
        y_max = y.max() + delta_mm[1] * fac
        x_lut, x_delta = self.get_axis_grid(delta_mm[0], [x_min, x_max])
        y_lut, y_delta = self.get_axis_grid(delta_mm[1], [y_min, y_max])
        xg, yg = np.meshgrid(x_lut, y_lut)
        xf, yf = xg.flatten(), yg.flatten()
        contour_grid = np.vstack((xf, yf)).T
        contour_lut = [x_lut, y_lut]

        return contour_grid, contour_lut

    @staticmethod
    def get_axis_grid(delta_mm, grid_axis):
        """
            Returns the up sampled axis by given resolution in mm

        :param delta_mm: desired resolution
        :param grid_axis: x,y,x axis from LUT
        :return: up sampled axis and delta grid
        """
        fc = (delta_mm + abs(grid_axis[-1] - grid_axis[0])) / (
            delta_mm * len(grid_axis))
        n_grid = int(round(len(grid_axis) * fc))

        up_sampled_axis, dt = np.linspace(
            grid_axis[0], grid_axis[-1], n_grid, retstep=True)

        # avoid inverted axis swap.  Always absolute step
        dt = abs(dt)

        return up_sampled_axis, dt

    @staticmethod
    def calculate_contour_dvh(mask, doseplane, bins, maxdose, grid_delta):
        """Calculate the differential DVH for the given contour and dose plane."""

        # Multiply the structure mask by the dose plane to get the dose mask
        mask1 = ma.array(doseplane, mask=~mask)

        # Calculate the differential dvh
        hist, edges = np.histogram(
            mask1.compressed(), bins=bins, range=(0, maxdose))

        # Calculate the volume for the contour for the given dose plane
        vol = np.sum(hist) * grid_delta[0] * grid_delta[1] * grid_delta[2]

        return hist, vol

    def prepare_dvh_data(self, volume, hist):
        # TODO prepare it to be like DICOM RD dvh data
        # TODO create a serialised DVH storage format
        # volume units are given in cm^3
        volume /= 1000
        # Rescale the histogram to reflect the total volume
        hist = hist * volume / sum(hist)
        chist = get_cdvh_numba(hist)

        # todo clean up negative volumes or just enforce structures inside external ?
        # chist[chist < 0] = 0.
        cdvh = np.trim_zeros(chist, trim='b')

        # cdvh = chist

        # DICOM DVH FORMAT
        scaling = float(self.bin_size)
        units = str(self.dose.unit.symbol).upper()
        # TODO inspect nbins change
        # TODO round data to 3 decimal places ?
        # dvh_data = {
        #     'data': list(cdvh),  # round 3 decimal
        #     'bins': len(cdvh),
        #     'type': 'CUMULATIVE',
        #     'doseunits': units,
        #     'volumeunits': 'cm3',
        #     'scaling': scaling,
        #     'roi_number': self.structure.roi_number,
        #     'name': self.structure.name,
        #     'min': get_dvh_min(cdvh) * scaling,
        #     'max': get_dvh_max(cdvh, scaling) * scaling,
        #     'mean': get_dvh_mean(cdvh) * scaling
        # }

        dvh_data = {
            'data': list(np.round(cdvh, 2)),  # round 3 decimal
            'bins': len(cdvh),
            'type': 'CUMULATIVE',
            'doseunits': units,
            'volumeunits': 'cm3',
            'scaling': scaling,
            'roi_number': self.structure.roi_number,
            'name': self.structure.name,
            'min': np.round(get_dvh_min(cdvh) * scaling, 2),
            'max': np.round(get_dvh_max(cdvh, scaling) * scaling, 2),
            'mean': np.round(get_dvh_mean(cdvh) * scaling, 2)
        }

        return dvh_data


class DVHCalculationMP:
    def __init__(self, dose, structures, grids, verbose=True):
        self._grids = None
        self._dose = None
        self._structures = None
        self.dvhs = {}
        self.verbose = verbose
        # setters
        self.structures = structures
        self.dose = dose
        self.grids = grids
        # sanity check
        if not len(self.structures) == len(self.grids):
            raise ValueError(
                "PyStructure and grid lists should be equal sized")

    @property
    def dose(self):
        return self._dose

    @dose.setter
    def dose(self, value):
        """

        :param value: Dose3D instance
        :type value: Dose3D
        """
        # if not isinstance(value, Dose3D):
        #     raise TypeError('Dose instance should be type Dose3D')

        self._dose = value

    @property
    def structures(self):
        return self._structures

    @structures.setter
    def structures(self, value):
        if isinstance(value, list):
            self._structures = value
        else:
            raise TypeError("Argument should be a list.")

    @property
    def grids(self):
        return self._grids

    @grids.setter
    def grids(self, value):
        if isinstance(value, list):
            for g in value:
                if g is not None:
                    if len(g) != 3:
                        raise ValueError(
                            'Calculation grid should be size 3, (dx,dy,dz)')

            self._grids = value
        else:
            raise TypeError("Argument grid should be a list.")

    @property
    def calc_data(self):
        return dict(zip(self.structures, self.grids))

    @staticmethod
    def calculate(structure, grid, dose, verbose):
        """
            Calculate DVH per structure

        :param structure: PyStructure instance
        :type structure: PyStructure
        :param grid: grid delta
        :type grid: tuple
        :param dose: Dose3D instance
        :type dose: Dose3D
        :param verbose: Prints message to terminal
        :type verbose: bool
        :return: DVH calculated
        :rtype: dict
        """

        dvh_calc = DVHCalculation(structure, dose, calc_grid=grid)
        res = dvh_calc.calculate(verbose)
        # map thread/process result to its roi number
        res['roi_number'] = structure.roi_number
        return res

    # @timeit
    def calculate_dvh_mp(self):

        if self.verbose:
            print(" ---- Starting multiprocessing -----")

        res = Parallel(n_jobs=-1)(delayed(self.calculate)(s, g, self.dose, self.verbose)
                         for s, g in self.calc_data.items())
        # map name, grid and roi_number
        cdvh = {}
        for struc_dvh in res:
            cdvh[struc_dvh['roi_number']] = struc_dvh

        if self.verbose:
            print("----- End multiprocessing -------")

        return cdvh


class DVHCalculator:
    def __init__(self, rt_case=None, calculation_options=None):
        self._rt_case = None
        self._calculation_options = None
        self._dvh_data = {}
        self.iteration = 0

        # setters
        if rt_case is not None:
            self.rt_case = rt_case

        if calculation_options is not None:
            self.calculation_options = calculation_options

    @property
    def rt_case(self):
        return self._rt_case

    @rt_case.setter
    def rt_case(self, value):
        self._rt_case = value

    @property
    def calculation_options(self):
        return self._calculation_options

    @calculation_options.setter
    def calculation_options(self, value):
        self._calculation_options = value

    @property
    def dvh_data(self):
        return self._dvh_data

    @property
    def calculation_setup(self):
        """
            Return wrapped structures and calculation grids
        :return: structures_py, grids
        """
        # setup PysStructures and calculation grid
        structures_py = [
            PyStructure(s, self.end_cap) for s in self.rt_case.calc_structures
        ]
        grids = self.get_grid_array(structures_py)
        return structures_py, grids

    @property
    def voxel_size(self):
        return tuple([self.calculation_options['voxel_size']] * 3)

    @property
    def end_cap(self):
        return self.calculation_options['end_cap']

    @property
    def max_vol_upsampling(self):
        """
            Return maximum volume to be upsampled
        :return: Threshold volume in cc
        """
        return self.calculation_options['maximum_upsampled_volume_cc']

    @property
    def up_sampling(self):
        return self.calculation_options['up_sampling']

    def get_grid_array(self, structures_py):
        grids = []
        for s in structures_py:
            # check if upsampling
            if self.up_sampling:
                # Check if it is contiguous and monotonic z spacing
                # TODO fix bug when end capping
                if len(np.unique(s.z_axis_delta)
                       ) == 1 and s.volume < self.max_vol_upsampling:
                    grids.append(self.voxel_size)
                else:
                    grids.append(None)
            else:
                grids.append(None)

        return grids

    def calculate_mp(self, dose_3d):
        """
            Recieves a dose3D object, calculate DVH's and return dvh dict
        :param dose_3d:
        :return:
        """
        structures_py, grids = self.calculation_setup
        calc_mp = DVHCalculationMP(dose_3d, structures_py, grids)
        self._dvh_data = calc_mp.calculate_dvh_mp()
        return dict(self._dvh_data)

    @timeit
    def calculate_all(self, dose_3d):
        structures_py, grids = self.calculation_setup

        cdvh = {}
        for structure, grid in zip(structures_py, grids):
            dvh_calc = DVHCalculation(structure, dose_3d, calc_grid=grid)
            res = dvh_calc.calculate(True)
            # map thread/process result to its roi number
            res['roi_number'] = structure.roi_number
            cdvh[structure.roi_number] = res
            self.iteration += 1

        self._dvh_data = cdvh

        return cdvh


def get_calculation_options(ini_file_path):
    """
        Helper method to read app *.ini file
    :param ini_file_path:
    :return:
    """

    # Get calculation defaults
    config = configparser.ConfigParser()
    config.read(ini_file_path)
    calculation_options = dict()
    calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
    calculation_options['use_tps_dvh'] = config.getboolean(
        'DEFAULT', 'use_tps_dvh')
    calculation_options['use_tps_structures'] = config.getboolean(
        'DEFAULT', 'use_tps_structures')
    calculation_options['up_sampling'] = config.getboolean(
        'DEFAULT', 'up_sampling')
    calculation_options['maximum_upsampled_volume_cc'] = config.getfloat(
        'DEFAULT', 'maximum_upsampled_volume_cc')
    calculation_options['voxel_size'] = config.getfloat(
        'DEFAULT', 'voxel_size')
    calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
    calculation_options['save_dvh_figure'] = config.getboolean(
        'DEFAULT', 'save_dvh_figure')
    calculation_options['save_dvh_data'] = config.getboolean(
        'DEFAULT', 'save_dvh_data')
    calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']

    return calculation_options
