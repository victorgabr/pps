"""
Classes to calculate DVH using up-sampling after reading formatted DICOM RT data.
Copyright (c) 2017      Victor Gabriel Leandro Alves
    references:
            http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.8.8.html
            http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1

"""

import time

import numpy as np
from joblib import Parallel, delayed
from numpy import ma

from core.dvhdoses import get_cdvh_numba, get_dvh_min, get_dvh_max, get_dvh_mean
from core.geometry import get_oversampled_structure, check_contour_inside, calc_area, get_contour_mask_wn
from core.types import Dose3D, StructureBase, DoseValue


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
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

    def calculate_volume(self, structure_planes, grid_delta):
        """
            Calculates the volume for the given structure.
            it considers end-capping or truncate last slice
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
            for i, contour in enumerate(contours):
                # Skip if this is the largest contour
                if not (i == largestIndex):
                    inside = check_contour_inside(contour['data'], contours[largestIndex]['data'])
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
                    s_volume = float(s_volume) + float(area) * float(grid_delta) * self._end_cap
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
                structure = get_oversampled_structure(self.structure, z_grid_resolution)
                self.structure = structure
                self.planes = structure['planes']
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
            data = np.asarray(list(map(lambda x: x[0:2], contour['contourData'])))

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
        if not isinstance(value, PyStructure):
            raise TypeError("Not a PyStruture instance")
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
        if not isinstance(value, Dose3D):
            raise TypeError('Dose instance should be type Dose3D')

        self._dose = value

    @property
    def calc_grid(self):
        return self._calc_grid

    @calc_grid.setter
    def calc_grid(self, value):

        if value is None:
            self._calc_grid = (self.dose.x_res, self.dose.y_res, self.structure.contour_spacing)
        elif len(value) != 3:
            raise ValueError('Calculation grid should be size 3, (dx, dy, dz) mm')
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
        :return: dvh
        """
        if verbose:
            print(' ----- DVH Calculation -----')
            print('Structure: {}  \n volume [cc]: {:0.1f}'.format(self.structure.name, self.structure.volume))
        max_dose = float(self.dose.dose_max_3d)
        hist = np.zeros(self.n_bins)
        volume = 0
        for z in self.structure.planes.keys():
            # Get the contours with calculated areas and the largest contour index
            contours, largest_index = self.structure.get_plane_contours_areas(z)
            # Calculate the histogram for each contour
            for j, contour in enumerate(contours):
                # Get the dose plane for the current structure contour at plane
                contour_dose_grid, ctr_dose_lut = self.get_contour_roi_grid(contour['data'], self.calc_grid)

                # get contour roi doseplane
                dose_plane = self.dose.get_z_dose_plane(float(z), ctr_dose_lut)
                m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid, contour['data'])
                h, vol = self.calculate_contour_dvh(m, dose_plane, self.n_bins, max_dose, self.calc_grid)

                # If this is the largest contour, just add to the total histogram
                if j == largest_index:
                    hist += h
                    volume += vol
                # Otherwise, determine whether to add or subtract histogram
                # depending if the contour is within the largest contour or not
                else:
                    inside = check_contour_inside(contour['data'], contours[largest_index]['data'])
                    # If the contour is inside, subtract it from the total histogram
                    if inside:
                        hist -= h
                        volume -= vol
                    # Otherwise it is outside, so add it to the total histogram
                    else:
                        hist += h
                        volume += vol

        # generate dvh curve
        return self.prepare_dvh_data(volume, hist)

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
        fc = (delta_mm + abs(grid_axis[-1] - grid_axis[0])) / (delta_mm * len(grid_axis))
        n_grid = int(round(len(grid_axis) * fc))

        up_sampled_axis, dt = np.linspace(grid_axis[0], grid_axis[-1], n_grid, retstep=True)

        # avoid inverted axis swap.  Always absolute step
        dt = abs(dt)

        return up_sampled_axis, dt

    @staticmethod
    def calculate_contour_dvh(mask, doseplane, bins, maxdose, grid_delta):
        """Calculate the differential DVH for the given contour and dose plane."""

        # Multiply the structure mask by the dose plane to get the dose mask
        mask1 = ma.array(doseplane, mask=~mask)

        # Calculate the differential dvh
        hist, edges = np.histogram(mask1.compressed(),
                                   bins=bins,
                                   range=(0, maxdose))

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

        idx = np.nonzero(chist)  # remove 0 volumes from DVH
        cdvh = chist[idx]
        # cdvh = chist

        # DICOM DVH FORMAT
        scaling = float(self.bin_size)
        units = str(self.dose.unit.symbol).upper()
        # TODO inspect nbins change
        dvh_data = {'data': list(cdvh),
                    'bins': len(cdvh),
                    'type': 'CUMULATIVE',
                    'doseunits': units,
                    'volumeunits': 'cm3',
                    'scaling': scaling,
                    'roi_number': self.structure.roi_number,
                    'min': get_dvh_min(cdvh) * scaling,
                    'max': get_dvh_max(cdvh, scaling),
                    'mean': get_dvh_mean(cdvh) * scaling}

        return dvh_data


class DVHCalculationMP:
    def __init__(self, dose, structures, grids, verbose=False):
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
            raise ValueError("PyStructure and grid lists should be equal sized")

    @property
    def dose(self):
        return self._dose

    @dose.setter
    def dose(self, value):
        """

        :param value: Dose3D instance
        :type value: Dose3D
        """
        if not isinstance(value, Dose3D):
            raise TypeError('Dose instance should be type Dose3D')

        self._dose = value

    @property
    def structures(self):
        return self._structures

    @structures.setter
    def structures(self, value):
        if isinstance(value, list):
            if not isinstance(value[0], PyStructure):
                raise TypeError("list element should be PyStructure")

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
                        raise ValueError('Calculation grid should be size 3, (dx,dy,dz)')

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

    @timeit
    def calculate_dvh_mp(self):

        if self.verbose:
            print(" ---- Starting multiprocessing -----")

        res = Parallel()(delayed(self.calculate)(s, g, self.dose, self.verbose) for s, g in self.calc_data.items())
        # map name, grid and roi_number
        cdvh = {}
        for struc_dvh in res:
            cdvh[struc_dvh['roi_number']] = struc_dvh

        if self.verbose:
            print("----- End multiprocessing -------")

        return cdvh


if __name__ == '__main__':
    from core.tests import dose_3d, structures

    # call all structures DVH without up-sampling
    structures_py = [PyStructure(v) for k, v in structures.items()]
    grids = [None] * len(structures_py)
    calc_mp = DVHCalculationMP(dose_3d, structures_py, grids, True)
    result_mp = calc_mp.calculate_dvh_mp()
