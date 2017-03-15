from __future__ import division

import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from joblib import Parallel
from joblib import delayed

from pyplanscoring.dev.geometry import get_contour_mask_wn, get_dose_grid_3d, \
    get_axis_grid, get_dose_grid, \
    get_interpolated_structure_planes, contour_rasterization_numba, check_contour_inside, \
    get_contour_roi_grid, wrap_xy_coordinates, wrap_z_coordinates
from pyplanscoring.dicomparser import ScoringDicomParser
from pyplanscoring.dvhcalc import calculate_contour_areas_numba, save
from pyplanscoring.dvhdoses import get_dvh_min, get_dvh_max, get_dvh_mean, get_cdvh_numba

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})
# np.set_printoptions(formatter={'str_kind': str_formatter})

# TODO DEBUG HI RES STRUCRURES volume calculation z-slices spacing error.
'''

http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.8.8.html
http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1
'''


def prepare_dvh_data(dhist, dvh):
    dvhdata = {}
    dvhdata['dose_axis'] = dhist
    dvhdata['data'] = dvh
    dvhdata['bins'] = len(dvh)
    dvhdata['type'] = 'CUMULATIVE'
    dvhdata['doseunits'] = 'cGY'
    dvhdata['volumeunits'] = 'cm3'
    dvhdata['scaling'] = np.diff(dhist)[0]
    # dvhdata['scaling'] = 1.0  # standard 1 cGy bins
    dvhdata['min'] = get_dvh_min(dvh)
    dvhdata['max'] = get_dvh_max(dvh)
    dvhdata['mean'] = get_dvh_mean(dvh)
    return dvhdata


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


def contour2index(r_contour, dose_lut):
    xgrid = dose_lut[0]
    x = r_contour[:, 0]

    delta_x = np.abs(xgrid[0] - xgrid[1])
    ix = (x - xgrid[0]) / delta_x + 1

    ygrid = dose_lut[1]
    y = r_contour[:, 1]

    delta_y = np.abs(ygrid[0] - ygrid[1])
    iy = (y - ygrid[0]) / delta_y + 1

    roi_corners = np.dstack((ix, iy)).astype(dtype=np.int32)

    return roi_corners


def get_planes_thickness(planesDict):
    ordered_keys = [z for z, sPlane in planesDict.items()]
    ordered_keys.sort(key=float)
    planes = np.array(ordered_keys, dtype=float)

    delta = np.diff(planes)
    delta = np.append(delta, delta[0])
    planes_thickness = dict(zip(ordered_keys, delta))

    return planes_thickness


def get_capped_structure(structure, shift=0):
    """
        Return structure planes dict end caped
    :param structure: Structure Dict
    :param shift: end cap shift - (millimeters)
    :return: Structure Planes end-caped by shift
    """
    if shift == 0.0:
        return structure['planes']
    else:
        planesDict = structure['planes']

        out_Dict = deepcopy(planesDict)
        ordered_keys = [z for z in planesDict.keys()]
        ordered_keys.sort(key=float)
        planes = np.array(ordered_keys, dtype=float)
        start_cap = (planes[0] - shift)
        start_cap_key = '%.2f' % start_cap
        start_cap_values = planesDict[ordered_keys[0]]

        end_cap = (planes[-1] + shift)
        end_cap_key = '%.2f' % end_cap
        end_cap_values = planesDict[ordered_keys[-1]]

        out_Dict.pop(ordered_keys[0])
        out_Dict.pop(ordered_keys[-1])
        # adding structure caps
        out_Dict[start_cap_key] = start_cap_values
        out_Dict[end_cap_key] = end_cap_values

        return out_Dict


def get_bounding_lut(xmin, xmax, ymin, ymax, delta_mm, grid_delta):
    if delta_mm[0] != 0 and delta_mm[1] != 0:
        xmin -= delta_mm[0]
        xmax += delta_mm[0]
        ymin -= delta_mm[1]
        ymax += delta_mm[1]
        x_lut, x_delta = get_axis_grid(abs(delta_mm[0]), [xmin, xmax])
        y_lut, y_delta = get_axis_grid(abs(delta_mm[1]), [ymin, ymax])
        roi_lut = [x_lut, y_lut]
        return roi_lut
    else:
        x_lut, x_delta = get_axis_grid(abs(grid_delta[0]), [xmin, xmax])
        y_lut, y_delta = get_axis_grid(abs(grid_delta[1]), [ymin, ymax])
        roi_lut = [x_lut, y_lut]
    return roi_lut


def test_contour_roi_grid(contour_points, grid_delta, fac=1.0):
    x = contour_points[:, 0]
    y = contour_points[:, 1]

    deltas = [(-grid_delta[0] * fac, -grid_delta[1] * fac), (0, 0), (grid_delta[0] * fac, grid_delta[1] * fac)]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

    bound_rectangles = [get_bounding_lut(xmin, xmax, ymin, ymax, delta, grid_delta) for delta in deltas]

    return bound_rectangles


def gradient_info_boundary(contour, grid_delta, mapped_coord, dose_interp, z_c):
    br = test_contour_roi_grid(contour, grid_delta)
    st = {}
    columns = ['internal', 'bounding', 'external']
    for i in range(len(br)):
        xi, yi = wrap_xy_coordinates(br[i], mapped_coord)
        doseplane = dose_interp((z_c, yi, xi))
        sts = doseplane.max() - doseplane.min()
        st[columns[i]] = sts
    return st


def rms_gradient(df):
    """
        Mean gradient difference between 3 rectangles (internal, bounding and external at contour z)
        Gradient is defined by D_max - D_min (cGy)
    :param df: df_rectangles Pandas DataFrame
    :return: Mean gradient Difference
    """
    rms = (df['internal'] - df['bounding']) ** 2 + \
          (df['internal'] - df['external']) ** 2 + \
          (df['bounding'] - df['external']) ** 2
    return np.sqrt(rms / 3.0)


def get_boundary_stats(dose_mask, kind='max-min'):
    if kind == 'min':
        return dose_mask.min()
    elif kind == 'max':
        return dose_mask.max()
    elif kind == 'max-min':
        return dose_mask.max() - dose_mask.min()
    elif kind == 'max/min':
        return dose_mask.max() / dose_mask.min()
    elif kind == 'std':
        return dose_mask.std(ddof=1)
    elif kind == 'cv':
        return dose_mask.std(ddof=1) / dose_mask.mean()
    elif kind == 'mean':
        return dose_mask.mean()
    else:
        raise NotImplemented


def plot_slice_gradient(gradient_z, structure_name):
    try:
        df_grad = pd.DataFrame(gradient_z).T
        df_grad.columns = ['Average Gradient']
        fig, ax = plt.subplots()
        df_grad.plot(ax=ax)
        txt = structure_name + ' Average Boundary Gradient'
        ax.set_title(txt)
        ax.set_xlabel('Slice Position - z (mm)')
        ax.set_ylabel('Average Boundary gradient (cGy)')
        plt.show()
    except:
        raise NotImplemented


class Structure(object):
    def __init__(self, dicom_structure, calculation_options):
        """
            Class to encapsulate up-sampling and DVH calculation methodologies
        :param dicom_structure: Structure Dict - Planes and contours
        :param end_cap: Structure end cap.
        """
        self.structure = dicom_structure
        self.calculation_options = calculation_options
        self.contour_spacing = dicom_structure['thickness']
        self.grid_spacing = np.zeros(3)
        self.dose_lut = None
        self.dose_grid_points = None
        self.hi_res_structure = None
        self.dvh = np.array([])
        voxel_size = calculation_options['voxel_size']
        self.delta_mm = np.asarray([voxel_size, voxel_size, voxel_size])
        self.vol_lim = calculation_options['maximum_upsampled_volume_cc']
        self.organ2dvh = None
        self.dvh_data = None

    def set_delta(self, delta_mm):
        """
            Set oversampling voxel size (mm, mm, mm)
        :param delta_mm: (dx, dy, dz)
        """
        self.delta_mm = delta_mm

    @property
    def name(self):
        return self.structure['name']

    # @poperty Fix python 2.7 compatibility but it is very ineficient
    # @lazyproperty
    @property
    def planes(self):
        # TODO improve end capping method
        return get_capped_structure(self.structure, self.calculation_options['end_cap'])

    @property
    def volume_original(self):
        grid = [self.structure['thickness'], self.structure['thickness'], self.structure['thickness']]
        vol_cc = self.calculate_volume(self.structure['planes'], grid)
        return vol_cc

    # @lazyproperty
    @property
    def volume_cc(self):
        grid = [self.structure['thickness'], self.structure['thickness'], self.structure['thickness']]
        vol_cc = self.calculate_volume(self.planes, grid)
        return vol_cc

    # Fix python 2.7 compatibility
    # @lazyproperty
    @property
    def ordered_planes(self):
        """
            Return a 1D array from structure z planes coordinates
        :return:
        """
        ordered_keys = [z for z in self.planes.keys()]
        ordered_keys.sort(key=float)
        return np.array(ordered_keys, dtype=float)

    def up_sampling(self, lut_grid_3d, delta_mm):
        """
            Performas structure upsampling
        :param lut_grid_3d: X,Y,Z grid (mm)
        :param delta_mm: Voxel size in mm (dx,dy,dz)
        :return: sPlanes oversampled, dose_grid_points, grid_delta, dose_lut

        """
        self.dose_grid_points, self.dose_lut, self.grid_spacing = get_dose_grid_3d(lut_grid_3d, delta_mm)
        zi, dz = get_axis_grid(self.grid_spacing[2], self.ordered_planes)
        self.grid_spacing[2] = dz
        self.hi_res_structure = get_interpolated_structure_planes(self.planes, zi)

        return self.hi_res_structure, self.dose_grid_points, self.grid_spacing, self.dose_lut

    def _prepare_data(self, grid_3d, upsample):
        """
            Prepare structure data to run DVH loop calculation
        :param grid_3d: X,Y,Z grid coordinates (mm)
        :param upsample: True/False
        :return: sPlanes oversampled, dose_lut, dose_grid_points, grid_delta
        """
        if upsample:
            # upsample only small size structures
            # get structure slice position
            # set up-sample only small size structures
            if self.volume_original < self.vol_lim:
                # self._set_upsample_delta()
                hi_resolution_structure, grid_ds, grid_delta, dose_lut = self.up_sampling(grid_3d, self.delta_mm)
                dose_grid_points = grid_ds[:, :2]
                return hi_resolution_structure, dose_lut, dose_grid_points, grid_delta

            else:
                return self._get_calculation_data(grid_3d)

        else:
            return self._get_calculation_data(grid_3d)

    def _get_calculation_data(self, grid_3d):
        """
            Return all data need to calculate DVH without up-sampling
        :param grid_3d: X,Y,Z grid coordinates (mm)
        :return: sPlanes, dose_lut, dose_grid_points, grid_delta
        """
        dose_lut = [grid_3d[0], grid_3d[1]]
        dose_grid_points = get_dose_grid(dose_lut)
        x_delta = abs(grid_3d[0][0] - grid_3d[0][1])
        y_delta = abs(grid_3d[1][0] - grid_3d[1][1])
        # get structure slice position
        # ordered_z = self.ordered_planes
        z_delta = self.structure['thickness']
        grid_delta = [x_delta, y_delta, z_delta]
        return self.planes, dose_lut, dose_grid_points, grid_delta

    def _set_upsample_delta(self):
        if self.volume_original < 5:
            self.delta_mm = (0.05, 0.05, 0.05)
            # elif 10 < self.volume_original <= 40:
            #     self.delta_mm = (0.25, 0.25, 0.25)
            # elif 40 < self.volume_original <= 100:
            #     self.delta_mm = (0.5, 0.5, 0.5)
            # else:
            #     self.delta_mm = (1, 1, 1)

    def calculate_dvh(self, dicom_dose, bin_size=1.0, timing=False):
        """
            Calculates structure DVH using Winding Number(wn) method to check contour boundaries
        :param dicom_dose: DICOM-RT dose ScoringDicomParser object
        :param bin_size: histogram bin size in cGy
        :param up_sample: True/False
        :return: dose_range (cGy), cumulative dvh (cc)
        """
        print(' ----- DVH Calculation -----')
        print('Structure Name: %s - volume (cc) %1.3f' % (self.name, self.volume_cc))

        # 3D DOSE TRI-LINEAR INTERPOLATION
        dose_interp, grid_3d, mapped_coord = dicom_dose.DoseRegularGridInterpolator()
        # Set up_sampling or not
        up_sample = self.calculation_options['up_sampling']
        sPlanes, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, up_sample)

        print('End caping (mm): %1.2f ' % self.calculation_options['end_cap'])
        print('Grid delta (mm): (%1.2f, %1.2f, %1.2f) ' % (grid_delta[0], grid_delta[1], grid_delta[2]))

        # wrap z axis
        z_c, ordered_keys = wrap_z_coordinates(sPlanes, mapped_coord)

        # Create an empty array of bins to store the histogram in cGy
        # only if the structure has contour data or the dose grid exists
        maxdose = dicom_dose.global_max
        nbins = int(maxdose / bin_size)
        hist = np.zeros(nbins)
        volume = 0

        st = time.time()
        tested_voxels = []
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            s_plane = sPlanes[z]
            # print('calculating slice z: %.1f' % float(z))
            # Get the contours with calculated areas and the largest contour index
            contours, largest_index = calculate_contour_areas_numba(s_plane)

            # Calculate the histogram for each contour
            for j, contour in enumerate(contours):
                # Get the dose plane for the current structure contour at plane
                contour_dose_grid, ctr_dose_lut = get_contour_roi_grid(contour['data'], grid_delta, fac=1)

                x_c, y_c = wrap_xy_coordinates(ctr_dose_lut, mapped_coord)

                doseplane = dose_interp((z_c[i], y_c, x_c))
                tested_voxels.append(doseplane.size)

                m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid, contour['data'])

                h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)

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

        # Volume units are given in cm^3
        volume /= 1000
        # Rescale the histogram to reflect the total volume
        hist = hist * volume / sum(hist)
        self.hist = hist
        chist = get_cdvh_numba(hist)
        dhist = (np.arange(0, nbins) / nbins) * maxdose
        idx = np.nonzero(chist)  # remove 0 volumes from DVH

        # dose_range, cdvh = dhist, chist
        dose_range, cdvh = dhist[idx], chist[idx]
        end = time.time()
        el = end - st
        # print('elapsed (s):', el)
        if timing:
            return self.name, self.volume_original, np.sum(tested_voxels), self.delta_mm[0], el
        else:
            self.dvh_data = prepare_dvh_data(dose_range, cdvh)
            return dose_range, cdvh

    def get_dvh_data(self):
        return self.dvh_data

    def calculate_dvh_raster(self, dicom_dose, bin_size=1.0, up_sample=False):
        """
            Calculates structure DVH using Winding Number(wn) method to check contour boundaries
        :param dicom_dose: DICOM-RT dose ScoringDicomParser object
        :param bin_size: histogram bin size in cGy
        :param up_sample: True/False
        :return: dose_range (cGy), cumulative dvh (cc)
        """
        # print(' ----- DVH Calculation -----')
        # print('Structure Name: %s - volume (cc) %1.3f' % (self.name, self.volume_cc))

        # 3D DOSE TRI-LINEAR INTERPOLATION
        dose_interp, grid_3d, mapped_coord = dicom_dose.DoseRegularGridInterpolator()
        sPlanes, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, up_sample)

        # print('End caping:  ' + str(self.end_cap))
        # print('Grid delta (mm): ', grid_delta)


        # wrap z axis
        z_c, ordered_keys = wrap_z_coordinates(sPlanes, mapped_coord)

        # only if the structure has contour data or the dose grid exists
        maxdose = dicom_dose.global_max
        # Remove values above the limit (cGy) if specified
        nbins = int(maxdose / bin_size)
        hist = np.zeros(nbins)
        volume = 0
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            sPlane = sPlanes[z]

            # Get the contours with calculated areas and the largest contour index
            contours, largestIndex = calculate_contour_areas_numba(sPlane)

            # Calculate the histogram for each contour
            for j, contour in enumerate(contours):
                # Get the dose plane for the current structure contour at plane
                contour_dose_grid, ctr_dose_lut = get_contour_roi_grid(contour['data'], grid_delta, fac=1.0)

                x_c, y_c = wrap_xy_coordinates(ctr_dose_lut, mapped_coord)

                doseplane = dose_interp((z_c[i], y_c, x_c))

                xx, yy = np.meshgrid(ctr_dose_lut[0], ctr_dose_lut[1], indexing='xy', sparse=True)
                m = contour_rasterization_numba(ctr_dose_lut, contour_dose_grid, contour, xx, yy)
                # m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid, contour['data'])

                h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)

                if j == largestIndex:
                    hist += h
                    volume += vol
                # Otherwise, determine whether to add or subtract histogram
                # depending if the contour is within the largest contour or not
                else:
                    inside = check_contour_inside(contour['data'], contours[largestIndex]['data'])
                    # If the contour is inside, subtract it from the total histogram
                    if inside:
                        hist -= h
                        volume -= vol
                    # Otherwise it is outside, so add it to the total histogram
                    else:
                        hist += h
                        volume += vol

        # Volume units are given in cm^3
        volume /= 1000
        # Rescale the histogram to reflect the total volume
        hist = hist * volume / sum(hist)

        chist = get_cdvh_numba(hist)
        dhist = (np.arange(0, nbins) / nbins) * maxdose
        # idx = np.nonzero(chist)  # remove 0 volumes from DVH
        # dose_range, cdvh = dhist[idx], chist[idx]
        dose_range, cdvh = dhist, chist
        self.dvh_data = prepare_dvh_data(dose_range, cdvh)

        return dose_range, cdvh

    def calc_conformation_index(self, rtdose, lowerlimit, upsample=False):
        """From a selected structure and isodose line, return conformality index.
            Up sample structures calculation by Victor Alves
        Read "A simple scoring ratio to index the conformity of radiosurgical
        treatment plans" by Ian Paddick.
        J Neurosurg (Suppl 3) 93:219-222, 2000"""

        # print(' ----- Conformality index calculation -----')
        # print('Structure Name: %s - volume (cc) %1.3f - lower_limit (cGy):  %1.2f' % (
        #     self.name, self.volume_cc, lowerlimit))

        # 3D DOSE TRI-LINEAR INTERPOLATION
        dose_interp, grid_3d, mapped_coord = rtdose.DoseRegularGridInterpolator()

        sPlanes, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, upsample)

        xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)

        # Iterate over each plane in the structure
        # wrap coordinates
        fx, fy, fz = mapped_coord
        ordered_keys = [z for z, sPlane in sPlanes.items()]
        ordered_keys.sort(key=float)
        x_cord = fx(xx)
        y_cord = fy(yy)
        z_cord = fz(ordered_keys)

        PITV = 0  # Rx isodose volume in cc
        CV = 0  # coverage volume
        # st = time.time()
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            sPlane = sPlanes[z]

            # Get the contours with calculated areas and the largest contour index
            contours, largestIndex = calculate_contour_areas_numba(sPlane)

            # Get the dose plane for the current structure plane
            doseplane = dose_interp((z_cord[i], y_cord, x_cord))

            # If there is no dose for the current plane, go to the next plane
            if not len(doseplane):
                break

            for j, contour in enumerate(contours):
                m = get_contour_mask_wn(dose_lut, dosegrid_points, contour['data'])
                PITV_vol, CV_vol = self.calc_ci_vol(m, doseplane, lowerlimit, grid_delta)

                # If this is the largest contour, just add to the total volume
                if j == largestIndex:
                    PITV += PITV_vol
                    CV += CV_vol
                # Otherwise, determine whether to add or subtract
                # depending if the contour is within the largest contour or not
                else:
                    inside = check_contour_inside(contour['data'], contours[largestIndex]['data'])
                    # only add covered volume if contour is not inside the largest
                    if not inside:
                        CV += CV_vol

        # Volume units are given in cm^3
        PITV /= 1000.0
        CV /= 1000.0
        TV = self.calculate_volume(sPlanes, grid_delta)
        CI = CV * CV / (TV * PITV)
        # print('Conformity index: ', CI)
        # print('elapsed (s):', end - st)
        return CI

    @staticmethod
    def calc_ci_vol(mask, dose_plane, lowerlimit, grid_delta):

        """
            Calculate contour's volumes: coverage slice volume and PITV sclice volume
        :param mask: Contour slice boolean mask
        :param dose_plane: doseplane at z plane (cGy)
        :param lowerlimit: Dose limit (cGy)
        :param grid_delta: Voxel size (dx,dy,xz)
        :return: PITV_vol, CV_vol
        """
        cv_mask = dose_plane * mask

        # Calculate the volume for the contour for the given dose plane
        PITV_vol = np.sum(dose_plane > lowerlimit) * (grid_delta[0] * grid_delta[1] * grid_delta[2])

        CV_vol = np.sum(cv_mask > lowerlimit) * (grid_delta[0] * grid_delta[1] * grid_delta[2])

        return PITV_vol, CV_vol

    @staticmethod
    def calculate_volume(structure_planes, grid_delta):
        """Calculates the volume for the given structure.
        :rtype: float
        :param structure_planes: Structure planes dict
        :param grid_delta: Voxel size (dx,dy,xz)
        :return: Structure Volume
        """

        ordered_keys = [z for z, sPlane in structure_planes.items()]
        ordered_keys.sort(key=float)

        # Store the total volume of the structure
        sVolume = 0
        n = 0
        for z in ordered_keys:
            sPlane = structure_planes[z]
            # calculate contour areas
            contours, largestIndex = calculate_contour_areas_numba(sPlane)
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
            # only add half of the volume, otherwise add the full slice thickness
            if (n == 0) or (n == len(structure_planes) - 1):
                sVolume = float(sVolume) + float(area) * float(grid_delta[2]) * 0.5
            else:
                sVolume = float(sVolume) + float(area) * float(grid_delta[2])
            # Increment the current plane number
            n += 1

        # Since DICOM uses millimeters, convert from mm^3 to cm^3
        volume = sVolume / 1000

        return volume

    def boundary_rectangles(self, dicom_dose, up_sample=False):
        print(' ----- Boundary Gradient Calculation -----')
        print('Structure Name: %s - volume (cc) %1.3f' % (self.name, self.volume_cc))
        # 3D DOSE TRI-LINEAR INTERPOLATOR
        dose_interp, grid_3d, mapped_coord = dicom_dose.DoseRegularGridInterpolator()
        planes_dict, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, up_sample)
        print('End caping:  ' + str(self.end_cap))
        print('Grid delta (mm): ', grid_delta)
        # wrap z axis
        z_c, ordered_keys = wrap_z_coordinates(planes_dict, mapped_coord)
        internal = []
        bounding = []
        external = []
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            s_plane = planes_dict[z]
            # Get the contours with calculated areas and the largest contour index
            contours, largest_index = calculate_contour_areas_numba(s_plane)

            # Calculate the histogram for each contour
            for j, contour in enumerate(contours):
                # Get only the largest rectangle.
                if j == largest_index:
                    dfc = gradient_info_boundary(contour['data'], grid_delta, mapped_coord, dose_interp, z_c[i])
                    internal.append(dfc['internal'])
                    bounding.append(dfc['bounding'])
                    external.append(dfc['external'])

        df_rectangles = pd.DataFrame(internal, index=ordered_keys, columns=['internal'])
        df_rectangles['bounding'] = bounding
        df_rectangles['external'] = external
        df_rectangles['Boundary gradient'] = rms_gradient(df_rectangles)

        return df_rectangles


def get_dvh_upsampled(structure, dose, key, calculation_options):
    """Get a calculated cumulative DVH along with the associated parameters."""

    struc_teste = Structure(structure, calculation_options)
    dhist, chist = struc_teste.calculate_dvh(dose)
    dvh_data = prepare_dvh_data(dhist, chist)
    dvh_data['key'] = key

    return dvh_data


def calc_dvhs_upsampled(name, rs_file, rd_file, struc_names, out_file=False, calculation_options=None):
    """
        Computes structures DVH using a RS-DICOM and RD-DICOM diles
    :param rs_file: path to RS dicom-file
    :param rd_file: path to RD dicom-file
    :return: dict - computed DVHs
    """
    rtss = ScoringDicomParser(filename=rs_file)
    rtdose = ScoringDicomParser(filename=rd_file)
    # Obtain the structures and DVHs from the DICOM data
    structures = rtss.GetStructures()

    res = Parallel(n_jobs=calculation_options['num_cores'], verbose=11, backend='threading')(
        delayed(get_dvh_upsampled)(structure, rtdose, key, calculation_options) for key, structure in structures.items()
        if
        structure['name'] in struc_names)
    cdvh = {}
    for k in res:
        key = k['key']
        cdvh[structures[key]['name']] = k

    if out_file:
        out_obj = {'participant': name,
                   'DVH': cdvh}
        save(out_obj, out_file)

    return cdvh


def save_dicom_dvhs(name, rs_file, rd_file, out_file=False):
    """
        Computes structures DVH using a RS-DICOM and RD-DICOM diles
    :param rs_file: path to RS dicom-file
    :param rd_file: path to RD dicom-file
    :return: dict - computed DVHs
    """
    rtss = ScoringDicomParser(filename=rs_file)

    rtdose = ScoringDicomParser(filename=rd_file)
    # Obtain the structures and DVHs from the DICOM data
    structures = rtss.GetStructures()

    cdvh = rtdose.GetDVHs()

    out_dvh = {}
    for k, item in cdvh.items():
        cdvh_i = item['data']
        dose_cdvh = np.arange(cdvh_i.size) * item['scaling']
        dvh_data = prepare_dvh_data(dose_cdvh, cdvh_i)
        dvh_data['key'] = k
        out_dvh[structures[k]['name']] = dvh_data

    if out_file:
        out_obj = {'participant': name,
                   'DVH': out_dvh}
        save(out_obj, out_file)

    return cdvh


if __name__ == '__main__':
    pass
