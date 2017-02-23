from __future__ import division

from copy import deepcopy

import numpy as np
import numpy.ma as ma
from joblib import Parallel
from joblib import delayed

from pyplanscoring.dev.geometry import get_contour_mask_wn, get_dose_grid_3d, \
    get_axis_grid, get_dose_grid, \
    get_interpolated_structure_planes, contour_rasterization_numba, check_contour_inside, \
    wrap_coordinates, get_contour_roi_grid, wrap_xy_coordinates
from pyplanscoring.dicomparser import ScoringDicomParser, lazyproperty
from pyplanscoring.dvhcalc import get_cdvh_numba, calculate_contour_areas_numba, save
from pyplanscoring.dvhdoses import get_dvh_min, get_dvh_max, get_dvh_mean

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


def get_capped_structure(structure):
    planesDict = structure['planes']

    out_Dict = deepcopy(planesDict)
    ordered_keys = [z for z, sPlane in planesDict.items()]
    ordered_keys.sort(key=float)
    planes = np.array(ordered_keys, dtype=float)
    start_cap = (planes[0] - structure['thickness'] / 2.0)
    start_cap_key = '%.2f' % start_cap
    start_cap_values = planesDict[ordered_keys[0]]

    end_cap = (planes[-1] + structure['thickness'] / 2.0)
    end_cap_key = '%.2f' % end_cap
    end_cap_values = planesDict[ordered_keys[-1]]

    out_Dict.pop(ordered_keys[0])
    out_Dict.pop(ordered_keys[-1])
    # adding structure caps
    out_Dict[start_cap_key] = start_cap_values
    out_Dict[end_cap_key] = end_cap_values

    return out_Dict


class Structure(object):
    def __init__(self, dicom_structure, end_cap=False):
        """
            Class to encapsulate up-sampling and DVH calculation methodologies
        :param dicom_structure: Structure Dict - Planes and contours
        :param end_cap: Structure end cap.
        """
        self.structure = dicom_structure
        self.end_cap = end_cap
        self.contour_spacing = dicom_structure['thickness']
        self.grid_spacing = np.zeros(3)
        self.dose_lut = None
        self.dose_grid_points = None
        self.hi_res_structure = None
        self.dvh = np.array([])
        self.delta_mm = np.asarray([0.25, 0.25, 0.1])
        self.vol_lim = 100
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
    @lazyproperty
    def planes(self):
        if self.end_cap:
            # TODO improve end capping method
            return get_capped_structure(self.structure)
        else:
            return self.structure['planes']

    @property
    def volume_original(self):
        grid = [self.structure['thickness'], self.structure['thickness'], self.structure['thickness']]
        vol_cc = self.calculate_volume(self.structure['planes'], grid)
        return vol_cc

    @lazyproperty
    def volume_cc(self):
        grid = [self.structure['thickness'], self.structure['thickness'], self.structure['thickness']]
        vol_cc = self.calculate_volume(self.planes, grid)
        return vol_cc

    # Fix python 2.7 compatibility
    @lazyproperty
    def ordered_planes(self):
        """
            Return a 1D array from structure z planes coordinates
        :return:
        """
        ordered_keys = [z for z, sPlane in self.planes.items()]
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
            if self.volume_cc < self.vol_lim:
                # get structure slice position
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
        ordered_z = self.ordered_planes
        z_delta = abs(ordered_z[1] - ordered_z[2])
        grid_delta = [x_delta, y_delta, z_delta]
        return self.planes, dose_lut, dose_grid_points, grid_delta

    def calculate_dvh(self, dicom_dose, bin_size=1.0, up_sample=False):
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
        sPlanes, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, up_sample)
        print('End caping:  ' + str(self.end_cap))
        print('Grid delta (mm): ', grid_delta)

        # wrap coordinates
        _, __, z_c, ordered_keys = wrap_coordinates(sPlanes, dose_lut, mapped_coord)

        # Create an empty array of bins to store the histogram in cGy
        # only if the structure has contour data or the dose grid exists
        maxdose = dicom_dose.global_max
        nbins = int(maxdose / bin_size)
        hist = np.zeros(nbins)
        volume = 0
        import time
        st = time.time()
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            sPlane = sPlanes[z]
            # print('calculating slice z: %.1f' % float(z))
            # Get the contours with calculated areas and the largest contour index
            contours, largestIndex = calculate_contour_areas_numba(sPlane)

            # If there is no dose for the current plane, go to the next plane
            # if not len(doseplane):
            #     break

            # Calculate the histogram for each contour
            for j, contour in enumerate(contours):
                # Get the dose plane for the current structure contour at plane
                contour_dose_grid, ctr_dose_lut = get_contour_roi_grid(contour['data'], grid_delta, fac=1)

                x_c, y_c = wrap_xy_coordinates(ctr_dose_lut, mapped_coord)

                doseplane = dose_interp((z_c[i], y_c, x_c))

                m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid, contour['data'])

                h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)

                # If this is the largest contour, just add to the total histogram
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
        idx = np.nonzero(chist)  # remove 0 volumes from DVH
        dose_range, cdvh = dhist[idx], chist[idx]
        end = time.time()
        print('elapsed original (s)', end - st)

        # dose_range, cdvh = dhist, chist
        self.dvh_data = prepare_dvh_data(dose_range, cdvh)

        return dose_range, cdvh

    def calculate_dvh_slow(self, dicom_dose, bin_size=1.0, up_sample=False):
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
        sPlanes, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, up_sample)
        print('End caping:  ' + str(self.end_cap))
        print('Grid delta (mm): ', grid_delta)

        # wrap coordinates
        x_c, y_c, z_c, ordered_keys = wrap_coordinates(sPlanes, dose_lut, mapped_coord)

        # Create an empty array of bins to store the histogram in cGy
        # only if the structure has contour data or the dose grid exists
        maxdose = dicom_dose.global_max
        nbins = int(maxdose / bin_size)
        hist = np.zeros(nbins)
        volume = 0
        import time
        st = time.time()
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            sPlane = sPlanes[z]
            print('calculating slice z: %.1f' % float(z))
            # Get the contours with calculated areas and the largest contour index
            contours, largestIndex = calculate_contour_areas_numba(sPlane)

            # Get the dose plane for the current structure plane
            doseplane = dose_interp((z_c[i], y_c, x_c))

            # # If there is no dose for the current plane, go to the next plane
            # if not len(doseplane):
            #     break

            # Calculate the histogram for each contour
            for j, contour in enumerate(contours):
                m = get_contour_mask_wn(dose_lut, dosegrid_points, contour['data'])
                h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)

                # If this is the largest contour, just add to the total histogram
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
        idx = np.nonzero(chist)  # remove 0 volumes from DVH
        dose_range, cdvh = dhist[idx], chist[idx]
        end = time.time()
        print('elapsed original (s)', end - st)

        # dose_range, cdvh = dhist, chist
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


        # wrap coordinates
        x_c, y_c, z_c, ordered_keys = wrap_coordinates(sPlanes, dose_lut, mapped_coord)

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
        J Neurosurg (Suppl 3) 93:219-222, 2000

        :param rtdose: DICOM-RT dose ScoringDicomParser object
        :param lowerlimit: Lower limit dose value (cGy)
        :param upsample: True/False
        :return: conformality index
        """

        # print(' ----- Conformality index calculation -----')
        # print('Structure Name: %s - volume (cc) %1.3f - lower_limit (cGy):  %1.2f' % (
        #     self.name, self.volume_cc, lowerlimit))

        # 3D DOSE TRI-LINEAR INTERPOLATION
        dose_interp, grid_3d, mapped_coord = rtdose.DoseRegularGridInterpolator()

        sPlanes, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, upsample)

        # wrap coordinates
        x_c, y_c, z_c, ordered_keys = wrap_coordinates(sPlanes, dose_lut, mapped_coord)

        PITV = 0  # Rx isodose volume in cc
        CV = 0  # coverage volume
        # st = time.time()
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            sPlane = sPlanes[z]

            # Get the contours with calculated areas and the largest contour index
            contours, largestIndex = calculate_contour_areas_numba(sPlane)

            # Calculate the histogram for each contour
            for j, contour in enumerate(contours):
                # Get the dose plane for the current structure contour at plane
                contour_dose_grid, ctr_dose_lut = get_contour_roi_grid(contour['data'], grid_delta)

                x_c, y_c = wrap_xy_coordinates(ctr_dose_lut, mapped_coord)

                doseplane = dose_interp((z_c[i], y_c, x_c))

                m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid, contour['data'])

                PITV_vol, CV_vol = self.calc_ci_vol(m, doseplane, lowerlimit, grid_delta)

                # If this is the largest contour, just add to the total volume
                if i == largestIndex:
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


def get_dvh_upsampled(structure, dose, key, end_cap=False, upsample=True):
    """Get a calculated cumulative DVH along with the associated parameters."""

    struc_teste = Structure(structure, end_cap=end_cap)
    dhist, chist = struc_teste.calculate_dvh(dose, up_sample=upsample)
    dvh_data = prepare_dvh_data(dhist, chist)
    dvh_data['key'] = key

    return dvh_data


def calc_dvhs_upsampled(name, rs_file, rd_file, struc_names, out_file=False, end_cap=False, upsample=True):
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

    if upsample:
        upsample = True
    else:
        upsample = False

    # backend = 'threading'
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(get_dvh_upsampled)(structure, rtdose, key, end_cap, upsample) for key, structure in structures.items()
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
