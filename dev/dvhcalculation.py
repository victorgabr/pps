import time

import numpy as np
import numpy.ma as ma
from copy import deepcopy

from joblib import Parallel
from joblib import delayed
from scipy.signal import savgol_filter

from dev.geometry import get_contour_mask_wn, expand_roi, calculate_planes_contour_areas, get_dose_grid_3d, \
    get_axis_grid, get_dose_grid, \
    get_interpolated_structure_planes, point_in_contour, calculate_structure_volume
from dicomparser import ScoringDicomParser, lazyproperty
from dvhcalc import get_cdvh_numba, calculate_contour_areas_numba, save
from dvhdoses import get_dvh_min, get_dvh_max, get_dvh_mean

float_formatter = lambda x: "%.2f" % x
# str_formatter = lambda x: "%.2s" % x
np.set_printoptions(formatter={'float_kind': float_formatter})
# np.set_printoptions(formatter={'str_kind': str_formatter})

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
    dvhdata['volumeunits'] = 'CM3'
    dvhdata['scaling'] = np.diff(dhist)[0]
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


def get_contour_opencv(doseplane, contour, dose_lut):
    r_contour = contour['data']
    # fill the ROI so it doesn't get wiped out when the mask is applied
    mask_teste = np.zeros(doseplane.shape, dtype=np.uint8)
    ignore_mask_color = (1,)
    roi_corners = contour2index(r_contour, dose_lut)
    cv2.fillPoly(mask_teste, roi_corners, ignore_mask_color)

    return mask_teste.astype(bool)


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


def calc_vol(mask, doseplane, lowerlimit, grid_delta):
    """Calculate the differential DVH for the given contour and dose plane."""

    # Multiply the structure mask by the dose plane to get the dose mask
    mask = ma.array(doseplane, mask=~mask)

    # Calculate the volume for the contour for the given dose plane
    PITV_vol = np.sum(doseplane > lowerlimit) * (grid_delta[0] * grid_delta[1] * grid_delta[2])

    CV_vol = np.sum(mask > lowerlimit) * (grid_delta[0] * grid_delta[1] * grid_delta[2])

    return PITV_vol, CV_vol


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

    # adding structure caps
    out_Dict[start_cap_key] = start_cap_values
    out_Dict[end_cap_key] = end_cap_values

    return out_Dict


class Structure(object):
    # TODO REFACTOR STRUCTURE CLASS TO UPSAMPLE OR NOT SMALL STRUCTURES.
    def __init__(self, dicom_structure, end_cap=False):
        self.structure = dicom_structure
        self.end_cap = end_cap
        self.contour_spacing = dicom_structure['thickness']
        self.grid_spacing = np.zeros(3)
        self.dose_lut = None
        self.dose_grid_points = None
        self.hi_res_structure = None
        self.dvh = np.array([])

    @property
    def name(self):
        return self.structure['name']

    @lazyproperty
    def planes(self):
        if self.end_cap:
            return get_capped_structure(self.structure)
        else:
            return self.structure['planes']

    @lazyproperty
    def volume_cc(self):
        return calculate_structure_volume(self.structure)

    @lazyproperty
    def ordered_planes(self):
        ordered_keys = [z for z, sPlane in self.planes.items()]
        ordered_keys.sort(key=float)
        return np.array(ordered_keys, dtype=float)

    def get_expanded_roi(self, delta_mm):
        # TODO refactor API
        """
            Expand roi by 1/2 structure thickness in x,y and z axis
        """
        return expand_roi(self.planes, delta=delta_mm)

    @lazyproperty
    def planes_expanded(self):
        return expand_roi(self.planes, self.contour_spacing / 2.0)

    def up_sampling(self, lut_grid_3d, delta_mm, expanded=False):
        if expanded:
            planes = self.planes_expanded
        else:
            planes = self.planes

        # get structure slice position
        # ordered_z = np.unique(np.concatenate(planes)[:, 2])

        # ROI UP SAMPLING IN X, Y, Z
        self.dose_grid_points, self.dose_lut, self.grid_spacing = get_dose_grid_3d(lut_grid_3d, delta_mm)
        zi, dz = get_axis_grid(self.grid_spacing[2], self.ordered_planes)
        self.grid_spacing[2] = dz
        print('Upsampling ON')

        self.hi_res_structure = get_interpolated_structure_planes(self.planes, zi)

        return self.hi_res_structure, self.dose_grid_points, self.grid_spacing, self.dose_lut

    def _prepare_data(self, dose, upsample, delta_cm):

        grid_3d = dose.get_grid_3d()
        if upsample:
            # TODO auto select upsampling delta from dose and structure grids
            if self.volume_cc < 10:
                ds, grid_ds, grid_delta, dose_lut = self.up_sampling(grid_3d, delta_cm)
                dosegrid_points = grid_ds[:, :2]
                return ds, dose_lut, dosegrid_points, grid_delta
            else:
                ds = self.planes
                dose_lut = [grid_3d[0], grid_3d[1]]
                dosegrid_points = get_dose_grid(dose_lut)
                x_delta = abs(grid_3d[0][0] - grid_3d[0][1])
                y_delta = abs(grid_3d[1][0] - grid_3d[1][1])
                # get structure slice position
                ordered_z = self.ordered_planes
                z_delta = abs(ordered_z[0] - ordered_z[1])
                grid_delta = [x_delta, y_delta, z_delta]
                return ds, dose_lut, dosegrid_points, grid_delta

        else:
            dose_lut = [grid_3d[0], grid_3d[1]]
            dosegrid_points = get_dose_grid(dose_lut)
            x_delta = abs(grid_3d[0][0] - grid_3d[0][1])
            y_delta = abs(grid_3d[1][0] - grid_3d[1][1])
            # get structure slice position
            ordered_z = self.ordered_planes
            z_delta = abs(ordered_z[0] - ordered_z[1])
            grid_delta = [x_delta, y_delta, z_delta]
            return self.planes, dose_lut, dosegrid_points, grid_delta

    def calculate_dvh(self, dicom_dose, bin_size=1.0, upsample=False, delta_cm=(.5, .5, .5)):

        print(' ----- DVH Calculation -----')
        print('Structure Name: %s - volume (cc) %1.3f' % (self.name, self.volume_cc))
        sPlanes, dose_lut, dosegrid_points, grid_delta = self._prepare_data(dicom_dose, upsample, delta_cm)
        print('End caping:  ' + str(self.end_cap))
        print('Grid delta (mm): ', grid_delta)

        # 3D DOSE TRI-LINEAR INTERPOLATION
        dose_interp, values = dicom_dose.DoseRegularGridInterpolator()
        xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)

        # Create an empty array of bins to store the histogram in cGy
        # only if the structure has contour data or the dose grid exists
        dd = dicom_dose.GetDoseData()
        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)

        # Remove values above the limit (cGy) if specified
        nbins = int(maxdose / bin_size)
        hist = np.zeros(nbins)

        n_voxels = []
        st = time.time()
        volume = 0

        # Iterate over each plane in the structure
        planes_dz = get_planes_thickness(sPlanes)

        # ordered keys
        ordered_keys = [z for z, sPlane in sPlanes.items()]
        ordered_keys.sort(key=float)

        for z in ordered_keys:
            # for z, sPlane in sPlanes.items():
            sPlane = sPlanes[z]
            print('calculating slice z: %.1f' % float(z))
            grid_delta[2] = planes_dz[z]
            # Get the contours with calculated areas and the largest contour index
            contours, largestIndex = calculate_contour_areas_numba(sPlane)

            # Get the dose plane for the current structure plane
            doseplane = dose_interp((z, yy, xx))
            # If there is no dose for the current plane, go to the next plane

            if not len(doseplane):
                break

            # Calculate the histogram for each contour
            for i, contour in enumerate(contours):
                m = get_contour_mask_wn(dose_lut, dosegrid_points, contour['data'])
                h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)

                mask = ma.array(doseplane, mask=~m)
                n_voxels.append(len(mask.compressed()))

                # If this is the largest contour, just add to the total histogram
                if i == largestIndex:
                    hist += h
                    volume += vol
                # Otherwise, determine whether to add or subtract histogram
                # depending if the contour is within the largest contour or not
                else:
                    contour['inside'] = False
                    for point in contour['data']:
                        poly = contours[largestIndex]['data']
                        if point_in_contour(point, poly):
                            contour['inside'] = True
                            # Assume if one point is inside, all will be inside
                            break
                    # If the contour is inside, subtract it from the total histogram
                    if contour['inside']:
                        hist -= h
                        volume -= vol
                    # Otherwise it is outside, so add it to the total histogram
                    else:
                        hist += h
                        volume += vol

        # Volume units are given in cm^3
        volume /= 1000
        # volume = self.volume_cc
        # Rescale the histogram to reflect the total volume
        hist = hist * volume / sum(hist)
        # Remove the bins above the max dose for the structure
        chist = get_cdvh_numba(hist)
        dhist = (np.arange(0, nbins) / nbins) * maxdose
        idx = np.nonzero(chist)  # remove 0 volumes from DVH

        # dose_range, cdvh = dhist, chist
        dose_range, cdvh = dhist[idx], chist[idx]
        end = time.time()

        print('elapsed (s):', end - st)
        print('number of structure voxels: %i' % np.sum(n_voxels))
        print(' ----- END DVH Calculation -----')

        return dose_range, cdvh

    @lazyproperty
    def smoothed_dvh(self):
        window_size = int(len(self.dvh) / 10)
        if window_size % 2 == 0:
            window_size += 1

        return savgol_filter(self.dvh, window_size, 3)

    def CalculateCI(self, rtdose, lowerlimit, upsample=False, delta_cm=(.5, .5, .5)):
        """From a selected structure and isodose line, return conformality index.
            Up sample structures calculation by Victor Alves
        Read "A simple scoring ratio to index the conformity of radiosurgical
        treatment plans" by Ian Paddick.
        J Neurosurg (Suppl 3) 93:219-222, 2000"""
        # TODO REFACTOR CI CALCULATION

        print(' ----- Conformality index calculation -----')
        print('Structure Name: %s - volume (cc) %1.3f' % (self.name, self.volume_cc))

        ds, dose_lut, dosegrid_points, grid_delta = self._prepare_data(rtdose, upsample, delta_cm)
        contours, largest_index = calculate_planes_contour_areas(ds)

        # 3D trilinear DOSE INTERPOLATION

        dose_interp, values = rtdose.DoseRegularGridInterpolator()
        xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)

        # Create an empty array of bins to store the histogram in cGy
        # only if the structure has contour data or the dose grid exists
        dd = rtdose.GetDoseData()

        PITV = 0  # Rx isodose volume in cc
        CV = 0  # coverage volume
        # Calculate the histogram for each contour
        for i, contour in enumerate(contours):
            z = contour['z']
            dose_plane = dose_interp((z, yy, xx))
            m = get_contour_mask_wn(dose_lut, dosegrid_points, contour['data'])
            PITV_vol, CV_vol = calc_vol(m, dose_plane, lowerlimit, grid_delta)
            PITV += PITV_vol
            CV += CV_vol

        # Volume units are given in cm^3
        PITV /= 1000.0
        CV /= 1000.0

        return PITV, CV


def get_dvh_upsampled(structure, dose, key):
    """Get a calculated cumulative DVH along with the associated parameters."""

    struc_teste = Structure(structure)
    dhist, chist = struc_teste.calculate_dvh(dose, upsample=True, delta_cm=(0.5, 0.5, 0.5))
    dvh_data = prepare_dvh_data(dhist, chist)
    dvh_data['key'] = key

    return dvh_data


def calc_dvhs_upsampled(name, rs_file, rd_file, out_file=False):
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
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(get_dvh_upsampled)(structure, rtdose, key) for key, structure in structures.items())
    cdvh = {}
    for k in res:
        key = k['key']
        cdvh[structures[key]['name']] = k

    if out_file:
        out_obj = {'participant': name,
                   'DVH': cdvh}
        save(out_obj, out_file)

    return cdvh


if __name__ == '__main__':
    # Hi resolution
    # rs_file = r'/media/victor/TOURO Mobile/COMPETITION 2017/Send to Victor - Jan10 2017/AN Plan High Res All/RS.1.2.246.352.205.4880373775416368842.11450890729805262762.dcm'
    # rd_file = r'/media/victor/TOURO Mobile/COMPETITION 2017/Send to Victor - Jan10 2017/AN Plan High Res All/RD.1.2.246.352.71.7.584747638204.1746067.20170110180352.dcm'

    rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/testdata/DVH-Analysis-Data-Etc/STRUCTURES/Sphere_30_0.dcm'

    rd_file = '/home/victor/Dropbox/Plan_Competition_Project/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm'
    delta_mm = (1, 1, 1)

    # TODO DEBUG AND IMPLEMENT DVH CALCULATION USING UPSAMPLING

    dose = ScoringDicomParser(filename=rd_file)
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()
    lut_grid_3d = dose.get_grid_3d()

    ecl_DVH = dose.GetDVHs()

    # st = time.time()
    # for structure in structures.values():
    #     if structure['id'] in ecl_DVH:
    #         # if structure['name'] == 'BRACHIAL PLEXUS':
    #         dicompyler_dvh = get_dvh(structure, dose)
    #         struc_teste = Structure(structure)
    #         ecl_dvh = ecl_DVH[structure['id']]['data']
    #         dhist, chist = struc_teste.calculate_dvh(dose, upsample=True, delta_cm=delta_mm)
    #         plt.figure()
    #         plt.plot(chist, label='Up Sampled Structure')
    #         plt.hold(True)
    #         plt.plot(ecl_dvh, label='Eclipse DVH')
    #         plt.title('ID: ' + str(structure['id']) + ' ' + structure['name'] + ' volume (cc): %1.3f' % ecl_dvh[0])
    #         plt.plot(dicompyler_dvh['data'], label='Not up sampled')
    #         plt.legend(loc='best')
    # end = time.time()
    # print('Total elapsed Time (min):  ', (end - st) / 60)
    # plt.show()

    structure = structures[2]

    struc_teste = Structure(structure)
    grid_3d = dose.get_grid_3d()
    hi_res_structure, dose_grid_points, grid_spacing, dose_lut = struc_teste.up_sampling(grid_3d, delta_mm=(1, 1, 1))
    np.set_printoptions(precision=4)

    delta = np.diff(planes)
    delta = np.append(delta, delta[0])
    planes_thickness = dict(zip(ordered_keys, delta))
