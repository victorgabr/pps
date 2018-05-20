import os
from collections import OrderedDict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from shapely.geometry import Polygon

# from pyplanscoring.competition.constrains_evaluation import CompareDVH, save1, load1
from pyplanscoring.core.dicom_reader import ScoringDicomParser
from pyplanscoring.core.dosimetric import read_scoring_criteria
from pyplanscoring.core.dvhcalculation import get_boundary_stats, get_capped_structure
from pyplanscoring.core.geometry import wrap_z_coordinates, calc_area, get_contour_roi_grid, wrap_xy_coordinates, \
    get_contour_mask_wn, check_contour_inside, calculate_contour_areas, get_dose_grid, get_axis_grid, \
    get_interpolated_structure_planes, get_dose_grid_3d
from pyplanscoring.core.scoring import get_matched_names
from validation.validation import get_competition_data


def calculate_gradient_stats():
    rs = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/DICOM Sets/RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    #
    root_path = '/media/victor/TOURO Mobile/PLAN_TESTING_DATA'
    data = get_competition_data(root_path)

    rd_data = data[data[1] == 'rtdose']['index']
    res = []
    for rd in rd_data:
        df = calc_dvh_uncertainty(rd, rs, 'max', factor=0.5)
        tmp = df['mean'].copy()
        tmp.index = df['name']
        res.append(tmp)

    df_res = pd.concat(res, axis=1)

    return df_res


def msgd(gradient_measure):
    """
        Mean squarted gradient difference "max - max" contours difference
    :param gradient_measure: Gradient measure from contours boundary
    :return:
    """
    tmp = list(combinations(gradient_measure, 2))
    if len(gradient_measure) == 3:
        test = np.array(tmp)
        grad = np.sqrt(np.sum((test[:, 0] - test[:, 1]) ** 2) / len(tmp))

        mae = np.sum(np.abs(np.array(gradient_measure) - np.mean(gradient_measure))) / len(gradient_measure)

        return mae

    else:
        return np.NaN

        #


def plot_plane_contours_gradient(s_plane, structure_name, save_fig=False):
    ctrs, li = calculate_contours_uncertainty(s_plane, 1)
    fig, ax = plt.subplots()
    txt = structure_name + ' slice z: ' + str(s_plane[0]['contourData'][:, 2][0])
    for c in ctrs:
        ax.plot(c['data'][:, 0], -c['data'][:, 1], '.', label='Original')
        ax.plot(c['expanded'][:, 0], -c['expanded'][:, 1], '.', label='Expanded')
        ax.plot(c['shrunken'][:, 0], -c['shrunken'][:, 1], '.', label='shrunken')
        ax.set_title(txt)
        ax.legend()
        plt.axis('equal')
    if save_fig:
        fig.savefig(txt + '.png', format='png', dpi=100)


def expand_contour(contour_xy, distance):
    """
        Returns a geometry with an envelope at a distance from the object's envelope
          A negative delta has a "shrink" effect.
      :param contour_xy: Contour defined by [xi,yi] - [xn-1,yn-1], i = [0,n]
      :param distance: distance to expand or shrink.
      :return: modified contour
    """
    # repeat the first vertex at end
    poly_xy = np.zeros((contour_xy.shape[0] + 1, contour_xy.shape[1]))
    poly_xy[:-1] = contour_xy
    poly_xy[-1] = contour_xy[0]
    ctr_pol = Polygon(poly_xy)
    mod = ctr_pol.buffer(distance)

    return np.array(mod.exterior.coords)


def calculate_contours_uncertainty(plane, distance):
    """Calculate the area of each contour for the given plane.
        calculate_integrate both expanded and shrunken contours by a distance in mm
       Additionally calculate_integrate and return the largest contour index.
       :param plane: Plane Dictionary
       :param distance: distance in mm
       :return: Contours list and Largest index
       """

    # Calculate the area for each contour in the current plane
    contours = []
    largest = 0
    largestIndex = 0
    expanded = []
    shrunken = []
    for c, contour in enumerate(plane):
        cArea = calc_area(contour['contourData'][:, 0], contour['contourData'][:, 1])
        if cArea < 1:
            print('Warning contour with are less than 1 sq mm')
        try:
            expanded = expand_contour(contour['contourData'][:, :2], distance)
            shrunken = expand_contour(contour['contourData'][:, :2], -distance)
        except:
            print('impossible to shrink a contour of area: %1.2f by a distance %1.2f' % (cArea, distance))
            # shrunken = contour['contourData'][:, :2]

        # Add the contour area and points to the list of contours
        contours.append({'area': cArea,
                         'data': contour['contourData'][:, :2],
                         'expanded': expanded,
                         'shrunken': shrunken})

        # Determine which contour is the largest
        if cArea > largest:
            largest = cArea
            largestIndex = c
            # TODO implement a logger

    return contours, largestIndex


class StructurePaper:
    def __init__(self, dicom_structure, end_cap=False):
        self.structure = dicom_structure
        self.end_cap = end_cap
        self.contour_spacing = dicom_structure['thickness']
        self.grid_spacing = np.zeros(3)
        self.dose_lut = None
        self.dose_grid_points = None
        self.hi_res_structure = None
        self.vol_lim = 100.0
        self.dvh = np.array([])

    @property
    def name(self):
        return self.structure['name']

    # @poperty Fix python 2.7 compatibility but it is very ineficient
    # @lazyproperty
    @property
    def planes(self):
        # TODO improve end capping method
        return get_capped_structure(self.structure, self.end_cap)

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

    @staticmethod
    def calculate_volume(structure_planes, grid_delta):
        """Calculates the volume for the given structure.
        :rtype: float
        :param structure_planes: Structure planes dict
        :param grid_delta: Voxel size (dx,dy,xz)
        :return: Structure volume
        """

        ordered_keys = [z for z, sPlane in structure_planes.items()]
        ordered_keys.sort(key=float)

        # Store the total volume of the structure
        sVolume = 0
        n = 0
        for z in ordered_keys:
            sPlane = structure_planes[z]
            # calculate_integrate contour areas
            contours, largestIndex = calculate_contour_areas(sPlane)
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

    def _prepare_data(self, grid_3d, upsample):
        """
            Prepare structure data to run DVH loop calculation
        :param grid_3d: X,Y,Z grid coordinates (mm)
        :param upsample: True/False
        :return: sPlanes oversampled, dose_lut, dose_grid_points, grid_delta
        """
        if upsample:
            # upsample only small size structures
            # get structure slice Position
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
            Return all data need to calculate_integrate DVH without up-sampling
        :param grid_3d: X,Y,Z grid coordinates (mm)
        :return: sPlanes, dose_lut, dose_grid_points, grid_delta
        """
        dose_lut = [grid_3d[0], grid_3d[1]]
        dose_grid_points = get_dose_grid(dose_lut)
        x_delta = abs(grid_3d[0][0] - grid_3d[0][1])
        y_delta = abs(grid_3d[1][0] - grid_3d[1][1])
        # get structure slice Position
        # ordered_z = self.ordered_planes
        z_delta = self.structure['thickness']
        grid_delta = [x_delta, y_delta, z_delta]
        return self.planes, dose_lut, dose_grid_points, grid_delta

    def calc_boundary_gradient(self, dicom_dose, kind='max-min', up_sample=False, factor=1):
        print(' ----- Boundary Gradient Calculation -----')
        print('Structure Name: %s - volume (cc) %1.3f' % (self.name, self.volume_cc))
        # 3D DOSE TRI-LINEAR INTERPOLATOR
        dose_interp, grid_3d, mapped_coord = dicom_dose.DoseRegularGridInterpolator()
        planes_dict, dose_lut, dosegrid_points, grid_delta = self._prepare_data(grid_3d, up_sample)
        print('End caping:  ' + str(self.end_cap))
        print('Grid delta (mm): ', grid_delta)
        # wrap z axis
        z_c, ordered_keys = wrap_z_coordinates(planes_dict, mapped_coord)

        boundary_keys = ['shrunken', 'data', 'expanded']
        gradient_z = OrderedDict()
        for i in range(len(ordered_keys)):
            z = ordered_keys[i]
            s_plane = planes_dict[z]
            # Get the contours with calculated areas and the largest contour index
            contours, largest_index = calculate_contours_uncertainty(s_plane, grid_delta[0] * factor)

            # Calculate the histogram for each contour
            contour_gradient = []
            for j, contour in enumerate(contours):
                gradient_measure = []

                # check if there are 3 contours (shrunken, original and expanded)
                if len(contour['shrunken']) > 0 and len(contour['expanded']) > 0:
                    for k in boundary_keys:
                        # check if contour exists
                        if len(contour[k]) > 0:
                            contour_dose_grid, ctr_dose_lut = get_contour_roi_grid(contour[k], grid_delta)
                            x_c, y_c = wrap_xy_coordinates(ctr_dose_lut, mapped_coord)
                            doseplane = dose_interp((z_c[i], y_c, x_c))
                            m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid, contour['data'])
                            dose_filled_contour = doseplane * m
                            # get only nonzero doses to get stats
                            dose_mask = dose_filled_contour[np.nonzero(dose_filled_contour)]

                            # check if there is dose
                            if np.any(dose_mask):
                                # If this is the largest contour, just add to the total histogram
                                if j == largest_index:
                                    b_stats = get_boundary_stats(dose_mask, kind)
                                    gradient_measure.append(b_stats)

                                # Otherwise, determine whether to add or subtract
                                # depending if the contour is within the largest contour or not
                                else:
                                    if len(contours[largest_index][k]) > 0:
                                        inside = check_contour_inside(contour[k], contours[largest_index][k])
                                        # If the contour is inside, subtract it from the total histogram
                                        if not inside:
                                            b_stats = get_boundary_stats(dose_mask, kind)
                                            gradient_measure.append(b_stats)

                    # get gradient measure
                    contour_gradient.append(msgd(gradient_measure))

            gradient_z[z] = contour_gradient

        return gradient_z


def calc_gradient_pp(structure, dicom_dose, kind='max', factor=1):
    """
        Helper function to calculate_integrate the structure average boundary gradient difference (cGy)
    :param structure: Structure Dict
    :param dicom_dose: RR-DOSE - ScoringDicomParser object
    :return:
    """
    struc_test = StructurePaper(structure, end_cap=True)
    grad_z = struc_test.calc_boundary_gradient(dicom_dose, kind=kind, factor=factor)
    obs = np.concatenate([v for k, v in grad_z.items()])
    grad_mean = np.nanmean(obs)
    grad_std = np.nanstd(obs, ddof=1)
    grad_median = np.nanmedian(obs)
    return structure['name'], grad_mean, grad_std, grad_median


def calc_dvh_uncertainty(rd, rs, kind, factor):
    """
        Helper function to calculate_integrate using multiprocessing the average gradient
    :param rd: Path do DICOM-RTDOSE file
    :param rs: Path do DICOM-Structure file
    :return: Pandas Dataframe with estimated uncertainty on maximum dose (cGy)
    """
    rtss = ScoringDicomParser(filename=rs)
    dicom_dose = ScoringDicomParser(filename=rd)
    structures = rtss.GetStructures()

    snames = [  # 'PTV70',
        # 'PTV63',
        # 'PTV56',
        # 'OPTIC CHIASM',
        # 'OPTIC CHIASM PRV',
        # 'OPTIC N. RT',
        # 'OPTIC N. RT PRV',
        # 'OPTIC N. LT',
        # 'OPTIC N. LT PRV',
        # 'EYE RT',
        # 'EYE LT',
        # 'LENS RT',
        # 'LENS LT',
        # 'BRAINSTEM',
        # 'BRAINSTEM PRV',
        'SPINAL CORD',
        'MANDIBLE'
        # 'SPINAL CORD PRV',
        # 'PAROTID LT',
        # 'LIPS',
        # 'POST NECK',
        # 'ORAL CAVITY',
        # 'LARYNX',
        # 'BRACHIAL PLEXUS',
        # 'ESOPHAGUS']
    ]
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(calc_gradient_pp)(structure, dicom_dose, kind, factor) for key, structure in structures.items()
        if structure['name'] in snames)

    return pd.DataFrame(res, columns=['name', 'mean', 'std', 'median'])


def calc_batch_unc():
    rs = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm'
    sc = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/Scoring Criteria.txt'
    constrains, scores, criteria_df = read_scoring_criteria(sc)

    structures = ScoringDicomParser(filename=rs).GetStructures()
    # calculation_options = {}

    criteria_structure_names, names_dcm = get_matched_names(criteria_df.index.unique(), structures)
    structure_names = criteria_structure_names

    root_path = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/ECLIPSE'
    cmp = CompareDVH(root=root_path, rs_file=rs)
    folder_data = cmp.set_folder_data()

    res = []
    for k, val in folder_data.items():
        try:
            df = calc_dvh_uncertainty(val[2], rs, 'max', factor=0.5)
            tmp = df['mean'].copy()
            tmp.index = df['name']
            res.append(tmp)
        except:
            print('error')
    df_res = pd.concat(res, axis=1)


def save_all_unc():
    root_path = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/ECLIPSE'
    boundary_unc_path = os.path.join(root_path, 'Boundary_unceratinty_data.cmp')
    df = load1(boundary_unc_path)

    plt.style.use('ggplot')
    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/tests_paper/boundary_gradient'
    for row in df.iterrows():
        fig, ax = plt.subplots()
        row[1].plot(ax=ax, kind='hist')
        ax.set_xlabel('Mean boundary gradient [cGy]')
        ax.set_title(row[0])
        fig_name = row[0] + '_mean_boundary_gradient.png'
        fig.savefig(os.path.join(dest, fig_name), format='png', dpi=100)
        plt.close('all')


if __name__ == '__main__':
    rs = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm'
    rd = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/plans/Victor Alves 3180/RD.2017-PlanComp.Dose_PLAN.dcm'
    rs_dcm = ScoringDicomParser(filename=rs)

    structures = rs_dcm.GetStructures()

    # plt.style.use('ggplot')
    # structure = structures[24]
    # splanes = structure['planes']
    # s_plane = splanes['-1.50']
    # plot_plane_contours_gradient(s_plane, structure['name'], save_fig=False)

    # res_df = calc_dvh_uncertainty(rd, rs, 'max', 1)

    rtss = ScoringDicomParser(filename=rs)
    dicom_dose = ScoringDicomParser(filename=rd)
    structures = rtss.GetStructures()

    snames = ['SPINAL CORD', 'MANDIBLE']

    res = []
    grads = {}
    for key, structure in structures.items():
        if structure['name'] in snames:
            tmp = calc_gradient_pp(structure, dicom_dose, kind='max', factor=1)
            res.append(tmp)
            struc_test = StructurePaper(structure)
            grad_z = struc_test.calc_boundary_gradient(dicom_dose, kind='max', factor=1)
            grads[structure['name']] = grad_z
    res_df = pd.DataFrame(res, columns=['name', 'mean', 'std', 'median'])

    plt.style.use('ggplot')
    sc = pd.DataFrame(grads['SPINAL CORD']).T
    fig, ax = plt.subplots()
    ax.plot(sc.index, sc.values)
    ax.set_xlabel('Axial slice position - z[mm]')
    ax.set_ylabel('Maximum dose - MAE [cGy]')
    ax.set_title('H&N case - VMAT plan - SPINAL CORD - Mean Absolute Error')
    plt.show()

# sc = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/Scoring Criteria.txt'
# constrains, scores, criteria_df = read_scoring_criteria(sc)
#
# structures = ScoringDicomParser(filename=rs).get_structures()
# # calculation_options = {}
#
# criteria_structure_names, names_dcm = get_matched_names(criteria_df.index.unique(), structures)
# structure_names = criteria_structure_names
#
# root_path = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/ECLIPSE'
# cmp = CompareDVH(root=root_path, rs_file=rs)
# folder_data = cmp.set_folder_data()
#
# res = []
# for k, val in folder_data.items():
#     try:
#         df = calc_dvh_uncertainty(val[2], rs, 'max', factor=0.5)
#         tmp = df['mean'].copy()
#         tmp.index = df['name']
#         res.append(tmp)
#     except:
#         print('error')
# df_res = pd.concat(res, axis=1)
