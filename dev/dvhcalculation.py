import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.spatial.qhull import ConvexHull

from scipy.signal import savgol_filter
from dev.geometry import get_contour_mask_wn, get_structure_planes, \
    expand_roi, calculate_planes_contour_areas, get_dose_grid_3d, get_z_planes, get_axis_grid, get_dose_grid, \
    savitzky_golay
from dicomparser import ScoringDicomParser, lazyproperty
from dvhcalc import get_cdvh_numba, get_dvh

'''

http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.8.8.html
http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1
'''


def calculate_contour_dvh(mask, doseplane, bins, maxdose, grid_delta):
    """Calculate the differential DVH for the given contour and dose plane."""

    # Multiply the structure mask by the dose plane to get the dose mask
    mask = ma.array(doseplane, mask=~mask)

    # Calculate the differential dvh
    hist, edges = np.histogram(mask.compressed(),
                               bins=bins,
                               range=(0, maxdose))

    # Calculate the volume for the contour for the given dose plane
    vol = np.sum(hist) * grid_delta[0] * grid_delta[1] * grid_delta[2]

    return hist, vol


class Structure(object):
    def __init__(self, struc, end_cap=False):
        self.structure = struc
        self.end_cap = end_cap
        self.contour_spacing = struc['thickness']
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
        return get_structure_planes(self.structure, end_capping=self.end_cap)

    @lazyproperty
    def contours(self):
        contours, _ = calculate_planes_contour_areas(self.planes)
        return contours

    @lazyproperty
    def volume_cc(self):
        ordered_z = np.unique(np.concatenate(self.planes)[:, 2])
        dz = abs(ordered_z[1] - ordered_z[0])
        vol = np.sum([c['area'] * dz for c in self.contours]) / 1000.0  # volume in cc
        return vol

    def get_expanded_roi(self, delta_mm):
        """
            Expand roi by 1/2 structure thickness in x,y and z axis
        """
        return expand_roi(self.planes, delta=delta_mm)

    @lazyproperty
    def planes_expanded(self):
        return expand_roi(self.planes, self.contour_spacing / 2.0)

    def up_sampling(self, lut_grid_3d, delta_mm=(0.2, 0.2, 0.2), expanded=False):
        if expanded:
            planes = self.planes_expanded
        else:
            planes = self.planes

        # get structure slice position
        ordered_z = np.unique(np.concatenate(planes)[:, 2])

        # ROI UP SAMPLING IN X, Y, Z
        self.dose_grid_points, self.dose_lut, self.grid_spacing = get_dose_grid_3d(lut_grid_3d, delta_mm)
        zi, dz = get_axis_grid(self.grid_spacing[2], ordered_z)
        self.grid_spacing[2] = dz
        print('Grid delta (mm): ', self.grid_spacing)

        self.hi_res_structure = get_z_planes(planes, ordered_z, zi)

        return self.hi_res_structure, self.dose_grid_points, self.grid_spacing, self.dose_lut

    def _prepare_dvh_calc(self, dose, upsample, delta_cm):

        grid_3d = dose.get_grid_3d()
        if upsample:
            if self.volume_cc < 100.0:
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
                ordered_z = np.unique(np.concatenate(ds)[:, 2])
                z_delta = abs(ordered_z[0] - ordered_z[1])
                grid_delta = [x_delta, y_delta, z_delta]
                return ds, dose_lut, dosegrid_points, grid_delta

        else:
            ds = get_structure_planes(self.structure)
            dose_lut = [grid_3d[0], grid_3d[1]]
            dosegrid_points = get_dose_grid(dose_lut)
            x_delta = abs(grid_3d[0][0] - grid_3d[0][1])
            y_delta = abs(grid_3d[1][0] - grid_3d[1][1])
            # get structure slice position
            ordered_z = np.unique(np.concatenate(ds)[:, 2])
            z_delta = abs(ordered_z[0] - ordered_z[1])
            grid_delta = [x_delta, y_delta, z_delta]
            return ds, dose_lut, dosegrid_points, grid_delta

    def calculate_dvh(self, dicom_dose, bin_size=1.0, upsample=False, delta_cm=(.5, .5, .5)):

        # ROI UP SAMPLING IN X, Y, Z

        print(' ----- DVH Calculation -----')
        print('Structure Name: %s - volume (cc) %1.3f' % (self.name, self.volume_cc))
        ds, dose_lut, dosegrid_points, grid_delta = self._prepare_dvh_calc(dicom_dose, upsample, delta_cm)

        contours, largest_index = calculate_planes_contour_areas(ds)

        # DOSE INTERPOLATION
        dose_interp, values = dicom_dose.DoseRegularGridInterpolator()
        xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)

        # Create an empty array of bins to store the histogram in cGy
        # only if the structure has contour data or the dose grid exists
        dd = dicom_dose.GetDoseData()
        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)

        # Remove values above the limit (cGy) if specified
        nbins = int(maxdose / bin_size)
        hist = np.zeros(nbins)

        # Calculate the histogram for each contour
        calculated_z = []
        # TODO DEBUG DVH LOOP TO GET THE LARGEST INDEX using orginal implementation SPLANES
        volume = 0
        n_voxels = []
        st = time.time()
        for i, contour in enumerate(contours):
            z = contour['z']
            if z in calculated_z:
                print('Repeated slice z', z)
                continue
            print('calculating slice z', z)

            doseplane = dose_interp((z, yy, xx))
            # If there is no dose for the current plane, go to the next plane
            if not len(doseplane):
                break
            m = get_contour_mask_wn(dose_lut, dosegrid_points, contour['data'])
            h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)
            hist += h
            volume += vol
            # volume += contour['area'] * grid_delta[2]  # Area * delta_z
            calculated_z.append(z)

            # Multiply the structure mask by the dose plane to get the dose mask
            mask = ma.array(doseplane, mask=~m)

            n_voxels.append(len(mask.compressed()))

        volume /= 1000
        # Rescale the histogram to reflect the total volume
        hist = hist * volume / sum(hist)
        # Remove the bins above the max dose for the structure

        chist = get_cdvh_numba(hist)
        dhist = (np.arange(0, nbins) / nbins) * maxdose

        end = time.time()

        idx = np.nonzero(chist)  # remove 0 volumes from DVH

        self.dvh = chist[idx]
        print('elapsed (s):', end - st)
        print('number of structure voxels: %i' % np.sum(n_voxels))
        print(' ----- END DVH Calculation -----')
        return dhist[idx], chist[idx]

    @lazyproperty
    def smoothed_dvh(self):
        window_size = int(len(self.dvh) / 10)
        if window_size % 2 == 0:
            window_size += 1

        return savgol_filter(self.dvh, window_size, 3)


if __name__ == '__main__':
    # TODO IMPLEMENT DVH CALCULATION
    import pandas as pd

    sheet = 'Sphere'
    df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx', sheetname=sheet)
    adose = df['Dose (cGy)'].values
    advh = df['SI 3 mm'].values

    rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_30_0.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_02_0.dcm'
    # rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RS.PQRT END TO END.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_10_0.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/Cone_10_0.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_30_0.dcm'
    # rs_file = r'D:\Dropbox\Plan_Competition_Project\FantomaPQRT\RS.PQRT END TO END.dcm'

    rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm'
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm'
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm'
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm'
    # rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'
    # rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'

    dose = ScoringDicomParser(filename=rd_file)
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()

    st = 2

    structure = structures[st]
    sPlanes = structure['planes']
    dicompyler_dvh = get_dvh(structure, dose)

    up = (1, 1, 1)
    struc_teste = Structure(structure)
    dhist, chist = struc_teste.calculate_dvh(dose, bin_size=1, upsample=True, delta_cm=up)

    tbp = int(len(chist) / 3)
    if tbp % 2 == 0:
        tbp += 1

    shist = savitzky_golay(chist, tbp, 3)

    plt.plot(dhist, shist, label='smooth')
    plt.plot(adose, advh, label='Analytical')
    plt.plot(dhist, chist, label='Up-sampled (%1.1f, %1.1f, %1.1f) ' % up)
    plt.hold(True)
    # plt.title(structure['name'] + ' volume: %1.1f' % volume)
    plt.plot(dicompyler_dvh['data'], label='Original')
    plt.legend()
    plt.show()

#
# from dicomparser import ScoringDicomParser
# from dvhcalc import get_dvh
# from dev.dvhcalculation import Structure
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import numba as nb
# import cv2
# from math import factorial
# import numpy
#
# if __name__ == '__main__':
#
#     rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RS.PQRT END TO END.dcm'
#     rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'
#     #
#     # rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/Eclipse Plans/Venessa IMRT Eclipse/RD-Eclipse-Venessa-IMRTDose.dcm'
#     # rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/Competition Package/DICOM Sets/RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
#
#     dose = ScoringDicomParser(filename=rd_file)
#     struc = ScoringDicomParser(filename=rs_file)
#     structures = struc.GetStructures()
#
#     ecl_DVH = dose.GetDVHs()
#
#     up = (2, 2, 0.5)
#     st = time.time()
#     for structure in structures.values():
#         if structure['id'] in ecl_DVH:
#             if not structure['name'] == 'CORPO':
#                 dicompyler_dvh = get_dvh(structure, dose)
#                 struc_teste = Structure(structure)
#
#                 ecl_dvh = ecl_DVH[structure['id']]['data']
#                 dhist, chist = struc_teste.calculate_dvh(dose, bin_size=1, delta_cm=up)
#                 plt.figure()
#                 plt.plot(chist, label='Up Sampled Structure')
#                 plt.hold(True)
#                 # plt.plot(adose, advh)
#                 plt.plot(ecl_dvh, label='Eclipse DVH')
#                 plt.title(structure['name'] + ' volume (cc): %1.3f' % ecl_dvh[0])
#                 plt.plot(dicompyler_dvh['data'], label='Not up sampled')
#                 # plt.plot(struc_teste.smoothed_dvh, label='smoothed_dvh')
#                 plt.legend(loc='best')
#     end = time.time()
# 
#     print('Total elapsed Time (min):  ', (end - st) / 60)
#     plt.show()
#
#     # TODO STRUCTURE EXTRAPOLATION
#
#     plane = struc_teste.planes[3]
#
#     contours, p = calculate_planes_contour_areas(struc_teste.planes)
#
#     tmp = np.zeros((plane.shape[0] + 1, plane.shape[1]))
#     tmp[:-1] = plane
#     tmp[-1] = plane[0]
#
#     x = tmp[:, 0]
#     y = tmp[:, 1]
#
#     x1 = struc_teste.planes[3][:, 0]
#     y1 = struc_teste.planes[3][:, 1]
#
#     calc_area(x, y)
#     poly_area(x, y)
#     calc_area(x1, y1)
#     poly_area(x1, y1)
