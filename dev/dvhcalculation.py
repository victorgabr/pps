import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import time
from dev.geometry import get_contour_mask_wn, check_contour_inside, get_structure_planes, \
    expand_roi, calculate_planes_contour_areas, get_dose_grid_3d, get_z_planes, get_axis_grid, get_dose_grid
from dicomparser import ScoringDicomParser, lazyproperty
from dvhcalc import get_cdvh_numba, get_dvh, calculate_contour_areas

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
    def __init__(self, struc):
        self.structure = struc
        self.contour_spacing = struc['thickness']
        self.contours, self.largestIndex = None, None
        self.grid_spacing = np.zeros(3)
        self.dose_lut = None
        self.dose_grid_points = None
        self.hi_res_structure = None

    @lazyproperty
    def planes(self):
        return get_structure_planes(self.structure)

    def get_expanded_roi(self, delta_mm):
        """
            Expand roi by 1/2 structure thickness in x,y and z axis
        """
        return expand_roi(self.planes, delta=delta_mm)

    @lazyproperty
    def planes_expanded(self):
        return expand_roi(self.planes, self.contour_spacing / 2.0)

    def set_contours(self, expanded=False):
        if expanded:
            self.contours, self.largestIndex = calculate_planes_contour_areas(self.planes)
        else:
            self.contours, self.largestIndex = calculate_planes_contour_areas(self.planes_expanded)

    def up_sampling(self, lut_grid_3d, delta_mm=(0.2, 0.2, 0.2), expanded=False):
        if expanded:
            planes = self.planes_expanded
        else:
            planes = self.planes

        # get structure slice position
        ordered_z = np.unique(np.concatenate(planes)[:, 2])

        # ROI UP SAMPLING IN X, Y, Z
        self.dose_grid_points, self.dose_lut, self.grid_spacing = get_dose_grid_3d(lut_grid_3d, delta_mm)
        print('spacing before: ', self.grid_spacing)

        zi, dz = get_axis_grid(self.grid_spacing[2], ordered_z)
        self.grid_spacing[2] = dz
        print('spacing after: ', self.grid_spacing)

        self.hi_res_structure = get_z_planes(planes, ordered_z, zi)

        return self.hi_res_structure, self.dose_grid_points, self.grid_spacing, self.dose_lut

    def calculate_dvh(self, dose, bin_size=1, up_sampling=False, delta_cm=(.5, .5, .5)):
        grid_3d = dose.get_grid_3d()
        # ROI UP SAMPLING IN X, Y, Z

        if up_sampling:
            ds, grid_ds, grid_delta, dose_lut = self.up_sampling(grid_3d, delta_cm)
            dosegrid_points = grid_ds[:, :2]
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

        contours, largest_index = calculate_planes_contour_areas(ds)
        # largest_index = 0
        dose_interp, values = dose.DoseRegularGridInterpolator()
        xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)

        # Create an empty array of bins to store the histogram in cGy
        # only if the structure has contour data or the dose grid exists
        dd = dose.GetDoseData()
        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
        # Remove values above the limit (cGy) if specified
        nbins = int(maxdose / bin_size)
        hist = np.zeros(nbins)

        volume = 0
        n_voxels = []
        # Calculate the histogram for each contour
        calculated_z = []
        # TODO DEGUB DVH LOOP TO GET THE LARGEST INDEX using orginal implementation SPLANES
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
            # plt.imshow(m)
            # plt.show()
            h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)
            hist += h
            volume += vol
            calculated_z.append(z)


            # i = 0
            # largest_index = 0
            n_voxels.append(np.size(m) - np.count_nonzero(m))

            # print(largest_index)
            # TEST TIMING
            # if z > 1.0:A
            #     break
            # If this is the largest contour, just add to the total histogram

            #
            # if i == largest_index:
            #     hist += h
            #     volume += vol
            # # Otherwise, determine whether to add or subtract histogram
            # # depending if the contour is within the largest contour or not
            # else:
            #     print('non_largest')
            #     truth = check_contour_inside(contour['data'], contours[largest_index]['data'])
            #     # If the contour is inside, subtract it from the total histogram
            #     if truth:
            #         hist -= h
            #         volume -= vol
            #     # Otherwise it is outside, so add it to the total histogram
            #     else:
            #         hist += h
            #         volume += vol


        # if not (callback is None):
        #     callback(plane, len(sPlanes))
        # Volume units are given in cm^3
        volume /= 1000

        # Rescale the histogram to reflect the total volume
        hist = hist * volume / sum(hist)
        # Remove the bins above the max dose for the structure
        # hist = np.trim_zeros(hist, trim='b')
        print('number of structure voxels: %i' % np.sum(n_voxels))
        chist = get_cdvh_numba(hist)
        # chist = np.trim_zeros(chist, trim='b')
        dhist = np.linspace(0, maxdose, nbins)
        end = time.time()
        print('elapsed (s):', end - st)
        return dhist, chist


if __name__ == '__main__':
    # TODO IMPLEMENT DVH CALCULATION
    import pandas as pd

    sheet = 'RT_Cone'
    df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx', sheetname=sheet)
    adose = df['Dose (cGy)'].values
    advh = df['AP 3 mm'].values

    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_10_0.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_02_0.dcm'
    # rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RS.PQRT END TO END.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_10_0.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/Cone_10_0.dcm'
    rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_30_0.dcm'
    # rs_file = r'D:\Dropbox\Plan_Competition_Project\FantomaPQRT\RS.PQRT END TO END.dcm'

    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_2mm_Aligned.dcm'
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm'
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm'
    rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm'
    # rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'
    # rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'


    dose = ScoringDicomParser(filename=rd_file)
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()
    # ecl_dvh = dose.GetDVHs()[2]['data']

    st = 2
    structure = structures[st]
    sPlanes = structure['planes']
    dicompyler_dvh = get_dvh(structure, dose)

    struc_teste = Structure(structure)
    dhist, chist = struc_teste.calculate_dvh(dose, up_sampling=True, delta_cm=(0.2, 0.2, 0.2))
    # dhist, chist = struc_teste.calculate_dvh(dose)
    # plt.plot(dhist, np.abs(chist), '.')
    plt.plot(dhist, np.abs(chist))
    # plt.plot(np.abs(chist))
    plt.hold(True)
    plt.plot(adose, advh)
    # plt.plot(ecl_dvh, '*')
    # plt.title(structure['name'] + ' volume: %1.1f' % volume)
    plt.plot(dicompyler_dvh['data'])
    plt.show()
    grid_3d = dose.get_grid_3d()

    factor = 2
    planes = struc_teste.planes_expanded
    lut_grid_3d = grid_3d
    ordered_z = np.unique(np.concatenate(planes)[:, 2])

    # ROI UP SAMPLING IN X, Y, Z

    # ds, grid_ds, grid_delta, dose_lut = struc_teste.up_sampling(grid_3d, (.2, .2, .2))

    contours1, largest_index = calculate_planes_contour_areas(struc_teste.planes)

    # Iterate over each plane in the structure
    ctr = []
    for z, sPlane in sPlanes.items():
        # Get the contours with calculated areas and the largest contour index
        contours, largestIndex = calculate_contour_areas(sPlane)
        ctr.append((contours, largestIndex))


        # dosegrid_points = grid_ds[:, :2]
        # # dose_lut = [grid_ds[:, 0], grid_ds[:, 1]]
        # dz = grid_delta[2]
        #
        # dose_interp, values = dose.DoseRegularGridInterpolator()
        # xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)
        #
        # # Create an empty array of bins to store the histogram in cGy
        # # only if the structure has contour data or the dose grid exists
        # # if (len(ds)) and ("PixelData" in dose.ds):
        # # Get the dose and image data information
        # dd = dose.GetDoseData()
        # id = dose.GetImageData()
        # maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
        # # Remove values above the limit (cGy) if specified
        # nbins = int(maxdose / 1)
        # hist = np.zeros(nbins)
        #
        # volume = 0
        # plane = 0
        # n_voxels = []
        # # Calculate the histogram for each contour
        # calculated_z = []
        #
        #

        #
        # import time
        #
        # st = time.time()
        # for i, contour in enumerate(contours):
        #     z = contour['z']
        #     if z in calculated_z:
        #         print('Repeated slice z', z)
        #         continue
        #     print('calculating slice z', z)
        #
        #     doseplane = dose_interp((z, yy, xx))
        #     # If there is no dose for the current plane, go to the next plane
        #     if not len(doseplane):
        #         continue
        #
        #     m = get_contour_mask_wn(dose_lut, dosegrid_points, contour['data'])
        #     h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)
        #
        #     n_voxels.append(np.size(m) - np.count_nonzero(m))
        #
        #     # TEST TIMING
        #     # if z > 1.0:
        #     #     break
        #
        #     # If this is the largest contour, just add to the total histogram
        #     if i == largestIndex:
        #         hist += h
        #         volume += vol
        #     # Otherwise, determine whether to add or subtract histogram
        #     # depending if the contour is within the largest contour or not
        #     else:
        #         truth = check_contour_inside(contour['data'], contours[largestIndex]['data'])
        #         # If the contour is inside, subtract it from the total histogram
        #         if truth:
        #             hist -= h
        #             volume -= vol
        #         # Otherwise it is outside, so add it to the total histogram
        #         else:
        #             hist += h
        #             volume += vol
        #     calculated_z.append(z)
        #     plane += 1
        #
        # # if not (callback is None):
        # #     callback(plane, len(sPlanes))
        #
        # # Volume units are given in cm^3
        # volume /= 1000
        # # Rescale the histogram to reflect the total volume
        # hist = hist * volume / sum(hist)
        # # Remove the bins above the max dose for the structure
        # # hist = np.trim_zeros(hist, trim='b')
        #
        # print('number of structure voxels: %i' % np.sum(n_voxels))
        # # tst = get_cdvh(hist)
        #
        # chist = get_cdvh_numba(hist)
        # dhist = np.linspace(0, maxdose, nbins)
        # end = time.time()
        # print('elapsed (s):', end - st)
        #
        #
        # plt.plot(dhist, np.abs(chist))
        # plt.hold(True)
        # plt.plot(adose, advh)
        # plt.title(structure['name'] + ' volume: %1.1f' % volume)
        # plt.plot(dicompyler_dvh['data'])
        # plt.show()
        #
        # # ntests = -1
        # contour1 = contour['data']
        # poly = contour1
