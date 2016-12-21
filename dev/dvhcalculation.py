from matplotlib.path import Path
from numba import jit, njit
from scipy.interpolate import RegularGridInterpolator

from dicomparser import ScoringDicomParser
import numpy as np
import numpy.ma as ma
from dvhcalc import get_contour_mask, get_cdvh, get_cdvh_numba, get_dvh
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

'''

http://dicom.nema.org/medical/Dicom/2016b/output/chtml/part03/sect_C.8.8.html

'''


def poly_area(x, y):
    """
         Calculate the area based on the Surveyor's formula
    :param x: x-coordinate
    :param y: y-coordinate
    :return: polygon area
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calculate_contour_dvh(mask, doseplane, bins, maxdose, dd, id, dz):
    """Calculate the differential DVH for the given contour and dose plane."""

    # Multiply the structure mask by the dose plane to get the dose mask
    mask = ma.array(doseplane * dd['dosegridscaling'] * 100, mask=~mask)
    # Calculate the differential dvh
    hist, edges = np.histogram(mask.compressed(),
                               bins=bins,
                               range=(0, maxdose))

    # Calculate the volume for the contour for the given dose plane
    vol = sum(hist) * (id['pixelspacing'][0] * id['pixelspacing'][1] * dz)

    return hist, vol


def calculate_planes_contour_areas(planes):
    """Calculate the area of each contour for the given plane.
       Additionally calculate and return the largest contour index."""
    # Calculate the area for each contour in the current plane
    contours = []
    largest = 0
    largestIndex = 0
    for c, contour in enumerate(planes):
        x = contour[:, 0]
        y = contour[:, 1]
        z = contour[0, 2]

        # Calculate the area based on the Surveyor's formula
        cArea = poly_area(x, y)
        # Remove the z coordinate from the xyz point tuple
        data = list(map(lambda x: x[0:2], contour[:, :2]))

        # Add the contour area and points to the list of contours
        contours.append({'area': cArea, 'data': np.squeeze(data), 'z': z})
        # Determine which contour is the largest
        if cArea > largest:
            largest = cArea
            largestIndex = c

    return contours, largestIndex


def InterpolateDosePlanes(uplane, lplane, fz):
    """Interpolates a dose plane between two bounding planes at the given relative location."""

    # uplane and lplane are the upper and lower dose plane, between which the new dose plane
    #   will be interpolated.
    # fz is the fractional distance from the bottom to the top, where the new plane is located.
    #   E.g. if fz = 1, the plane is at the upper plane, fz = 0, it is at the lower plane.

    # A simple linear interpolation
    doseplane = fz * uplane + (1.0 - fz) * lplane

    return doseplane


def interpolate_plane(ub, lb, location, ubpoints, lbpoints):
    """Interpolates a plane between two bounding planes at the given location."""

    # If the number of points in the upper bound is higher, use it as the starting bound
    # otherwise switch the upper and lower bounds
    # if not (len(ubpoints) >= len(lbpoints)):
    #     lbCopy = lb
    #     lb = ub
    #     ub = lbCopy

    plane = []
    # Determine the closest point in the lower bound from each point in the upper bound
    for u, up in enumerate(ubpoints):
        dist = 100000  # Arbitrary large number
        # Determine the distance from each point in the upper bound to each point in the lower bound
        for l, lp in enumerate(lbpoints):
            newDist = np.sqrt((up[0] - lp[0]) ** 2 + (up[1] - lp[1]) ** 2 + (ub - lb) ** 2)
            # If the distance is smaller, then linearly interpolate the point
            if newDist < dist:
                dist = newDist
                x = lp[0] + (location - lb) * (up[0] - lp[0]) / (ub - lb)
                y = lp[1] + (location - lb) * (up[1] - lp[1]) / (ub - lb)
        if not (dist == 100000):
            plane.append([x, y, location])

    return np.squeeze(plane)


@njit
def interpolate_plane_numba(ub, lb, location, ubpoints, lbpoints):
    """Interpolates a plane between two bounding planes at the given location."""

    # If the number of points in the upper bound is higher, use it as the starting bound
    # otherwise switch the upper and lower bounds
    # if not (len(ubpoints) >= len(lbpoints)):
    #     lbCopy = lb
    #     lb = ub
    #     ub = lbCopy
    tmp = np.zeros(3)
    plane = np.zeros((len(ubpoints), 3))
    # Determine the closest point in the lower bound from each point in the upper bound
    # for u, up in enumerate(ubpoints):
    for u in range(len(ubpoints)):
        up = ubpoints[u]
        dist = 10000000  # Arbitrary large number
        # Determine the distance from each point in the upper bound to each point in the lower bound
        for l in range(len(lbpoints)):
            lp = lbpoints[l]
            newDist = np.sqrt((up[0] - lp[0]) ** 2 + (up[1] - lp[1]) ** 2 + (ub - lb) ** 2)
            # If the distance is smaller, then linearly interpolate the point
            if newDist < dist:
                dist = newDist
                x = lp[0] + (location - lb) * (up[0] - lp[0]) / (ub - lb)
                y = lp[1] + (location - lb) * (up[1] - lp[1]) / (ub - lb)
                tmp[0] = x
                tmp[1] = y
                tmp[2] = location
        if not (dist == 10000000):
            plane[u] = tmp

    return plane


def interp_structure_planes(structure_dict, n_planes=5, verbose=False):
    """
        Interpolates all structures planes inserting interpolated planes centered exactly between
    the original dose plane locations (sorted by z)

    :param structure_dict: RS structure dict object
    :param n_planes: Number of planes to be inserted
    :return: list containing
    """

    # TODO IMPLEMENT ROI SUPERSAMPLING IN X Y Z

    sPlanes = structure_dict['planes']
    dz = structure_dict['thickness'] / 2

    ## INTERPOLATE PLANES IN Z AXIS
    # Iterate over each plane in the structure
    zval = [z for z, sPlane in sPlanes.items()]
    zval.sort(key=float)

    structure_planes = []
    for z in zval:
        plane_i = sPlanes[z]
        structure_planes.append(np.array(plane_i[0]['contourData']))

    # extending a start-end cap slice
    # extending end cap slice by 1/2 CT slice thickness
    start_cap = structure_planes[0].copy()
    start_cap[:, 2] = start_cap[:, 2] - dz

    # extending end cap slice by 1/2 CT slice thickness
    end_cap = structure_planes[-1].copy()
    end_cap[:, 2] = end_cap[:, 2] + dz

    # extending end caps to original plans
    # structure_planes = [start_cap] + structure_planes + [end_cap]
    structure_planes[0] = start_cap
    structure_planes[-1] = end_cap

    # TODO to estimate number of interpolated planes to reach ~ 30000 voxels

    result = []
    result += [structure_planes[0]]
    for i in range(len(structure_planes) - 1):
        ub = structure_planes[i + 1][0][2]
        lb = structure_planes[i][0][2]
        loc = np.linspace(lb, ub, num=n_planes + 2)
        loc = loc[1:-1]
        ubpoints = structure_planes[i + 1]
        lbpoints = structure_planes[i]
        interp_planes = []
        if verbose:
            print('bounds', lb, ub)
            print('interpolated planes: ', loc)

        if not (len(ubpoints) >= len(lbpoints)):
            # if upper bounds does not have more points, swap planes to interpolate
            lbCopy = lb
            lb = ub
            ub = lbCopy
            ubpoints = structure_planes[i]
            lbpoints = structure_planes[i + 1]

        for l in loc:
            pi = interpolate_plane_numba(ub, lb, l, ubpoints, lbpoints)
            interp_planes.append(pi)
        result += interp_planes + [ubpoints]

    # adding last slice to result
    result += [structure_planes[-1]]
    # return planes sorted by z-axis position

    return sorted(result, key=lambda p: p[0][2])


def plot_planes(planes, color='r', marker='_'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c in planes:
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=color, marker=marker)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


if __name__ == '__main__':
    rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_30_0.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_20_0.dcm'
    # rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_10_0.dcm'
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()
    st = 2
    structure = structures[st]
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm'
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm'
    rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'
    # DVH ORIGINAL
    rtss = ScoringDicomParser(filename=rs_file)
    rtdose = ScoringDicomParser(filename=rd_file)
    dv = get_dvh(structure, rtdose)

    dose = ScoringDicomParser(filename=rd_file)

    # TODO TEST 3D interpolation

    # Get the dose to pixel LUT
    doselut = dose.GetPatientToPixelLUT()

    x = doselut[0]
    y = doselut[1]

    # UPSAMPLING
    xx = np.linspace(doselut[0][0], doselut[0][-1], 1024)
    yy = np.linspace(doselut[1][0], doselut[1][-1], 1024)

    #
    # for i in range(values.shape[0]):
    #     plt.imshow(values[i, :, :])
    #     plt.title('index: %i , position: %i' % (i, z[i]))
    #     plt.show()

    #
    my_interpolating_function, values = dose.DoseRegularGridInterpolator()

    # GENERATE MESH XY TO GET INTERPOLATED PLANE
    xx, yy = np.meshgrid(xx, yy, indexing='xy', sparse=True)
    res = my_interpolating_function((0.8, yy, xx))
    plt.imshow(res)
    plt.title('interpolated')
    plt.figure()
    original = values[41, :, :]
    plt.imshow(original)
    plt.title('original')
    plt.show()
#
#     planes_total = interp_structure_planes(structure, 25)
#
#     # ## TODO ENCAPSULATE DVH CALCULATION USING STRUCTURE UPSAMPLING
#     #
#     # CHECK PLANES DATA
#
#
#
#     # Get the contours with calculated areas and the largest contour index
#     contours, largestIndex = calculate_planes_contour_areas(planes_total)
#
#     ordered_z = np.array([i['z'] for i in contours])
#
#     # interpolated structure tickness
#     delta = np.diff(ordered_z)
#     delta_z = ordered_z.copy()
#     delta_z[:-1] = delta
#     delta_z[-1] = delta[-1]
#
#     # # plot_planes(planes_orig, 'r', '^')
#     # # plt.title('ORIGINAL')
#     # plot_planes(planes_total, 'r', '^')
#     # plt.title('interpolated')
#     # plt.show()
#
#     limit = None
#     sPlanes = structure['planes']
#
#     # Get the dose to pixel LUT
#     doselut = dose.GetPatientToPixelLUT()
#
#     xx = doselut[0]
#     yy = doselut[1]
#
#     # TODO SUPERSAMPING XY AXIS
#     # xx = np.linspace(doselut[0][0], doselut[0][-1])
#     # yy = np.linspace(doselut[1][0], doselut[1][-1])
#     #
#     # doselut = [xx, yy]
#
#     # Generate a 2d mesh grid to create a polygon mask in dose coordinates
#     # Code taken from Stack Overflow Answer from Joe Kington:
#     # http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/3655582
#     # Create vertex coordinates for each grid cell
#     x, y = np.meshgrid(xx, yy)
#     x, y = x.flatten(), y.flatten()
#     dosegridpoints = np.vstack((x, y)).T
#
#     # Create an empty array of bins to store the histogram in cGy
#     # only if the structure has contour data or the dose grid exists
#     if (len(sPlanes)) and ("PixelData" in dose.ds):
#         # Get the dose and image data information
#         dd = dose.GetDoseData()
#     id = dose.GetImageData()
#     maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
#     # Remove values above the limit (cGy) if specified
#     nbins = int(maxdose / 1)
#     hist = np.zeros(nbins)
#
#     volume = 0
#     plane = 0
#
#     n_voxels = []
#     # Calculate the histogram for each contour
#     calculated_z = []
#     for i, contour in enumerate(contours):
#         dz = delta_z[i]
#     z = contour['z']
#     if z in calculated_z:
#         print('Repeated slice z', z)
#     continue
#     print('calculating slice z', z)
#     doseplane = dose.GetDoseGrid(z, threshold=0.0)
#     # If there is no dose for the current plane, go to the next plane
#     if not len(doseplane):
#         break
#
#     m = get_contour_mask(doselut, dosegridpoints, contour['data'])
#
#     h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, dd, id, dz)
#
#     n_voxels.append(np.size(m) - np.count_nonzero(m))
#     # plt.imshow(mask)
#     # plt.show()
#
#     # If this is the largest contour, just add to the total histogram
#     if i == largestIndex:
#         hist += h
#         volume += vol
#     # Otherwise, determine whether to add or subtract histogram
#     # depending if the contour is within the largest contour or not
#     else:
#         contour['inside'] = False
#         for point in contour['data']:
#             p = Path(np.array(contours[largestIndex]['data']))
#             if p.contains_point(point):
#                 contour['inside'] = True
#                 # Assume if one point is inside, all will be inside
#                 break
#         # If the contour is inside, subtract it from the total histogram
#         if contour['inside']:
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
# import pandas as pd
#
# df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx')
# adose = df['Dose (cGy)'].values
# advh = df['SI 3 mm'].values
# plt.plot(dhist, np.abs(chist))
# plt.hold(True)
# plt.plot(adose, advh)
# # plt.plot(chist)
# plt.title(structure['name'] + ' volume: %1.1f' % volume)
# plt.plot(dv['data'])
#
# plt.show()  # for c, contour in enumerate(sPlane):
# #     # Create arrays for the x,y coordinate pair for the triangulation
# #     x = []
# #     y = []
# #     for point in contour['contourData']:
# #         x.append(point[0])
# #         y.append(point[1])
