from collections import OrderedDict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from pyplanscoring.dev.dvhcalculation import Structure, get_boundary_stats
from pyplanscoring.dev.geometry import wrap_z_coordinates, calc_area, get_contour_roi_grid, wrap_xy_coordinates, \
    get_contour_mask_wn, check_contour_inside


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
        return grad

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
        calculate both expanded and shrunken contours by a distance in mm
       Additionally calculate and return the largest contour index.
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


class StructurePaper(Structure):
    def __init__(self, dicom_structure, end_cap=False):
        Structure.__init__(dicom_structure, end_cap)

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


if __name__ == '__main__':
    pass
