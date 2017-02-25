from __future__ import division

import functools
from copy import deepcopy
from math import factorial

import numba as nb
import numpy as np
from scipy.interpolate import interp1d

# add fast-math
if int(nb.__version__.split('.')[1]) >= 31:
    njit = functools.partial(nb.njit, fastmath=True, cache=True, nogil=True)
else:
    njit = nb.njit(cache=True, nogil=True)


def cn_PnPoly(P, V):
    cn = 0  # the crossing number counter

    # repeat the first vertex at end
    V = tuple(V[:]) + (V[0],)

    # loop through all edges of the polygon
    for i in range(len(V) - 1):  # edge from V[i] to V[i+1]
        if ((V[i][1] <= P[1] and V[i + 1][1] > P[1])  # an upward crossing
            or (V[i][1] > P[1] and V[i + 1][1] <= P[1])):  # a downward crossing
            # compute the actual edge-ray intersect x-coordinate
            vt = (P[1] - V[i][1]) / float(V[i + 1][1] - V[i][1])
            if P[0] < V[i][0] + vt * (V[i + 1][0] - V[i][0]):  # P[0] < intersect
                cn += 1  # a valid crossing of y=P[1] right of P[0]

    return cn % 2  # 0 if even (out), and 1 if odd (in)


# ===================================================================


def wn_PnPoly1(P, V):
    """
        # point_in_contour(): winding number test for a point in a polygon
     Input:  P = a point,
             V[] = vertex points of a polygon
     Return: wn = the winding number (=0 only if P is outside V[])

    :param P: a point [x,y] array
    :param V: V[] = vertex points of a polygon
    :return:  winding number test for a point in a polygon
    """
    wn = 0  # the winding number counter

    # repeat the first vertex at end
    V = tuple(V[:]) + (V[0],)

    # loop through all edges of the polygon
    for i in range(len(V) - 1):  # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:  # start y <= P[1]
            if V[i + 1][1] > P[1]:  # an upward crossing
                if is_left(V[i], V[i + 1], P) > 0:  # P left of edge
                    wn += 1  # have a valid up intersect
        else:  # start y > P[1] (no test needed)
            if V[i + 1][1] <= P[1]:  # a downward crossing
                if is_left(V[i], V[i + 1], P) < 0:  # P right of edge
                    wn -= 1  # have a valid down intersect
    return wn


@njit(nb.double(nb.double[:], nb.double[:], nb.double[:]))
def is_left(p0, p1, p2):
    """

       is_left(): tests if a point is Left|On|Right of an infinite line.
    Input:  three points P0, P1, and P2
    Return: >0 for P2 left of the line through P0 and P1
            =0 for P2  on the line
            <0 for P2  right of the line
        See: Algorithm 1 "Area of Triangles and Polygons"
        http://geomalgorithms.com/a03-_inclusion.html

    :param p0: point [x,y] array
    :param p1: point [x,y] array
    :param p2: point [x,y] array
    :return:
    """
    v = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    return v


@njit(nb.int64(nb.double[:], nb.double[:, :]))
def point_in_contour(P, polygon):
    """

    :param P:
    :param polygon:
    :return:
    """
    wn = 0  # the  winding number counter
    # repeat the first vertex at end
    V = np.zeros((polygon.shape[0] + 1, polygon.shape[1]))
    V[:-1] = polygon
    V[-1] = polygon[0]
    n = len(V)
    #  loop through all edges of the polygon
    for i in range(n - 1):  # edge from V[i] to  V[i+1]
        if V[i][1] <= P[1]:  # start y <= P[1]
            if V[i + 1][1] > P[1]:  # an upward crossing
                if is_left(V[i], V[i + 1], P) > 0:  # P left of  edge
                    wn += 1  # // have  a valid up intersect

        else:  # start y > P[1] (no test needed)
            if V[i + 1][1] <= P[1]:  # a downward crossing
                if is_left(V[i], V[i + 1], P) < 0:  # P right of  edge
                    wn -= 1  # have  a valid down intersect
    return wn


@njit(nb.boolean[:](nb.boolean[:], nb.double[:, :], nb.double[:, :]))
def wn_contains_points(out, poly, points):
    """
        Winding number test for a list of point in a polygon
        Numba implementation 8 - 10 x times faster than Matplotlib Path.contains_points()
    :param out: output boolean array
    :param poly: polygon (list of points/vertex)
    :param points: list of points to check inside polygon
    :return: Boolean array
        adapted from c++ code at:
        http://geomalgorithms.com/a03-_inclusion.html

    """
    n = len(points)

    for i in range(n):
        point = points[i]
        wn = 0  # the  winding number counter
        N = len(poly)
        # // loop through all edges of the polygon
        for k in range(N - 1):  # edge from V[i] to  V[i+1]

            if poly[k][1] <= point[1]:  # start y <= P[1]
                if poly[k + 1][1] > point[1]:  # an upward crossing
                    is_left_value = is_left(poly[k], poly[k + 1], point)
                    if is_left_value >= 0:  # P left of  edge
                        wn += 1  # // have  a valid up intersect

            else:  # start y > P[1] (no test needed)
                if poly[k + 1][1] <= point[1]:  # a downward crossing
                    is_left_value = is_left(poly[k], poly[k + 1], point)
                    if is_left_value <= 0:  # P right of  edge
                        wn -= 1  # have  a valid down intersect

            out[i] = wn

    return out


@njit(nb.boolean(nb.double, nb.double, nb.double[:, :]))
def point_inside_polygon(x, y, poly):
    n = len(poly)
    # determine if a point is inside a given polygon or not
    # Polygon is a list of (x,y) pairs.
    p1x = 0.0
    p1y = 0.0
    p2x = 0.0
    p2y = 0.0
    xinters = 0.0
    plx = 0.0
    ply = 0.0
    idx = 0
    inside = False

    # p1x, p1y = poly[0]
    p1x = poly[0][0]
    p1y = poly[0][1]

    for i in range(n + 1):
        idx = i % n
        p2x = poly[idx][0]
        p2y = poly[idx][1]
        # p2x, p2y = poly[idx]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x = p2x
        p1y = p2y
        # p1x, p1y = p2x, p2y

    return inside


@njit(nb.boolean[:](nb.boolean[:], nb.double[:, :], nb.double[:, :]))
def contains_points(out, poly, points):
    n = len(points)
    for i in range(n):
        point = points[i]
        tmp = point_inside_polygon(point[0], point[1], poly)
        out[i] = tmp
    return out


@njit(nb.boolean[:](nb.boolean[:], nb.double[:, :], nb.double[:, :]))
def numba_contains_points(out, poly, points):
    n = len(points)
    p1x = 0.0
    p1y = 0.0
    p2x = 0.0
    p2y = 0.0
    xinters = 0.0
    plx = 0.0
    ply = 0.0
    idx = 0
    inside = False
    x = 0
    y = 0
    N = len(poly)

    for i in range(n):
        point = points[i]
        x = point[0]
        y = point[1]
        # tmp = point_inside_polygon(point[0], point[1], poly)
        inside = False

        # determine if a point is inside a given polygon or not
        # Polygon is a list of (x,y) pairs.
        p1x = poly[0][0]
        p1y = poly[0][1]
        for j in range(N + 1):
            idx = j % N
            p2x = poly[idx][0]
            p2y = poly[idx][1]
            # p2x, p2y = poly[idx]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside

            p1x = p2x
            p1y = p2y

        out[i] = inside

    return out


def get_contour_mask_wn(doselut, dosegrid_points, poly):
    """
        Get the mask for the contour with respect to the dose plane.
    :param doselut: Dicom 3D dose LUT (x,y)
    :param dosegrid_points: dosegrid_points
    :param poly: contour
    :return: contour mask on grid
    """

    n = len(dosegrid_points)
    out = np.zeros(n, dtype=bool)
    # preparing data to wn test
    # repeat the first vertex at end
    poly_wn = np.zeros((poly.shape[0] + 1, poly.shape[1]))
    poly_wn[:-1] = poly
    poly_wn[-1] = poly[0]

    grid = wn_contains_points(out, poly_wn, dosegrid_points)
    grid = grid.reshape((len(doselut[1]), len(doselut[0])))

    return grid


def poly_area(x, y):
    """
         Calculate the area based on the Surveyor's formula
    :param x: x-coordinate
    :param y: y-coordinate
    :return: polygon area-
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@njit
def centroid_of_polygon(x, y):
    """
        http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    :param x: x-axis coordinates
    :param y: y-axis coordinates
    :return: centroid of polygon
    """
    area = calc_area(x, y)
    imax = len(x) - 1

    result_x = 0
    result_y = 0
    for i in range(0, imax):
        result_x += (x[i] + x[i + 1]) * ((x[i] * y[i + 1]) - (x[i + 1] * y[i]))
        result_y += (y[i] + y[i + 1]) * ((x[i] * y[i + 1]) - (x[i + 1] * y[i]))

    result_x += (x[imax] + x[0]) * ((x[imax] * y[0]) - (x[0] * y[imax]))
    result_y += (y[imax] + y[0]) * ((x[imax] * y[0]) - (x[0] * y[imax]))
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)

    return result_x, result_y


@njit(nb.boolean(nb.double[:, :], nb.double[:, :]))
def check_contour_inside(contour, largest):
    inside = False
    for i in range(len(contour)):
        point = contour[i]
        p = point_in_contour(point, largest)
        if p:
            inside = True
            # Assume if one point is inside, all will be inside
            break
    return inside


def k_nearest_neighbors(k, feature_train, features_query):
    """

    :param k: kn neighbors
    :param feature_train: reference 1D array grid
    :param features_query: query grid
    :return: lower and upper neighbors
    """
    ec_dist = abs(feature_train - features_query)

    if k == 1:
        neighbors = ec_dist.argmin()
    else:
        neighbors = np.argsort(ec_dist)[:k]

    return neighbors


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

        cArea = calc_area(x, y)
        # cArea = poly_area(x, y)

        # Remove the z coordinate from the xyz point tuple
        # data = list(map(lambda x: x[0:2], contour[:, :2]))

        # Add the contour area and points to the list of contours
        contours.append({'area': cArea, 'data': contour[:, :2], 'z': z})
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


def get_dose_grid(dose_lut):
    # Generate a 2d mesh grid to create a polygon mask in dose coordinates
    # Code taken from Stack Overflow Answer from Joe Kington:
    # http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/3655582
    # Create vertex coordinates for each grid cell
    x_lut = dose_lut[0]  # zoom(dose_lut[0], super_sampling_fator)
    y_lut = dose_lut[1]  # , super_sampling_fator)

    x, y = np.meshgrid(x_lut, y_lut)
    x, y = x.flatten(), y.flatten()
    dose_grid_points = np.vstack((x, y)).T

    return dose_grid_points


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


def get_dose_grid_3d(grid_3d, delta_mm=(2, 2, 2)):
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

    x_lut, x_delta = get_axis_grid(delta_mm[0], xi)
    y_lut, y_delta = get_axis_grid(delta_mm[1], yi)
    z_lut, z_delta = get_axis_grid(delta_mm[2], zi)

    xg, yg = np.meshgrid(x_lut, y_lut)
    xf, yf = xg.flatten(), yg.flatten()
    dose_grid_points = np.vstack((xf, yf)).T

    up_dose_lut = [x_lut, y_lut, z_lut]

    spacing = [x_delta, x_delta, z_delta]

    return dose_grid_points, up_dose_lut, spacing


@njit(nb.double(nb.double[:], nb.double[:]))
def calc_area(x, y):
    """
        Calculate the area based on the Surveyor's formula
    :param x: x vertex coordinates array
    :param y: x vertex coordinates array
    :return: Polygon area
    """
    cArea = 0
    xi = np.zeros(len(x) + 1)
    yi = np.zeros(len(y) + 1)
    # Fix by adding end points vertex
    xi[:-1] = x
    xi[-1] = x[0]
    yi[:-1] = y
    yi[-1] = y[0]

    for i in range(0, len(xi) - 1):
        cArea = cArea + xi[i] * yi[i + 1] - xi[i + 1] * yi[i]
    cArea = abs(cArea / 2.0)

    return cArea


@njit(nb.boolean(nb.double[:], nb.double[:], nb.double[:]))
def ccw(a, b, c):
    """Tests whether the turn formed by A, B, and C is ccw"""
    return (b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0])


@njit(nb.boolean(nb.double[:, :]))
def is_convex(points):
    """
    https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation
        Test if a contour of points [xi,yi] - [xn, yn] is convex

    :param points: Array of 2d points
    :return: boolean
    """
    n = len(points)

    for i in range(n):
        # Check every triplet of points
        ia = i % n
        ib = (i + 1) % n
        ic = (i + 2) % n
        a = points[ia]
        b = points[ib]
        c = points[ic]

        if not ccw(a, b, c):
            return False

    return True


def savitzky_golay(y, window_size=501, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise (ValueError("window_size and order have to be of type int"))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def interp_contour(z_plane, ub_z, lb_z, ubound_contour, lb_contour):
    if not (len(ubound_contour) >= len(lb_contour)):
        # if upper bounds does not have more points, swap planes to interpolate
        lbCopy = lb_z
        lb_z = ub_z
        ub_z = lbCopy
        lb_contour_copy = lb_contour.copy()  # copy of original array

        lb_contour = ubound_contour
        ubound_contour = lb_contour_copy

    return interpolate_plane_numba(ub_z, lb_z, z_plane, ubound_contour, lb_contour)


def get_interpolated_structure_planes(dicom_planes, z_interp_positions):
    s_planes = deepcopy(dicom_planes)
    ordered_keys = [z for z, sPlane in s_planes.items()]
    ordered_keys.sort(key=float)
    ordered_z = np.array(ordered_keys, dtype=float)

    interpolated_planes = {}
    for zi in z_interp_positions:
        # TODO FIX interlopation between original planes
        if not np.isclose(zi, ordered_z).any():
            # get grid knn
            u_idx = ordered_z.searchsorted(zi)
            l_idx = u_idx - 1

            # get upper and lower z values and contour points
            ub = ordered_z[u_idx]
            lb = ordered_z[l_idx]

            # get a list of contours per planes
            ub_points = s_planes[ordered_keys[u_idx]]
            lb_points = s_planes[ordered_keys[l_idx]]

            # if 1 contour per slice
            result = []
            truth = len(ub_points) == 1 and len(lb_points) == 1
            if truth:

                lc_contour = lb_points[0]['contourData']
                up_contour = ub_points[0]['contourData']
                interpolated_contour = interp_contour(zi, ub, lb, up_contour, lc_contour)
                result += [{'contourData': interpolated_contour}]
                interpolated_planes[str(zi)] = result

            elif not (truth and len(ub_points) == len(lb_points)):

                lb_centroids = np.asarray([c['centroid'] for c in lb_points])
                ub_centroids = np.asarray([c['centroid'] for c in ub_points])

                # lb_centroids = [c['centroid'] for c in lb_points]
                # ub_centroids = [c['centroid'] for c in ub_points]

                u_idx = [nearest_neighbor(ub_centroids, lbc) for lbc in lb_centroids]

                for j in range(len(lb_points)):
                    lc_contour = lb_points[j]['contourData']
                    up_contour = ub_points[u_idx[j]]['contourData']
                    interpolated_contour = interp_contour(zi, ub, lb, up_contour, lc_contour)
                    result += [{'contourData': interpolated_contour}]

                interpolated_planes[str(zi)] = result

        else:
            # Add original not interpolated plane
            ec_dist = abs(ordered_z - zi)
            neighbor = ec_dist.argmin()
            interpolated_planes[str(zi)] = s_planes[ordered_keys[neighbor]]

    # s_planes.update(interpolated_planes)

    return interpolated_planes


def set_interp_bounds(s_planes, kn):
    if kn[1] < kn[0]:
        l_idx = kn[1]
        u_idx = l_idx + 1
        if u_idx >= len(s_planes):
            u_idx = -1
            l_idx = kn[0]
    else:
        l_idx = kn[0]
        u_idx = kn[1]

    return l_idx, u_idx


def get_structure_planes(struc, end_capping=False):
    sPlanes = struc['planes']

    # Iterate over each plane in the structure
    zval = [z for z, sPlane in sPlanes.items()]
    zval.sort(key=float)

    # sorted Z axis planes

    structure_planes = []
    zplanes = []
    for z in zval:
        plane_i = sPlanes[z]
        for i in range(len(plane_i)):
            structure_planes.append(np.asarray(plane_i[i]['contourData']))
            zplanes.append(z)

    if end_capping:
        cap_delta = struc['thickness'] / 2
        start_cap = structure_planes[0].copy()
        start_cap[:, 2] = start_cap[:, 2] - cap_delta
        end_cap = structure_planes[-1].copy()
        end_cap[:, 2] = end_cap[:, 2] + cap_delta

        # # extending end caps to original plans
        structure_planes[0] = start_cap
        structure_planes[-1] = end_cap

    return structure_planes, np.array(zplanes, dtype=float)


def planes2array(s_planes):
    """
        Return all structure contour points as Point cloud array (x,y,z) points
    :param s_planes: Structure planes dict
    :return: points cloud contour points
    """
    zval = [z for z, sPlane in s_planes.items()]
    zval.sort(key=float)
    # sorted Z axis planes
    structure_planes = []
    zplanes = []
    for z in zval:
        plane_i = s_planes[z]
        for i in range(len(plane_i)):
            structure_planes.append(np.asarray(plane_i[i]['contourData']))
            zplanes.append(z)

    return np.concatenate(structure_planes), np.asarray(zplanes, dtype=float)


def nearest_neighbor(features_train, feature_query):
    """

    :param k: kn neighbors
    :param feature_train: reference 1D array grid
    :param features_query: query grid
    :return: lower and upper neighbors
    """
    ec_dist = np.sqrt((np.sum(features_train - feature_query, axis=1) ** 2.0))

    return ec_dist.argmin()


def calculate_structure_volume(structure):
    """Calculates the volume for the given structure."""

    sPlanes = structure['planes']

    # Store the total volume of the structure
    sVolume = 0

    n = 0
    # Iterate over each plane in the structure
    for sPlane in sPlanes.values():

        # Calculate the area for each contour in the current plane
        contours = []
        largest = 0
        largestIndex = 0
        for c, contour in enumerate(sPlane):
            # Create arrays for the x,y coordinate pair for the triangulation
            x = contour['contourData'][:, 0]
            y = contour['contourData'][:, 1]
            # # Calculate the area based on the Surveyor's formula
            cArea = calc_area(np.asarray(x), np.asarray(y))

            contours.append({'area': cArea, 'data': contour['contourData']})

            # Determine which contour is the largest
            if (cArea > largest):
                largest = cArea
                largestIndex = c

        # See if the rest of the contours are within the largest contour
        area = contours[largestIndex]['area']
        for i, contour in enumerate(contours):
            # Skip if this is the largest contour
            if not (i == largestIndex):
                contour['inside'] = False
                for point in contour['data']:
                    if point_in_contour(point, contours[largestIndex]['data']):
                        contour['inside'] = True
                        # Assume if one point is inside, all will be inside
                        break
                # If the contour is inside, subtract it from the total area
                if contour['inside']:
                    area = area - contour['area']
                # Otherwise it is outside, so add it to the total area
                else:
                    area = area + contour['area']

        # If the plane is the first or last slice
        # only add half of the volume, otherwise add the full slice thickness
        if (n == 0) or (n == len(sPlanes) - 1):
            sVolume = float(sVolume) + float(area) * float(structure['thickness']) * 0.5
        else:
            sVolume = float(sVolume) + float(area) * float(structure['thickness'])
        # Increment the current plane number
        n += 1

    # Since DICOM uses millimeters, convert from mm^3 to cm^3
    volume = sVolume / 1000

    return volume


def get_z_planes(struc_planes, ordered_z, z_interp_positions):
    result = []
    for zi in z_interp_positions:
        if zi not in ordered_z:
            # get grid knn
            kn = k_nearest_neighbors(2, ordered_z, zi)
            # define upper and lower bounds
            if kn[1] < kn[0]:
                l_idx = kn[1]
                u_idx = l_idx + 1
                if u_idx >= len(struc_planes):
                    u_idx = -1
                    l_idx = kn[0]
            else:
                l_idx = kn[0]
                u_idx = kn[1]

            # get upper and lower z values and contour points
            ub = struc_planes[u_idx][0][2]
            lb = struc_planes[l_idx][0][2]
            ub_points = struc_planes[u_idx]
            lb_points = struc_planes[l_idx]

            if not (len(ub_points) >= len(lb_points)):
                # if upper bounds does not have more points, swap planes to interpolate
                lbCopy = lb
                lb = ub
                ub = lbCopy
                ub_points = struc_planes[l_idx]
                lb_points = struc_planes[u_idx]

            interp_plane = interpolate_plane_numba(ub, lb, zi, ub_points, lb_points)
            result += [interp_plane]

        else:
            ec_dist = abs(ordered_z - zi)
            neighbor = ec_dist.argmin()
            result += [struc_planes[neighbor]]

    return result


def get_z_planes_dict(struc_planes, ordered_z, z_interp_positions):
    result = []
    for zi in z_interp_positions:
        if zi not in ordered_z:
            # get grid knn
            kn = k_nearest_neighbors(2, ordered_z, zi)
            # define upper and lower bounds
            if kn[1] < kn[0]:
                l_idx = kn[1]
                u_idx = l_idx + 1
                if u_idx >= len(struc_planes):
                    u_idx = -1
                    l_idx = kn[0]
            else:
                l_idx = kn[0]
                u_idx = kn[1]

            # get upper and lower z values and contour points
            ub = struc_planes[u_idx][0][2]
            lb = struc_planes[l_idx][0][2]
            ub_points = struc_planes[u_idx]
            lb_points = struc_planes[l_idx]
            if not (len(ub_points) >= len(lb_points)):
                # if upper bounds does not have more points, swap planes to interpolate
                lbCopy = lb
                lb = ub
                ub = lbCopy
                ub_points = struc_planes[l_idx]
                lb_points = struc_planes[u_idx]
            interp_plane = interpolate_plane_numba(ub, lb, zi, ub_points, lb_points)

            result += [interp_plane]

        else:
            ec_dist = abs(ordered_z - zi)
            neighbor = ec_dist.argmin()
            result += [struc_planes[neighbor]]

    return result


def calculate_contour_areas_numba(plane):
    """Calculate the area of each contour for the given plane.
       Additionally calculate and return the largest contour index."""

    # Calculate the area for each contour in the current plane
    contours = []
    largest = 0
    largestIndex = 0
    for c, contour in enumerate(plane):
        # Create arrays for the x,y coordinate pair for the triangulation
        x = contour['contourData'][:, 0]
        y = contour['contourData'][:, 1]

        cArea = calc_area(x, y)
        # cArea1 = poly_area(x, y)
        # np.testing.assert_almost_equal(cArea, cArea1)

        # Remove the z coordinate from the xyz point tuple
        data = np.asarray(list(map(lambda x: x[0:2], contour['contourData'])))

        # Add the contour area and points to the list of contours
        contours.append({'area': cArea, 'data': data})

        # Determine which contour is the largest
        if cArea > largest:
            largest = cArea
            largestIndex = c

    return contours, largestIndex


def contour_rasterization(dose_lut, dosegrid_points, contour, fx, fy, y_cord):
    polyY = fy(contour['data'][:, 1])
    polyX = fx(contour['data'][:, 0])

    n = len(dosegrid_points)
    out = np.zeros(n, dtype=bool)
    out = out.reshape((len(dose_lut[1]), len(dose_lut[0])))

    IMAGE_TOP = 0
    IMAGE_BOT = out.shape[0]
    IMAGE_RIGHT = out.shape[1]
    IMAGE_LEFT = 0
    polyCorners = len(contour['data'])
    # Loop through the rows of the image.
    for pixelY in range(IMAGE_TOP, IMAGE_BOT):

        # Build a list of nodes.
        nodes = 0
        j = polyCorners - 1
        nodeX = np.zeros(polyCorners, dtype=int)
        for i in range(polyCorners):
            b1 = (polyY[i] < y_cord[pixelY]) and (polyY[j] >= y_cord[pixelY])
            b2 = (polyY[j] < y_cord[pixelY]) and (polyY[i] >= y_cord[pixelY])
            if b1 or b2:
                f1 = pixelY - polyY[i]
                f2 = polyY[j] - polyY[i]
                f3 = polyX[j] - polyX[i]
                nodeX[nodes] = (polyX[i] + (f1 / f2) * f3)
                nodes += 1

            j = i

        # Sort the nodes, via a simple "Bubble" sort.
        i = 0
        while i < nodes - 1:
            if nodeX[i] > nodeX[i + 1]:
                swap = nodeX[i]
                nodeX[i] = nodeX[i + 1]
                nodeX[i + 1] = swap
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Fill the pixels between node pairs.
        for i in range(0, nodes, 2):
            if nodeX[i] >= IMAGE_RIGHT:
                break
            if nodeX[i + 1] > IMAGE_LEFT:
                if nodeX[i] < IMAGE_LEFT:
                    nodeX[i] = IMAGE_LEFT
                if nodeX[i + 1] > IMAGE_RIGHT:
                    nodeX[i + 1] = IMAGE_RIGHT

                for pixelX in range(nodeX[i], nodeX[i + 1]):
                    out[pixelY, pixelX] = True

    return out


# @nb.njit(nb.boolean[:, :](nb.boolean[:, :], nb.double[:], nb.double[:], nb.int64[:]))

@njit
def raster(out, polyX, polyY, y_cord):
    # /  public-domain code by Darel Rex Finley, 2007

    IMAGE_TOP = 0
    IMAGE_BOT = out.shape[0]
    IMAGE_RIGHT = out.shape[1] - 1
    IMAGE_LEFT = 0
    polyCorners = len(polyX)
    # Loop through the rows of the image.
    for pixelY in range(IMAGE_TOP, IMAGE_BOT):

        # Build a list of nodes.
        nodes = 0
        j = polyCorners - 1
        nodeX = np.zeros(polyCorners)
        for i in range(polyCorners):
            b1 = (polyY[i] < y_cord[pixelY]) and (polyY[j] >= y_cord[pixelY])
            b2 = (polyY[j] < y_cord[pixelY]) and (polyY[i] >= y_cord[pixelY])
            if b1 or b2:
                f1 = pixelY - polyY[i]
                f2 = polyY[j] - polyY[i]
                f3 = polyX[j] - polyX[i]
                nodeX[nodes] = int(polyX[i] + (f1 / f2) * f3 + 0.5)  # TODO add 0.5 ?
                nodes += 1

            j = i

        # Sort the nodes, via a simple "Bubble" sort.
        i = 0
        while i < nodes - 1:
            if nodeX[i] > nodeX[i + 1]:
                swap = nodeX[i]
                nodeX[i] = nodeX[i + 1]
                nodeX[i + 1] = swap
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Fill the pixels between node pairs.
        for i in range(0, nodes, 2):
            if nodeX[i] >= IMAGE_RIGHT:
                break
            if nodeX[i + 1] > IMAGE_LEFT:
                if nodeX[i] < IMAGE_LEFT:
                    nodeX[i] = IMAGE_LEFT
                if nodeX[i + 1] > IMAGE_RIGHT:
                    nodeX[i + 1] = IMAGE_RIGHT

                x1 = nodeX[i]
                x2 = nodeX[i + 1]

                for pixelX in range(x1, x2):
                    out[pixelY, pixelX] = True

    return out


def contour_rasterization_numba(dose_lut, dosegrid_points, contour, xx, yy):
    # TODO write unit tests and optimize this implementation
    # Wrap variables and coordinates to raster procedure
    xi = xx.flatten()
    yi = yy.flatten()
    # Convert high resolution grid to integer pixel coordinates
    raster_x_coord = np.arange(len(xi))
    raster_y_coord = np.arange(len(yi))
    raster_fx = interp1d(xi, raster_x_coord, fill_value='extrapolate')
    raster_fy = interp1d(yi, raster_y_coord, fill_value='extrapolate')
    poly_x = raster_fx(contour['data'][:, 0])
    poly_y = raster_fy(contour['data'][:, 1])
    n = len(dosegrid_points)
    out = np.zeros(n, dtype=bool)
    out = out.reshape((len(dose_lut[1]), len(dose_lut[0])))

    return raster(out, poly_x, poly_y, raster_y_coord)


def planes_point_cloud(sPlanes_dict):
    """
        Get point cloud from structure planes dict
    :param sPlanes_dict: DICOM Structure planes z dictionary
    :return: point cloud (x,y,z)
    """
    contour_data_planes = [plane for k, plane in sPlanes_dict.items()]
    ctr = []
    for p in contour_data_planes:
        for ctri in p:
            ctr.append(ctri['contourData'])
    point_cloud = np.concatenate(ctr)

    return point_cloud


def get_contour_roi_grid(contour_points, delta_mm=(0, 0), fac=1):
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    x_min = x.min() - delta_mm[0] * fac
    x_max = x.max() + delta_mm[0] * fac
    y_min = y.min() - delta_mm[1] * fac
    y_max = y.max() + delta_mm[1] * fac
    x_lut, x_delta = get_axis_grid(delta_mm[0], [x_min, x_max])
    y_lut, y_delta = get_axis_grid(delta_mm[1], [y_min, y_max])
    xg, yg = np.meshgrid(x_lut, y_lut)
    xf, yf = xg.flatten(), yg.flatten()
    contour_dose_grid = np.vstack((xf, yf)).T
    up_dose_lut = [x_lut, y_lut]

    return contour_dose_grid, up_dose_lut


def wrap_xy_coordinates(dose_lut, mapped_coord):
    """
        Wrap 3D structure and dose grid coordinates to regular ascending grid (x,y,z)
    :rtype: array,array,array,  string array
    :param structure_planes: Structure planes dict
    :param dose_lut: Dose look up table (XY plane)
    :param mapped_coord: Mapped
    :return: x,y x, coordinates and structure planes z ordered
    """
    xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)
    fx, fy, fz = mapped_coord
    x_c = fx(xx)
    y_c = fy(yy)

    return x_c, y_c


def wrap_coordinates(structure_planes, dose_lut, mapped_coord):
    """
        Wrap 3D structure and dose grid coordinates to regular ascending grid (x,y,z)
    :rtype: array,array,array,  string array
    :param structure_planes: Structure planes dict
    :param dose_lut: Dose look up table (XY plane)
    :param mapped_coord: Mapped
    :return: x,y x, coordinates and structure planes z ordered
    """
    xx, yy = np.meshgrid(dose_lut[0], dose_lut[1], indexing='xy', sparse=True)
    fx, fy, fz = mapped_coord
    ordered_keys = [z for z, sPlane in structure_planes.items()]
    ordered_keys.sort(key=float)
    x_c = fx(xx)
    y_c = fy(yy)
    z_c = fz(ordered_keys)

    return x_c, y_c, z_c, ordered_keys


def wrap_z_coordinates(structure_planes, mapped_coord):
    """
        Wrap 3D structure and dose grid coordinates to regular ascending grid (x,y,z)
    :rtype: array,array,array,  string array
    :param structure_planes: Structure planes dict
    :param dose_lut: Dose look up table (XY plane)
    :param mapped_coord: Mapped
    :return: x,y x, coordinates and structure planes z ordered
    """

    ordered_keys = [z for z, sPlane in structure_planes.items()]
    ordered_keys.sort(key=float)
    fx, fy, fz = mapped_coord
    z_c = fz(ordered_keys)

    return z_c, ordered_keys
