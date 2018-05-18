import numpy as np
from numba import cuda, njit

from core.calculation import DVHCalculation
from core.geometry import check_contour_inside


class DVHCalculationGPU(DVHCalculation):

    def __init__(self, structure, dose, calc_grid=None):
        super().__init__(structure, dose, calc_grid)

    def calculate_gpu(self, verbose=True):

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
                # calculate on GPU
                m = get_contour_mask_gpu(ctr_dose_lut, contour_dose_grid, contour['data'])
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


is_left_gpu = cuda.jit(device=True)(is_left)


def loop_all_edges(point, poly, wn, startY, N, gridY):
    # // loop through all edges of the polygon
    for k in range(startY, N - 1, gridY):  # edge from V[i] to  V[i+1]

        if poly[k][1] <= point[1]:  # start y <= P[1]
            if poly[k + 1][1] > point[1]:  # an upward crossing
                is_left_value = is_left_gpu(poly[k], poly[k + 1], point)

                # p0 = poly[k]
                # p1 = poly[k + 1]
                # p2 = point
                #
                # is_left_value = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

                if is_left_value >= 0:  # P left of  edge
                    wn = wn + 1  # // have  a valid up intersect

        else:  # start y > P[1] (no test needed)
            if poly[k + 1][1] <= point[1]:  # a downward crossing
                is_left_value = is_left_gpu(poly[k], poly[k + 1], point)
                # p0 = poly[k]
                # p1 = poly[k + 1]
                # p2 = point
                # is_left_value = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
                if is_left_value <= 0:  # P right of  edge
                    wn = wn - 1  # have  a valid down intersect

    return wn


loop_all_edges_gpu = cuda.jit(device=True)(loop_all_edges)


@cuda.jit(device=True)
def contour_loop(point, poly, N):
    # loop through all edges of the polygon
    wn = 0
    for k in range(N - 1):  # edge from V[i] to  V[i+1]

        if poly[k][1] <= point[1]:  # start y <= P[1]
            if poly[k + 1][1] > point[1]:  # an upward crossing
                p0 = poly[k]
                p1 = poly[k + 1]
                p2 = point
                is_left_value = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
                if is_left_value >= 0:  # P left of  edge
                    wn += 1  # // have  a valid up intersect

        else:  # start y > P[1] (no test needed)
            if poly[k + 1][1] <= point[1]:  # a downward crossing
                p0 = poly[k]
                p1 = poly[k + 1]
                p2 = point
                is_left_value = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
                if is_left_value <= 0:  # P right of  edge
                    wn -= 1  # have  a valid down intersect

    return wn


@cuda.jit
def wn_contains_points_kernel(out, poly, points):
    """
        Winding number test for a list of point in a polygon
        Numba implementation 8 - 10 x times faster than Matplotlib Path.contains_points()
        CUDA-GPU kernel
    :param out: output boolean array
    :param poly: polygon (list of points/vertex)
    :param points: list of points to check inside polygon
    :return: Boolean array
        adapted from c++ code at:
        http://geomalgorithms.com/a03-_inclusion.html

    """
    n = len(points)
    N = len(poly)
    startX = cuda.grid(1)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    for i in range(startX, n, gridX):
        point = points[i]
        out[i] = contour_loop(point, poly, N)

        # wn = 0  # the  winding number counter
        # # // loop through all edges of the polygon
        # for k in range(startY, N - 1, gridY):  # edge from V[i] to  V[i+1]
        #
        #     if poly[k][1] <= point[1]:  # start y <= P[1]
        #         if poly[k + 1][1] > point[1]:  # an upward crossing
        #             is_left_value = is_left_gpu(poly[k], poly[k + 1], point)
        #             if is_left_value >= 0:  # P left of  edge
        #                 wn += 1  # // have  a valid up intersect
        #
        #     else:  # start y > P[1] (no test needed)
        #         if poly[k + 1][1] <= point[1]:  # a downward crossing
        #             is_left_value = is_left_gpu(poly[k], poly[k + 1], point)
        #             if is_left_value <= 0:  # P right of  edge
        #                 wn -= 1  # have  a valid down intersect
        #
        # cuda.syncthreads()


def get_contour_mask_gpu(doselut, dosegrid_points, poly):
    """
        Get the mask for the contour with respect to the dose plane.
        http://numba.pydata.org/numba-doc/dev/cuda/kernels.html
    :param doselut: Dicom 3D dose LUT (x,y)
    :param dosegrid_points: dosegrid_points
    :param poly: contour
    :return: contour mask on grid
    """

    n = len(dosegrid_points)
    grid = np.zeros(n, dtype=bool)
    # preparing data to wn test
    # repeat the first vertex at end
    poly_wn = np.zeros((poly.shape[0] + 1, poly.shape[1]))
    poly_wn[:-1] = poly
    poly_wn[-1] = poly[0]

    threadsperblock = 1024
    blockspergrid = (grid.size + (threadsperblock - 1)) // threadsperblock
    # blockspergrid = 8
    d_image = cuda.to_device(grid)
    # wn_contains_points_kernel[griddim, blockdim](d_image, poly_wn, dosegrid_points)
    wn_contains_points_kernel[blockspergrid, threadsperblock](d_image, poly_wn, dosegrid_points)
    d_image.to_host()

    grid = grid.reshape((len(doselut[1]), len(doselut[0])))

    return grid


def mandelc(x, y, max_iters):
    """
      Given the real and imaginary parts of a complex number,
      determine if it is a candidate for membership in the Mandelbrot
      set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


mandel_gpu = cuda.jit(device=True)(mandelc)


def mandel(x, y, max_iters):
    """
      Given the real and imaginary parts of a complex number,
      determine if it is a candidate for membership in the Mandelbrot
      set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


mandel_cpu = njit(mandel)


def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


@njit
def create_fractal_cpu(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel_cpu(real, imag, iters)
            image[y, x] = color


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel_gpu(real, imag, iters)


if __name__ == '__main__':
    from timeit import default_timer as timer

    from core.calculation import DVHCalculation, PyStructure
    import matplotlib.pyplot as plt
    from core.tests import dose_3d, lens
    import numpy.testing as npt
    from timeit import default_timer as timer

    # Small volume no end cap and upsampling
    braini = PyStructure(lens)
    dvh_calc_cpu = DVHCalculation(braini, dose_3d, calc_grid=(0.05, 0.05, 0.05))

    # Small volume no end cap and upsampling and GPU
    braini = PyStructure(lens)
    dvh_calc_gpu = DVHCalculation(braini, dose_3d, calc_grid=(0.05, 0.05, 0.05))
    dvh_gpu = dvh_calc_gpu.calculate_gpu()

    start = timer()
    dvh_cpu = dvh_calc_cpu.calculate()
    dt1 = timer() - start
    print("DVH on CPU %f s" % dt1)

    # dvh_calc_gpu.calculate_gpu()
    start = timer()
    dvh_gpu = dvh_calc_gpu.calculate_gpu()
    dt2 = timer() - start
    print("DVH on GPU %f s" % dt2)

    plt.plot(dvh_cpu['data'])
    plt.figure()
    plt.plot(dvh_gpu['data'])
    plt.show()
    npt.assert_array_almost_equal(dvh_cpu['data'], dvh_gpu['data'])
