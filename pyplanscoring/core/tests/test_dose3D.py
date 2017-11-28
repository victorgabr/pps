from unittest import TestCase

import quantities as pq

from core.geometry import get_dose_grid_3d, get_contour_roi_grid, calculate_contour_areas
from core.tests import rd_dcm, plot_flag, body
from core.types import Dose3D

dose_values = rd_dcm.get_dose_matrix()
grid = rd_dcm.get_grid_3d()
dose_3d = Dose3D(dose_values, grid, pq.Gy)


class TestDose3D(TestCase):
    def test_dose_max_3d_location(self):
        location = dose_3d.dose_max_location
        assert len(location) == 3
        # check value inside grid range
        for i in range(len(location)):
            assert grid[i].min() <= location[i] <= grid[i].max()

    def test_image_resolution(self):
        # test dose image resolution x,y,z
        self.assertAlmostEqual(dose_3d.x_res, 2.5)
        self.assertAlmostEqual(dose_3d.y_res, 2.5)
        self.assertAlmostEqual(dose_3d.z_res, 3.0)

    def test_x_size(self):
        assert dose_3d.x_size == len(grid[0])

    def test_y_size(self):
        assert dose_3d.y_size == len(grid[1])

    def test_z_size(self):
        assert dose_3d.z_size == len(grid[2])

    def test_get_z_dose_plane(self):
        import matplotlib.pyplot as plt
        zi = '100.50'
        grid_delta = (0.2, 0.2, 3)
        # test get z at position 0 mm and no lookup table
        dose_0 = dose_3d.get_z_dose_plane(float(zi))

        # test_high resolution xy grid
        dose_grid_points, up_dose_lut, spacing = get_dose_grid_3d(grid, delta_mm=grid_delta)
        dose_1 = dose_3d.get_z_dose_plane(float(zi), up_dose_lut)

        # plot brain dose contour at using rasterisation window
        brain_slices = list(body['planes'].keys())
        brain_slices.sort(key=float)

        slice_i = body['planes'][zi]
        contours, largest_index = calculate_contour_areas(slice_i)
        # contour
        contour_points = contours[0]['data']
        contour_dose_grid, ctr_dose_lut = get_contour_roi_grid(contour_points, delta_mm=grid_delta, fac=1)
        dose_2 = dose_3d.get_z_dose_plane(float(zi), ctr_dose_lut)
        if plot_flag:
            plt.imshow(dose_0, interpolation='none')
            plt.title("Dose at axis z: {} mm - grid resolution: 2.5 mm x 2.5 mm".format(zi))
            plt.figure()
            plt.imshow(dose_1, interpolation='none')
            plt.title("Dose at axis z: {} mm - grid resolution: 0.1 mm x 0.1 mm".format(zi))
            plt.figure()
            plt.imshow(dose_2, interpolation='none')
            plt.title("Windowed Dose at axis z: {} mm - grid resolution: 0.1 mm x 0.1 mm".format(zi))
            plt.show()

    def test_get_dose_to_point(self):
        # test dose max location
        dose_max = dose_3d.dose_max_3d
        location = dose_3d.dose_max_location
        d_teste = dose_3d.get_dose_to_point(location)
        self.assertAlmostEqual(d_teste, dose_max)

        # def test_get_voxels(self):
        #     self.fail()
        #
        # def test_set_voxels(self):
        #     self.fail()
        #
        # def test_voxel_to_dose_value(self):
        #     self.fail()
