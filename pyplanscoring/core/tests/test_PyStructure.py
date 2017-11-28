from unittest import TestCase

import numpy as np

from core.calculation import PyStructure
from core.tests import brain


class TestPyStructure(TestCase):
    def test_structure(self):
        # creating a instance with no end-cap
        obj = PyStructure(brain)
        # creating a instance with end-cap half of slice tickness
        end_cap = brain['thickness'] / 2.0
        obj1 = PyStructure(brain, end_cap)

        # testing high resolution structures
        z_grid = 0.2
        # creating a instance with no end-cap
        obj2 = PyStructure(brain)
        obj2.to_high_resolution(z_grid)
        # creating a instance with end-cap half of slice tickness
        end_cap = brain['thickness'] / 2.0
        obj3 = PyStructure(brain, end_cap)
        obj3.to_high_resolution(z_grid)

        assert obj.structure

        # assert contour spacing
        self.assertAlmostEqual(obj.contour_spacing, 3.0)
        self.assertAlmostEqual(obj1.contour_spacing, 3.0)

        # assert planes
        self.assertNotEqual(obj.planes, obj1.planes)

        # assert name
        assert obj.name == 'Brain'
        assert obj1.name == obj.name
        assert obj.id == 'Brain'

        # testing different point clouds
        pc = obj.point_cloud
        pc1 = obj1.point_cloud
        assert not np.allclose(pc, pc1)

        # test isocenter
        assert np.allclose(obj.center_point, obj1.center_point)

        # color
        assert np.any(obj1.color)

        # dicom type
        assert obj1.dicom_type

        # assert roi number
        assert obj1.roi_number

        # test volumes. End cap or truncate should be different
        self.assertNotEqual(obj.volume, obj1.volume)

        # assert both are high resolution
        assert obj2.is_high_resolution
        assert obj3.is_high_resolution

        # assert not equality of volumes
        self.assertNotEqual(obj.volume, obj2.volume)
        self.assertNotEqual(obj1.volume, obj2.volume)

        # def test_get_contours_on_image_plane(self):
        assert obj.get_contours_on_image_plane('25.50')

        with self.assertRaises(TypeError):
            obj.get_contours_on_image_plane(25.50)

        # test no existing plane
        assert not obj.get_contours_on_image_plane('25.5')
