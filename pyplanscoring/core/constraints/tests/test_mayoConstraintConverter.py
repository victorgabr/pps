from unittest import TestCase

from pyplanscoring.core.constraints.constraints import MayoConstraintConverter

converter = MayoConstraintConverter()


class TestMayoConstraintConverter(TestCase):
    def test_convert_to_dvh_constraint(self):
        self.fail()

    def test_get_volume_units(self):
        vu = converter.get_volume_units(0)
        assert vu.symbol == 'cc'
        vu = converter.get_volume_units(1)
        assert vu.symbol == '%'

    def test_get_dose_units(self):
        du = converter.get_dose_units(1)
        assert du.symbol == '%'
        du = converter.get_dose_units(2)
        assert du.symbol == 'Gy'
        du = converter.get_dose_units(3)
        assert du.symbol == 'cGy'
        du = converter.get_dose_units(4)
        assert du.symbol == 'dimensionless'

    def test_build_max_dose_constraint(self):
        self.fail()

    def test_build_min_dose_constraint(self):
        self.fail()

    def test_build_mean_dose_constraint(self):
        self.fail()

    def test_build_dose_at_volume_constraint(self):
        self.fail()

    def test_build_volume_at_dose_constraint(self):
        self.fail()

    def test_build_compliment_volume_constraint(self):
        self.fail()

    def test_build_dose_compliment_constraint(self):
        self.fail()
