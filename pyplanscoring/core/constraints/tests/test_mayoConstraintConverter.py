import os
from unittest import TestCase

from pyplanscoring.core.constraints.constraints import MayoConstraintConverter, MayoConstraint
from pyplanscoring.core.constraints.metrics import PlanningItem
from pyplanscoring.core.constraints.types import PriorityType
from pyplanscoring.core.dicomparser import ScoringDicomParser

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# DATA_DIR = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data'

rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')

rp_dcm = ScoringDicomParser(filename=rp)
rs_dcm = ScoringDicomParser(filename=rs)
rd_dcm = ScoringDicomParser(filename=rd)

pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)

converter = MayoConstraintConverter()


class TestMayoConstraintConverter(TestCase):
    def test_convert_to_dvh_constraint(self):
        # test string representation
        constrain = 'Max[Gy] <= 45'
        structure_name = 'SPINAL CORD'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

        # test constraint values

        constrain = 'Min[Gy] >= 65'
        structure_name = 'PTV 70'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

        constrain = 'Mean[Gy] >= 65'
        structure_name = 'PTV 70'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

        constrain = 'D95%[Gy] >= 70'
        structure_name = 'PTV70'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

        constrain = 'DC95%[Gy] >= 70'
        structure_name = 'PTV70'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

        constrain = 'DC95%[Gy] >= 70'
        structure_name = 'PTV70'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

        constrain = 'V95%[%] >= 95'
        structure_name = 'PTV70'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

        constrain = 'CV95%[%] >= 95'
        structure_name = 'PTV70'
        res = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        self.assertEqual(str(res), constrain)

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
        constrain = 'Max[Gy] <= 45'
        structure_name = 'SPINAL CORD'
        mc = MayoConstraint()
        mc.read(constrain)
        mdc = converter.build_max_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(mdc), constrain)

        constrain = 'Max[cGy] <= 4500'
        structure_name = 'SPINAL CORD'
        mc = MayoConstraint()
        mc.read(constrain)
        mdc = converter.build_max_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(mdc), constrain)

        constrain = 'Max[%] <= 64'
        structure_name = 'SPINAL CORD'
        mc = MayoConstraint()
        mc.read(constrain)
        mdc = converter.build_max_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(mdc), constrain)

    def test_build_min_dose_constraint(self):
        constrain = 'Min[Gy] >= 65'
        structure_name = 'PTV 70'
        mc = MayoConstraint()
        mc.read(constrain)
        mdc = converter.build_min_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(mdc), constrain)

        constrain = 'Min[cGy] >= 6550'
        structure_name = 'PTV 70'
        mc = MayoConstraint()
        mc.read(constrain)
        mdc = converter.build_min_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(mdc), constrain)

        constrain = 'Min[%] >= 90'
        structure_name = 'PTV 70'
        mc = MayoConstraint()
        mc.read(constrain)
        mdc = converter.build_min_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(mdc), constrain)

    def test_build_mean_dose_constraint(self):
        constrain = 'Mean[Gy] <= 20'
        structure_name = 'PAROTID'
        mc = MayoConstraint()
        mc.read(constrain)
        max_mean_c = converter.build_mean_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(max_mean_c), constrain)

        constrain = 'Mean[Gy] >= 20'
        structure_name = 'PTV20'
        mc = MayoConstraint()
        mc.read(constrain)
        min_mean_c = converter.build_mean_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_mean_c), constrain)

    def test_build_dose_at_volume_constraint(self):
        # Upper constrain
        constrain = 'D95%[Gy] >= 70'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_dose_at_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'D95%[cGy] >= 70'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_dose_at_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'D95cc[Gy] >= 70'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_dose_at_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'D10cc[cGy] <= 7000'
        structure_name = 'RECTUM'
        mc = MayoConstraint()
        mc.read(constrain)
        max_dv = converter.build_dose_at_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(max_dv), constrain)

    def test_build_volume_at_dose_constraint(self):
        constrain = 'V95%[%] >= 95'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_volume_at_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'V95%[cc] >= 95'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_volume_at_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'V20Gy[%] <= 20'
        structure_name = 'LUNGS'
        mc = MayoConstraint()
        mc.read(constrain)
        max_dv = converter.build_volume_at_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(max_dv), constrain)

        constrain = 'V20Gy[cc] <= 20'
        structure_name = 'LUNGS'
        mc = MayoConstraint()
        mc.read(constrain)
        max_dv = converter.build_volume_at_dose_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(max_dv), constrain)

    def test_build_compliment_volume_constraint(self):
        constrain = 'CV95%[%] >= 95'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_compliment_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'CV95%[cc] >= 95'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_compliment_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'CV20Gy[%] <= 20'
        structure_name = 'LUNGS'
        mc = MayoConstraint()
        mc.read(constrain)
        max_dv = converter.build_compliment_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(max_dv), constrain)

        constrain = 'CV20Gy[cc] <= 20'
        structure_name = 'LUNGS'
        mc = MayoConstraint()
        mc.read(constrain)
        max_dv = converter.build_compliment_volume_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(max_dv), constrain)

    def test_build_dose_compliment_constraint(self):
        # Upper constrain
        constrain = 'DC95%[Gy] >= 70'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_dose_compliment_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'DC95%[cGy] >= 70'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_dose_compliment_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'DC95cc[Gy] >= 70'
        structure_name = 'PTV70'
        mc = MayoConstraint()
        mc.read(constrain)
        min_dv = converter.build_dose_compliment_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(min_dv), constrain)

        constrain = 'DC10cc[cGy] <= 7000'
        structure_name = 'RECTUM'
        mc = MayoConstraint()
        mc.read(constrain)
        max_dv = converter.build_dose_compliment_constraint(mc, structure_name, PriorityType.IDEAL)
        self.assertEqual(str(max_dv), constrain)
