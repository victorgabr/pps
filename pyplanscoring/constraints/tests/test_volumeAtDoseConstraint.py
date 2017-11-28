# GETTING dvh DATA FROM DOSE
import os
from unittest import TestCase

from pyplanscoring.core.constraints.metrics import PlanningItem
from pyplanscoring.core.constraints.types import PriorityType, ResultType, VolumePresentation

from constraints.constraints import MayoConstraintConverter
from core.dicom_reader import ScoringDicomParser

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
# planning item
pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)

converter = MayoConstraintConverter()


class TestVolumeAtDoseConstraint(TestCase):
    def test_volume(self):
        structure_name = 'PTV_70_3mm'
        constrain = 'V6103.854532025905cGy[%] >= 95'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'V6103.854532025905cGy[%] >= 97'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 97 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        structure_name = 'PTV_70_3mm'
        constrain = 'V6103.854532025905cGy[%] <= 94'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 94 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        constrain = 'V6103.854532025905cGy[cc] <= 655.0261147733513'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 655.0261147733513 * VolumePresentation.absolute_cm3)
        constrain_result = mdc.constrain(pi)

        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED


        # def test_volume_type(self):
        #     self.fail()
        #
        # def test_passing_func(self):
        #     self.fail()
        #
        # def test_get_volume_at_dose(self):
        #     self.fail()
        #
        # def test_constrain(self):
        #     self.fail()
