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


class TestDoseAtVolumeConstraint(TestCase):
    def test_volume(self):
        # instantiate MayoConstraintConverter
        structure_name = 'PTV_70_3mm'
        constrain = 'D95%[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        structure_name = 'PTV_70_3mm'
        constrain = 'D655.0261147733513cc[Gy] >= 65 '
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 655.0261147733513 * VolumePresentation.absolute_cm3)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

    def test_volume_type(self):
        # instantiate MayoConstraintConverter
        structure_name = 'PTV_70_3mm'
        constrain = 'D95%[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume_type, VolumePresentation.relative)

        structure_name = 'PTV_70_3mm'
        constrain = 'D655.0261147733513cc[Gy] >= 65 '
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume_type, VolumePresentation.absolute_cm3)

    def test_passing_func(self):
        # instantiate MayoConstraintConverter
        structure_name = 'PTV_70_3mm'
        constrain = 'D95%[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        structure_name = 'PTV_70_3mm'
        constrain = 'D655.0261147733513cc[Gy] >= 65 '
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        # Not Passing Results
        constrain = 'D0.01cc[Gy] <= 45'
        structure_name = 'SPINAL CORD'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'D0.01cc[Gy] <= 20'
        structure_name = 'SPINAL CORD'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
