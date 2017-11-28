# GETTING dvh DATA FROM DOSE
import os
from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.metrics import PlanningItem
from core.dicom_reader import ScoringDicomParser
from core.types import PriorityType, ResultType, VolumePresentation

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


class TestComplimentDoseAtVolumeConstraint(TestCase):
    def test_get_dose_compliment_at_volume(self):
        'DC95%[cGy]'
        # instantiate MayoConstraintConverter
        structure_name = 'PTV_70_3mm'
        constrain = 'DC95%[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        structure_name = 'PTV_70_3mm'
        constrain = 'DC95%[%] >= 87'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        structure_name = 'PTV_70_3mm'
        constrain = 'DC655.0261147733513cc[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 655.0261147733513 * VolumePresentation.absolute_cm3)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED
