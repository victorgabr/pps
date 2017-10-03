# GETTING dvh DATA FROM DOSE
import os
from unittest import TestCase

from pyplanscoring.core.constraints.constraints import MayoConstraintConverter
from pyplanscoring.core.constraints.metrics import PlanningItem
from pyplanscoring.core.constraints.types import PriorityType, ResultType
from pyplanscoring.core.dicomparser import ScoringDicomParser

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# DATA_DIR = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data'

rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')


class TestMaxDoseConstraint(TestCase):
    def test_constrain(self):
        rp_dcm = ScoringDicomParser(filename=rp)
        rs_dcm = ScoringDicomParser(filename=rs)
        rd_dcm = ScoringDicomParser(filename=rd)
        # planning item
        pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)

        # instantiate MayoConstraintConverter
        converter = MayoConstraintConverter()
        constrain = 'Max[Gy] <= 45'
        structure_name = 'SPINAL CORD'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'Max[Gy] <= 20'
        max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = max_dc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
