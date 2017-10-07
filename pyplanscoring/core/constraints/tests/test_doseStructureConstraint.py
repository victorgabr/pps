# GETTING dvh DATA FROM DOSE
import os
from unittest import TestCase

from pyplanscoring.core.constraints.constraints import MayoConstraintConverter
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


class TestDoseStructureConstraint(TestCase):
    def test_can_constrain(self):
        rp_dcm = ScoringDicomParser(filename=rp)
        rs_dcm = ScoringDicomParser(filename=rs)
        rd_dcm = ScoringDicomParser(filename=rd)
        # planning item
        pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        structure_name = 'PTV70-BR.PLX 4MM'
        constrain = 'CI66.5Gy[] >= 0.6'
        converter = MayoConstraintConverter()
        max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        can_constraint_result = max_dc.can_constrain(pi)
        assert can_constraint_result.is_success
