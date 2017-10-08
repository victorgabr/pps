import os
from unittest import TestCase

from pyplanscoring.core.constraints.metrics import PlanningItem
from pyplanscoring.core.dicomparser import ScoringDicomParser
from pyplanscoring.core.dvhcalculation import load

# DATA_DIR = os.path.join(
#     os.path.dirname(os.path.realpath(__file__)),
#     'test_data',
# )

DATA_DIR = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core/constraints/tests/test_data'

rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')

dvh_path = os.path.join(DATA_DIR, 'PyPlanScoring_dvh.dvh')

rp_dcm = ScoringDicomParser(filename=rp)
rs_dcm = ScoringDicomParser(filename=rs)
rd_dcm = ScoringDicomParser(filename=rd)

pyplan_dvh = load(dvh_path)
dvh = pyplan_dvh['DVH']


class TestHistoricPlanData(TestCase):
    def test_set_participant_folder(self):
        # set planning item pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)

        self.fail()
