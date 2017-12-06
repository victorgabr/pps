from unittest import TestCase

import numpy as np

from core.calculation import PyStructure, DVHCalculation
from core.dicom_reader import PyDicomParser
from core.tests import parotid_lt, dose_3d
from radiobiology.ntcp_models import NTCPLKBModel
from radiobiology.tests import dvh_diff_file

"""
    (m,n,TD50)=get_coeff('models.inp',modnr,'lkbmod')
    print "coeffs: m="+str(m)+"; n="+str(n)+"; TD50="+str(TD50)
    Deff=calc_Deff(dvh_file,roiname,np.float(n))
    print "NTCP =" + str(100*Lyman_Kutcher_Burman(Deff,np.float(TD50),np.float(m),0.001)) + "%"
"""


def get_dose_vol_array(dvh_file, what):
    f = open(dvh_file)
    found = 0
    stage = 0
    lastrow = 0
    dose, vol = [], []
    for l in f:
        if stage and l[:3] == "200":
            lastrow = 1
        if stage:
            l_ = l.split('\t')
            dose.append(np.float(l_[1].strip()))
            vol.append(np.float(l_[2].strip()))
        if lastrow:
            found = 0
            stage = 0
        if found == 0 and str.find(l, what) > 0:
            found = 1
        if found:
            if str.find(l, 'Bin') >= 0: stage = 1
    dose_step = (dose[-1] - dose[0]) / (len(dose) - 1)  # [Gy]
    dose_vol_array = np.zeros((len(dose), 2))

    dose_vol_array[:, 0] = dose
    dose_vol_array[:, 1] = vol

    return dose_vol_array


# stats_dvh = StatisticalDVH()
# stats_dvh.load_data_from_hdf(database_file)
# plan_dvh = stats_dvh.get_plan_dvh(0)
# # pi = PlanningItemDVH(plan_dvh)
# structure_dvh = plan_dvh['PAROTID LT']
# # pi.get_structure(structure_id)
# # change to differential DVH
# ddvh = get_ddvh(structure_dvh["data"])
# dose = structure_dvh['dose_axis'] / 100
#
# dose_step = (dose[-1] - dose[0]) / (len(dose) - 1)  # [Gy]
# dose_vol_array = np.zeros((len(dose), 2))
#
# dose_vol_array[:, 0] = dose
# dose_vol_array[:, 1] = ddvh / (dose_step * 100)
#
# import matplotlib.pyplot as plt
# plt.plot(dose_vol_array[:, 0], dose_vol_array[:, 1])

class TestNTCP(TestCase):
    def test_calc_model(self):
        # test DVH diff from DICOM
        # TPS CALCULATED dDVHs
        dvh_dcm = PyDicomParser(filename=dvh_diff_file)
        diff_dvhs = dvh_dcm.GetDVHs()
        dvh_diff = diff_dvhs[23]

        # calculate cDVH
        # Voxel size 0.2 mm
        bodyi = PyStructure(parotid_lt)
        dvh_calc = DVHCalculation(bodyi, dose_3d, (.2, .2, .2))
        dvh = dvh_calc.calculate()
        # calculated dDVH

        # calc NTCP Int. J. Radiation Oncology Biol. Phys., Vol. 78, No. 2, pp. 449–453, 2010
        TD50 = 39.4
        m = 0.40
        n = 1.13

        # calculating
        ncalc0 = NTCPLKBModel(dvh_diff, [TD50, m, n])
        ncalc1 = NTCPLKBModel(dvh, [TD50, m, n])

        ntcp0 = ncalc0.calc_model()
        ntcp1 = ncalc1.calc_model()

        # NTCP result coincides within 3 places
        self.assertAlmostEqual(ntcp0, ntcp1, places=3)
