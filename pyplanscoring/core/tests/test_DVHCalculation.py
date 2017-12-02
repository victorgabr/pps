from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from core.calculation import DVHCalculation, PyStructure
from core.tests import body, dose_3d, plot_flag, ptv70, lens, rd_dcm, structures


def plot_dvh(dvh, title):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
    plt.plot(x, dvh['data'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def plot_dvh_comp(dvh_calc, dvh, title):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])
    x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
    plt.plot(x_calc, dvh_calc['data'] / dvh_calc['data'][0], label='PyPlanScoring')
    plt.plot(x, dvh['data'] / dvh['data'][0], label='Eclipse')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)


class TestDVHCalculation(TestCase):
    def test_calculate(self):
        # dvh calculation with no upsampling - body no end-cap
        bodyi = PyStructure(body)
        dvh_calc = DVHCalculation(bodyi, dose_3d)
        dvh = dvh_calc.calculate()

        # dvh calculation with no upsampling - brain no end-cap
        braini = PyStructure(ptv70)
        dvh_calci = DVHCalculation(braini, dose_3d)
        dvhb = dvh_calci.calculate()

        # small volume
        braini = PyStructure(lens)
        dvh_calci = DVHCalculation(braini, dose_3d)
        dvh_l = dvh_calci.calculate()

        # Small volume no end cap and upsampling
        braini = PyStructure(lens)
        dvh_calci = DVHCalculation(braini, dose_3d, calc_grid=(0.1, 0.1, 0.1))
        dvh_lu = dvh_calci.calculate()

        if plot_flag:
            plot_dvh(dvh, "BODY")
            plot_dvh(dvhb, "PTV 70")
            plot_dvh(dvh_l, "LENS LT")
            plot_dvh(dvh_lu, "LENS LT - voxel size: (0.1, 0.1, 0.1)")

            # compare with TPS DVH
            dvhs = rd_dcm.GetDVHs()
            dvh_calculated = {}
            for roi_number in dvhs.keys():
                struc_i = PyStructure(structures[roi_number])
                if struc_i.volume < 100:
                    dvh_calc = DVHCalculation(struc_i, dose_3d, calc_grid=(.5, .5, .5))
                else:
                    dvh_calc = DVHCalculation(struc_i, dose_3d)
                dvh = dvh_calc.calculate(verbose=True)
                dvh_calculated[roi_number] = dvh

            for roi_number in dvhs.keys():
                plot_dvh_comp(dvh_calculated[roi_number], dvhs[roi_number], structures[roi_number]['name'])
                plt.show()
