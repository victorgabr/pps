from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from core.calculation import DVHCalculation, PyStructure
from core.tests import body, dose_3d, plot_flag, ptv70, lens


def plot_dvh(dvh, title):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
    plt.plot(x, dvh['data'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
        dvh_calci = DVHCalculation(braini, dose_3d, calc_grid=np.array((0.1, 0.1, 0.1)))
        dvh_lu = dvh_calci.calculate()
        if plot_flag:
            plot_dvh(dvh, "BODY")
            plot_dvh(dvhb, "PTV 70")
            plot_dvh(dvh_l, "LENS LT")
            plot_dvh(dvh_lu, "LENS LT - voxel size: (0.1, 0.1, 0.1)")
            plt.show()
