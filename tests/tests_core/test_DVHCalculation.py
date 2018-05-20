import  os
import matplotlib.pyplot as plt
import numpy as np

from pyplanscoring.core.calculation import DVHCalculation, PyStructure
from pyplanscoring.core.dicom_reader import PyDicomParser
from pyplanscoring.core.types import Dose3D, DoseUnit


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


def test_calculation_with_end_cap(dose_3d, optic_chiasm, body, ptv70, lens):
    # dvh calculation with no upsampling - body no end-cap
    bodyi = PyStructure(body, end_cap=0.2)
    dvh_calc = DVHCalculation(bodyi, dose_3d)
    dvh = dvh_calc.calculate()

    # dvh calculation with no upsampling - brain no end-cap
    braini = PyStructure(ptv70, end_cap=0.2)
    dvh_calci = DVHCalculation(braini, dose_3d)
    dvhb = dvh_calci.calculate()

    # SMALL VOLUMES STRUCTURES
    bodyi = PyStructure(lens, end_cap=0.2)
    dvh_calc = DVHCalculation(bodyi, dose_3d)
    dvh = dvh_calc.calculate()

    # dvh calculation with no upsampling - brain no end-cap
    braini = PyStructure(optic_chiasm, end_cap=0.2)
    dvh_calci = DVHCalculation(braini, dose_3d)
    dvhb = dvh_calci.calculate()


def test_calculation_with_up_sampling_end_cap(dose_3d, optic_chiasm, lens):
    # SMALL VOLUMES STRUCTURES
    bodyi = PyStructure(lens, end_cap=0.2)
    dvh_calc = DVHCalculation(bodyi, dose_3d, calc_grid=(0.2, 0.2, 0.2))
    dvh = dvh_calc.calculate()

    # dvh calculation with no upsampling - brain no end-cap
    braini = PyStructure(optic_chiasm, end_cap=0.2)
    dvh_calci = DVHCalculation(braini, dose_3d, calc_grid=(0.2, 0.2, 0.2))
    dvhb = dvh_calci.calculate()


def test_calculate(structures, optic_chiasm, body, ptv70, lens, plot_flag, rd_dcm, dose_3d):
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
    dvh_calc_cpu = DVHCalculation(braini, dose_3d, calc_grid=(0.2, 0.2, 0.2))
    dvh_lu = dvh_calc_cpu.calculate()

    # Small volume no end cap and upsampling
    braini = PyStructure(optic_chiasm)
    dvh_calc_cpu = DVHCalculation(braini, dose_3d, calc_grid=(0.2, 0.2, 0.2))
    dvh_lu = dvh_calc_cpu.calculate()

    # Small volume no end cap and upsampling and GPU
    # braini = PyStructure(lens)
    # dvh_calc_gpu = DVHCalculation(braini, dose_3d, calc_grid=(0.05, 0.05, 0.05))
    # dvh_lu_gpu = dvh_calc_gpu.calculate_gpu()

    if plot_flag:
        plot_dvh(dvh, "BODY")
        plot_dvh(dvhb, "PTV 70")
        plot_dvh(dvh_l, "LENS LT")
        plot_dvh(dvh_lu, "LENS LT - voxel size [mm3]: (0.1, 0.1, 0.1)")
        # plot_dvh(dvh_lu_gpu, "GPU LENS LT - voxel size [mm3]: (0.1, 0.1, 0.1)")

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

# TODO REFACTOR
# def test_calc_structure_rings(dicom_folder):
#     """
#        Test case to lung SBRT structures.
#        roi_number: 34,
#        name: D2CM PRIMARY,
#        roi_number: 35,
#        name: D2CM LN
#     """
#     # given
#     rs_dvh = os.path.join(dicom_folder, 'RS.dcm')
#     rd = os.path.join(dicom_folder,'RD.dcm')
#
#     # 3D dose matrix
#     dose_dcm = PyDicomParser(filename=rd)
#     dose_values = dose_dcm.get_dose_matrix()
#     grid = dose_dcm.get_grid_3d()
#     dose_3d = Dose3D(dose_values, grid, DoseUnit.Gy)
#
#     # structures
#     structures = PyDicomParser(filename=rs_dvh).GetStructures()
#     d2cm_prim = PyStructure(structures[34])
#
#     dvh_calc = DVHCalculation(d2cm_prim, dose_3d)
#     d2cm_prim_dvh = dvh_calc.calculate()
#     pass
