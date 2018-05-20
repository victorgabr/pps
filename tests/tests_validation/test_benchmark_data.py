import os
from collections import OrderedDict

import re
import scipy.interpolate as itp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  pyplanscoring.constraints.metrics import DVHMetrics
from pyplanscoring.core.calculation import DVHCalculation
from pyplanscoring.core.types import DVHData

plt.style.use('ggplot')


class CurveCompare:
    """
        Statistical analysis of the DVH volume (%) error histograms. volume (cm 3 ) differences (numerical–analytical)
        were calculated for points on the DVH curve sampled at every 10 cGy then normalized to
        the structure's total volume (cm 3 ) to give the error in volume (%)
    """

    def __init__(self, a_dose, a_dvh, calc_dose, calc_dvh, structure_name='', dose_grid='', gradient=''):
        self.calc_data = ''
        self.ref_data = ''
        self.a_dose = a_dose
        self.a_dvh = a_dvh
        self.cal_dose = calc_dose
        self.calc_dvh = calc_dvh
        self.sampling_size = 10 / 100.0
        self.dose_samples = np.arange(0, len(calc_dvh) / 100,
                                      self.sampling_size)  # The DVH curve sampled at every 10 cGy
        self.ref_dvh = itp.interp1d(a_dose, a_dvh, fill_value='extrapolate')
        self.calc_dvh = itp.interp1d(calc_dose, calc_dvh, fill_value='extrapolate')
        self.delta_dvh = self.calc_dvh(self.dose_samples) - self.ref_dvh(self.dose_samples)
        self.delta_dvh_pp = (self.delta_dvh / a_dvh[0]) * 100
        # prepare data dict
        # self.calc_dvh_dict = _prepare_dvh_data(self.dose_samples, self.calc_dvh(self.dose_samples))
        # self.ref_dvh_dict = _prepare_dvh_data(self.dose_samples, self.ref_dvh(self.dose_samples))
        # title data
        self.structure_name = structure_name
        self.dose_grid = dose_grid
        self.gradient = gradient

    def stats(self):
        df = pd.DataFrame(self.delta_dvh_pp, columns=['delta_pp'])
        print(df.describe())

    @property
    def stats_paper(self):
        stats = {}
        stats['min'] = self.delta_dvh_pp.min().round(1)
        stats['max'] = self.delta_dvh_pp.max().round(1)
        stats['mean'] = self.delta_dvh_pp.mean().round(1)
        stats['std'] = self.delta_dvh_pp.std(ddof=1).round(1)
        return stats

    @property
    def stats_delta_cc(self):
        stats = {}
        stats['min'] = self.delta_dvh.min().round(1)
        stats['max'] = self.delta_dvh.max().round(1)
        stats['mean'] = self.delta_dvh.mean().round(1)
        stats['std'] = self.delta_dvh.std(ddof=1).round(1)
        return stats

    # def get_constrains(self, constrains_dict):
    #     ref_constrains = eval_constrains_dict(self.ref_dvh_dict, constrains_dict)
    #     calc_constrains = eval_constrains_dict(self.calc_dvh_dict, constrains_dict)
    #
    #     return ref_constrains, calc_constrains

    def eval_range(self, lim=0.2):
        t1 = self.delta_dvh < -lim
        t2 = self.delta_dvh > lim
        ok = np.sum(np.logical_or(t1, t2))

        pp = ok / len(self.delta_dvh) * 100
        print('pp %1.2f - %i of %i ' % (pp, ok, self.delta_dvh.size))

    def plot_results(self, ref_label, calc_label, title, dest_folder='', fig=None, ax=None):
        if not fig and not ax:
            fig, ax = plt.subplots()

        ref = self.ref_dvh(self.dose_samples)
        calc = self.calc_dvh(self.dose_samples)
        ax.plot(self.dose_samples, ref, label=ref_label)
        ax.plot(self.dose_samples, calc, label=calc_label)
        ax.set_ylabel('volume [cc]')
        ax.set_xlabel('Dose [Gy]')
        ax.set_title(title)
        ax.legend(loc='best')
        if dest_folder:
            name = title + '.png'
            fig.savefig(os.path.join(dest_folder, name), format='png', dpi=100)


def calc_data(row, dose_files_dict, structure_dict, constrains, calc_grid):
    struc_name = row[1]['Structure name'] + '.dcm'
    gradient_direction = row[1]['Gradient direction']
    dose_voxel = row[1]['Dose Voxel (mm)']
    py_struc = structure_dict[struc_name]
    dose_obj = dose_files_dict[gradient_direction][dose_voxel]
    dvh_calculator = DVHCalculation(py_struc, dose_obj, calc_grid=calc_grid)
    dvh = dvh_calculator.calculate(True)
    dvh_item = DVHMetrics(dvh)
    constraints_results = [dvh_item.execute_query(q) for q in constrains]
    constraints_results = [dvh_item.volume] + constraints_results
    return {row[0]: constraints_results}


def test_1(test1_calc_data, constraints_data, results_folder, grid=0.1, lim=3, save_data=False, ):
    """
    In Test 1, the axial contour spacing was kept constant at
    0.2 mm to essentially eliminate the variation and/or errors
    associated with rendering axial contours into volumes, and to
    focus solely on the effect of altering the dose grid resolution
    in various stages from fine (0.4 × 0.2 × 0.4 mm 3 ) to coarse (3
    × 3 × 3 mm 3 ).
    Analytical results for the following parameters
    per structure were compared to both PlanIQ (with supersam-
    pling turned on: Ref. 20) and PINCACLE: total volume (V );
    mean, maximum, and minimum dose (D mean , D max , D min );
    near-maximum (D1, dose covering 1% of the volume) and
    near-minimum (D99) doses; D95 and D5; and maximum dose
    to a small absolute (0.03 cm 3 ) volume (D0.03 cm 3 ). We were
    primarily interested in the high and low dose regions because
    with the linear dose gradient, they correspond to the structure
    boundary and this is where the deviations are expected to occur.

    Results of Test 1. Dose grid resolution is varied while axial contour
    spacing is kept at 0.2 mm. Numbers of points (n) exceeding 3% difference
    (∆) from analytical are presented along with the range of % ∆. Total number
    of structure/dose combinations is N = 40 (20 for V ).

    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  2.2min finished

    voxel (0.1, 0.1, 0.1) mm

                      count        range
    Total volume (cc)     0  [-0.7, 0.5]
    Dmin                  0  [-0.1, 2.6]
    Dmax                  0  [-0.4, 0.0]
    Dmean                 0  [-0.2, 0.3]
    D99                   0  [-1.9, 1.9]
    D95                   0  [-1.3, 0.4]
    D5                    0  [-0.3, 0.2]
    D1                    0  [-0.1, 0.2]
    D0.03cc               8  [-0.1, 5.8]

    #TODO Check interpotion at small D(0.03) cc

    """

    # DICOM FILES
    structure_dict, dose_files_dict, df = test1_calc_data
    calc_grid = (grid, grid, grid)
    # grab analytical data

    # loop
    result = {}
    for row in df.iterrows():
        struc_name = row[1]['Structure name'] + '.dcm'
        gradient_direction = row[1]['Gradient direction']
        dose_voxel = row[1]['Dose Voxel (mm)']
        print(row)
        py_struc = structure_dict[struc_name]
        if isinstance(dose_voxel, int):
            dose_voxel = str(dose_voxel)
        dose_obj = dose_files_dict[gradient_direction][dose_voxel]
        dvh_calculator = DVHCalculation(py_struc, dose_obj, calc_grid=calc_grid)
        dvh = dvh_calculator.calculate(True)
        dvh_item = DVHMetrics(dvh)
        constraints_results = [dvh_item.execute_query(q) for q in constraints_data]
        constraints_results = [dvh_item.volume] + constraints_results
        result[row[0]] = constraints_results

    result_df = pd.DataFrame.from_dict(result).T

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']
    num_col = ['Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = df[num_col]

    result_df.columns = df_num.columns
    result_num = result_df.astype(float)

    delta = ((result_num - df_num) / df_num) * 100

    res = OrderedDict()
    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 1), round(delta[col].max(), 1)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T

    # dest = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\tests\tests_validation\results'
    result_num.to_excel(os.path.join(results_folder, 'Test_1_result.xls'))
    test_table.to_excel(os.path.join(results_folder, 'test_1_table_paper.xls'))

    print(test_table)


def test_2(test2_calc_data, constraints_data, results_folder, grid=0.1, lim=3, save_data=False):
    # DICOM FILES
    structure_dict, dose_files_dict, df = test2_calc_data
    calc_grid = (grid, grid, grid)
    # grab analytical data

    # loop
    result = {}
    for row in df.iterrows():
        struc_name = row[1]['Structure name'] + '.dcm'
        gradient_direction = row[1]['Gradient direction']
        dose_voxel = row[1]['Dose Voxel (mm)']
        print(row)
        py_struc = structure_dict[struc_name]
        if isinstance(dose_voxel, int):
            dose_voxel = str(dose_voxel)
        dose_obj = dose_files_dict[gradient_direction][dose_voxel]
        dvh_calculator = DVHCalculation(py_struc, dose_obj, calc_grid=calc_grid)
        dvh = dvh_calculator.calculate(True)
        dvh_item = DVHMetrics(dvh)
        constraints_results = [dvh_item.execute_query(q) for q in constraints_data]
        constraints_results = [dvh_item.volume] + constraints_results
        result[row[0]] = constraints_results

    result_df = pd.DataFrame.from_dict(result).T

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']
    num_col = ['Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = df[num_col]

    result_df.columns = df_num.columns
    result_num = result_df.astype(float)

    delta = ((result_num - df_num) / df_num) * 100

    res = OrderedDict()
    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 1), round(delta[col].max(), 1)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T
    result_num.to_excel(os.path.join(results_folder, 'Test_2_result.xls'))
    test_table.to_excel(os.path.join(results_folder, 'test_2_table_paper.xls'))

    print(test_table)


def test_3(test3_calc_data, analytical_curves, results_folder, grid=0.1):
    """
                       Gradient Resolution (mm)  max  mean  min  std
    Sphere_10_0        Z(AP)               1 -0.0  -0.2 -0.3  0.1
    Sphere_10_0        Y(SI)               1  0.0  -0.2 -0.4  0.2
    Sphere_30_0        Z(AP)               3 -0.0  -1.1 -1.9  0.6
    Sphere_30_0        Y(SI)               3  0.3  -1.0 -2.1  0.7
    Cylinder_10_0      Z(AP)               1  0.5   0.3 -0.1  0.2
    Cylinder_10_0      Y(SI)               1  0.3   0.2 -0.0  0.1
    Cylinder_30_0      Z(AP)               3  0.3   0.1  0.0  0.1
    Cylinder_30_0      Y(SI)               3  0.3   0.2 -0.0  0.1
    RtCylinder_10_0    Z(AP)               1  0.5   0.3  0.0  0.1
    RtCylinder_10_0    Y(SI)               1  0.3   0.0 -0.2  0.1
    RtCylinder_30_0    Z(AP)               3  0.7   0.5  0.3  0.1
    RtCylinder_30_0    Y(SI)               3  1.1   0.3 -0.5  0.4
    Cone_10_0          Z(AP)               1  0.3   0.1 -0.1  0.2
    Cone_10_0          Y(SI)               1  1.0   0.3  0.2  0.2
    Cone_30_0          Z(AP)               3 -0.0  -0.8 -1.4  0.4
    Cone_30_0          Y(SI)               3  0.9  -1.3 -1.5  0.5
    RtCone_10_0        Z(AP)               1  0.0  -1.1 -1.4  0.3
    RtCone_10_0        Y(SI)               1  0.0  -1.0 -1.4  0.5
    RtCone_30_0        Z(AP)               3 -0.1  -2.5 -3.9  1.3
    RtCone_30_0        Y(SI)               3  0.5  -2.5 -4.4  1.8
    Average (N = 5)    Y(SI)               1  0.3  -0.1 -0.4  0.2
    Average (N = 5)    Z(AP)               1  0.3  -0.1 -0.4  0.2
    Average (N = 5)    Y(SI)               3  0.6  -0.9 -1.7  0.7
    Average (N = 5)    Z(AP)               3  0.2  -0.8 -1.4  0.5

    """

    # DICOM FILES
    structure_dict, dose_files_dict = test3_calc_data
    calc_grid = (grid, grid, grid)

    # adapters
    struc_adapter = lambda s: s.split('_')[0]
    grad_adapter = {'Z(AP)': 'AP', 'Y(SI)': 'SI'}
    # loop on all test data
    result = {}
    curve_compare = []
    teste = []
    for struc_name, struc_py in structure_dict.items():
        # dvhs_per_direction = {}
        for grad_name, dose_grids in dose_files_dict.items():
            # dvh_per_grid = {}
            for dose_grid_name, dose_3d in dose_grids.items():
                # match dose grid and contour resolution
                tick = str(int(int(re.findall(r'\d+', struc_name)[0]) / 10))
                if tick == dose_grid_name:
                    dvh_calculator = DVHCalculation(struc_py, dose_3d, calc_grid=calc_grid)
                    dvh = dvh_calculator.calculate(True)
                    calc_data = DVHData(dvh)
                    # curve compare
                    # grab analytical data
                    sname = struc_adapter(struc_name)
                    query = grad_adapter[grad_name] + ' ' + dose_grid_name + ' mm'
                    an_dvh_df = analytical_curves[sname][query]
                    an_dose_axis = analytical_curves[sname]['Dose (cGy)'] / 100.0

                    # get curve compare data
                    cmp = CurveCompare(an_dose_axis, an_dvh_df, calc_data.dose_axis, calc_data.volume_cc)
                    cmp.plot_results('Analytical', 'PyPlanScoring', struc_name[:-4] + ' - ' + query, results_folder)
                    curve_stats = cmp.stats_paper
                    curve_stats['Resolution (mm)'] = dose_grid_name
                    curve_stats['Gradient'] = grad_adapter[grad_name]
                    curve_compare.append(cmp)
                    tmp = pd.DataFrame(curve_stats, index=[struc_name[:-4]])
                    teste.append(tmp)
                # dvh_per_grid[dose_grid_name] = calc_data

            # dvhs_per_direction[grad_name] = dvh_per_grid

        # result[struc_name] = dvhs_per_direction

    df_final = pd.concat(teste)

    mask0 = np.logical_and(df_final['Resolution (mm)'] == '1', df_final['Gradient'] == 'SI')
    mask1 = np.logical_and(df_final['Resolution (mm)'] == '1', df_final['Gradient'] == 'AP')
    mask2 = np.logical_and(df_final['Resolution (mm)'] == '3', df_final['Gradient'] == 'SI')
    mask3 = np.logical_and(df_final['Resolution (mm)'] == '3', df_final['Gradient'] == 'AP')

    # Row 0
    r0 = pd.DataFrame(['SI'], index=['Average (N = 5)'], columns=['Gradient'])
    r0['Resolution (mm)'] = '1'
    ri = pd.DataFrame(df_final[mask0].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r0 = r0.join(ri)

    # Row 1
    r1 = pd.DataFrame(['AP'], index=['Average (N = 5)'], columns=['Gradient'])
    r1['Resolution (mm)'] = '1'
    ri = pd.DataFrame(df_final[mask1].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r1 = r1.join(ri)

    # Row 2
    r2 = pd.DataFrame(['SI'], index=['Average (N = 5)'], columns=['Gradient'])
    r2['Resolution (mm)'] = '3'
    ri = pd.DataFrame(df_final[mask2].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r2 = r2.join(ri)

    # Row 3
    r3 = pd.DataFrame(['AP'], index=['Average (N = 5)'], columns=['Gradient'])
    r3['Resolution (mm)'] = '3'
    ri = pd.DataFrame(df_final[mask3].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r3 = r3.join(ri)
    result_df = pd.concat([df_final, r0, r1, r2, r3])
    # SAVE DATA
    df_final.to_excel(os.path.join(results_folder, 'test_3_results.xls'))
    result_df.to_excel(os.path.join(results_folder, 'test_3_table.xls'))



def test_dicompyler(test_dicompyler_data, analytical_curves, results_folder, grid=0.1):
    #
    from dicompylercore import dvhcalc
    results_folder = os.path.join(results_folder, 'dicompyler')

    # DICOM FILES
    structure_dict, doses_dict, structure_files_dict, doses_files_dict = test_dicompyler_data
    calc_grid = (grid, grid, grid)

    # adapters
    struc_adapter = lambda s: s.split('_')[0]
    grad_adapter = {'Z(AP)': 'AP', 'Y(SI)': 'SI'}
    # loop on all test data
    result = {}
    curve_compare = []
    teste = []
    for struc_name, struc_py in structure_dict.items():
        # dvhs_per_direction = {}
        for grad_name, dose_grids in doses_dict.items():
            # dvh_per_grid = {}
            for dose_grid_name, dose_3d in dose_grids.items():
                # match dose grid and contour resolution
                tick = str(int(int(re.findall(r'\d+', struc_name)[0]) / 10))
                if tick == dose_grid_name:
                    # calculating using pps
                    dvh_calculator = DVHCalculation(struc_py, dose_3d, calc_grid=calc_grid)
                    dvh = dvh_calculator.calculate(True)
                    calc_data = DVHData(dvh)
                    # calculating using dicompyler core

                    # Calculate a DVH from DICOM RT data
                    struc_file = structure_files_dict[struc_name]
                    dose_file = doses_files_dict[grad_name][dose_grid_name]
                    calcdvh = dvhcalc.get_dvh(struc_file, dose_file, 2)
                    dp_dose = calcdvh.cumulative.bincenters
                    dp_vol = calcdvh.cumulative.counts
                    fig, ax = plt.subplots()
                    ax.plot(dp_dose, dp_vol, label='dicompyler-core')

                    # curve compare
                    # grab analytical data
                    sname = struc_adapter(struc_name)
                    query = grad_adapter[grad_name] + ' ' + dose_grid_name + ' mm'
                    an_dvh_df = analytical_curves[sname][query]
                    an_dose_axis = analytical_curves[sname]['Dose (cGy)'] / 100.0

                    # get curve compare data
                    cmp = CurveCompare(an_dose_axis, an_dvh_df, calc_data.dose_axis, calc_data.volume_cc)
                    cmp.plot_results('Analytical', 'PyPlanScoring', struc_name[:-4] + ' - ' + query, results_folder,
                                     fig=fig, ax=ax)

                # dvh_per_grid[dose_grid_name] = calc_data

            # dvhs_per_direction[grad_name] = dvh_per_grid

        # result[struc_name] = dvhs_per_direction
