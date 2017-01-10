from dicomparser import ScoringDicomParser
from dvhcalc import get_dvh
from dev.dvhcalculation import Structure
import matplotlib.pyplot as plt
import numpy as np
import time
import numba as nb
import cv2
from math import factorial
import numpy
import pandas as pd
import scipy.interpolate as itp

from scipy.signal import savgol_filter

from dvhdoses import get_dvh_max
from dvhdoses import get_dvh_min, get_dvh_mean
from scoring import DVHMetrics
from collections import OrderedDict


class CurveCompare(object):
    """
        Statistical analysis of the DVH volume (%) error histograms. volume (cm 3 ) differences (numerical–analytical)
        were calculated for points on the DVH curve sampled at every 10 cGy then normalized to
        the structure's total volume (cm 3 ) to give the error in volume (%)
    """

    def __init__(self, a_dose, a_dvh, calc_dose, calc_dvh):
        self.a_dose = a_dose
        self.a_dvh = a_dvh
        self.cal_dose = calc_dose
        self.calc_dvh = calc_dvh
        self.dose_samples = np.arange(0, len(calc_dvh), 10)  # The DVH curve sampled at every 10 cGy
        self.ref_dvh = itp.interp1d(a_dose, advh)
        self.calc_dvh = itp.interp1d(calc_dose, calc_dvh)
        self.delta_dvh = self.calc_dvh(self.dose_samples) - self.ref_dvh()

    def stats(self):
        df = pd.DataFrame(self.delta_dvh)
        print(df.describe())

    def eval_range(self, lim=0.2):
        t1 = self.delta_dvh > -lim
        t2 = self.delta_dvh < lim
        ok = np.sum(np.logical_and(t1, t2))

        pp = ok / len(self.delta_dvh) * 100
        print('pp %1.2f - %i of %i ' % (pp, ok, self.delta_dvh.size))

    def plot_results(self):
        # PLOT HISTOGRAM AND DELTA
        pass


def test1():
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
    of structure/dose combinations is N = 40 (20 for V )."""

    # TEST DICOM DATA
    structure_files = ['/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/Cylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/RtCylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/Cone_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_02_0.dcm']

    structure_name = ['Sphere_02_0', 'Cylinder_02_0', 'RtCylinder_02_0', 'Cone__02_0', 'RtCone_02_0']

    dose_files = [r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_1mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_2mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_1mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_2mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm']

    dose_headers = ['0.4x0.2x0.4', '1', '2', '3'] * 2

    # Structure Dict

    structure_dict = dict(zip(structure_name, structure_files))

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'0.4x0.2x0.4': dose_files[0], '1': dose_files[1], '2': dose_files[2], '3': dose_files[3]},
        'Y(SI)': {'0.4x0.2x0.4': dose_files[4], '1': dose_files[5], '2': dose_files[6], '3': dose_files[7]}}

    # grab analytical data
    sheet = 'Analytical'
    df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx', sheetname=sheet)
    mask = df['CT slice spacing (mm)'] == '0.2mm'
    df = df.loc[mask]

    # Constrains to get data

    # Constrains

    constrains = OrderedDict()
    constrains['Total_Volume'] = True
    constrains['min'] = 'min'
    constrains['max'] = 'max'
    constrains['mean'] = 'mean'
    constrains['D99'] = 99
    constrains['D95'] = 95
    constrains['D5'] = 5
    constrains['D1'] = 1
    constrains['Dcc'] = 0.03

    st = 2
    up = (0.4, 0.4, 0.2)
    df_concat = []
    sname = []
    # GET CALCULATED DATA
    for row in df.iterrows():
        idx, values = row[0], row[1]
        s_name = values['Structure name']
        voxel = str(values['Dose Voxel (mm)'])
        gradient = values['Gradient direction']

        dose_file = dose_files_dict[gradient][voxel]
        struc_file = structure_dict[s_name]

        # get structure and dose
        dicom_dose = ScoringDicomParser(filename=dose_file)
        struc = ScoringDicomParser(filename=struc_file)
        structures = struc.GetStructures()
        structure = structures[st]

        # set up sampled structure
        struc_teste = Structure(structure, end_cap=True)
        dhist, chist = struc_teste.calculate_dvh(dicom_dose, upsample=True, delta_cm=up)
        dvh_data = prepare_dvh_data(dhist, chist)

        # Setup DVH metrics class and get DVH DATA
        metrics = DVHMetrics(dvh_data)
        values_constrains = OrderedDict()
        for k in constrains.keys():
            ct = metrics.eval_constrain(k, constrains[k])
            values_constrains[k] = ct
        values_constrains['Gradient direction'] = gradient

        # Get data
        df_concat.append(pd.Series(values_constrains, name=voxel))
        sname.append(s_name)

    result = pd.concat(df_concat, axis=1).T.reset_index()
    result['Structure name'] = sname

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    num_col = ['Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = df[num_col]

    result_num = result[result.columns[1:-2]]
    result_num.columns = df_num.columns

    delta = ((result_num - df_num) / df_num) * 100

    res = OrderedDict()
    lim = 3.0
    for col in delta:
        t0 = delta[col] > lim
        t1 = delta[col] > -lim
        count = np.logical_and(t0, t1).sum()
        rg = np.array([delta[col].min(), delta[col].max()])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T

    return test_table


def prepare_dvh_data(dhist, dvh):
    dvhdata = {}
    dvhdata['data'] = dvh
    dvhdata['bins'] = len(dvh)
    dvhdata['type'] = 'CUMULATIVE'
    dvhdata['doseunits'] = 'GY'
    dvhdata['volumeunits'] = 'CM3'
    dvhdata['scaling'] = np.diff(dhist)[0]
    dvhdata['min'] = get_dvh_min(dvh)
    dvhdata['max'] = get_dvh_max(dvh)
    dvhdata['mean'] = get_dvh_mean(dvh)
    return dvhdata


if __name__ == '__main__':

    # TEST DICOM DATA
    structure_files = ['/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/Cylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/RtCylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/Cone_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_02_0.dcm']

    structure_name = ['Sphere_02_0', 'Cylinder_02_0', 'RtCylinder_02_0', 'Cone__02_0', 'RtCone_02_0']

    dose_files = [r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_1mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_2mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_1mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_2mm_Aligned.dcm',
                  r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm']

    dose_headers = ['0.4x0.2x0.4', '1', '2', '3'] * 2

    # Structure Dict

    structure_dict = dict(zip(structure_name, structure_files))

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'0.4x0.2x0.4': dose_files[0], '1': dose_files[1], '2': dose_files[2], '3': dose_files[3]},
        'Y(SI)': {'0.4x0.2x0.4': dose_files[4], '1': dose_files[5], '2': dose_files[6], '3': dose_files[7]}}

    # grab analytical data
    sheet = 'Analytical'
    df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx', sheetname=sheet)
    mask = df['CT slice spacing (mm)'] == '0.2mm'
    df = df.loc[mask]

    # Constrains to get data

    # Constrains

    constrains = OrderedDict()
    constrains['Total_Volume'] = True
    constrains['min'] = 'min'
    constrains['max'] = 'max'
    constrains['mean'] = 'mean'
    constrains['D99'] = 99
    constrains['D95'] = 95
    constrains['D5'] = 5
    constrains['D1'] = 1
    constrains['Dcc'] = 0.03

    st = 2
    up = (0.2, 0.2, 0.1)
    df_concat = []
    sname = []
    # GET CALCULATED DATA
    for row in df.iterrows():
        idx, values = row[0], row[1]
        s_name = values['Structure name']
        voxel = str(values['Dose Voxel (mm)'])
        gradient = values['Gradient direction']

        dose_file = dose_files_dict[gradient][voxel]
        struc_file = structure_dict[s_name]

        # get structure and dose
        dicom_dose = ScoringDicomParser(filename=dose_file)
        struc = ScoringDicomParser(filename=struc_file)
        structures = struc.GetStructures()
        structure = structures[st]

        # set up sampled structure
        struc_teste = Structure(structure, end_cap=True)
        dhist, chist = struc_teste.calculate_dvh(dicom_dose, upsample=True, delta_cm=up)
        dvh_data = prepare_dvh_data(dhist, chist)

        # Setup DVH metrics class and get DVH DATA
        metrics = DVHMetrics(dvh_data)
        values_constrains = OrderedDict()
        for k in constrains.keys():
            ct = metrics.eval_constrain(k, constrains[k])
            values_constrains[k] = ct
        values_constrains['Gradient direction'] = gradient

        # Get data
        df_concat.append(pd.Series(values_constrains, name=voxel))
        sname.append(s_name)

    result = pd.concat(df_concat, axis=1).T.reset_index()
    result['Structure name'] = sname

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    num_col = ['Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = df[num_col]

    result_num = result[result.columns[1:-2]]
    result_num.columns = df_num.columns

    delta = ((result_num - df_num) / df_num) * 100

    pcol = ['Total Volume (cc)', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1']

    res = OrderedDict()
    lim = 3.0
    for col in delta:
        t0 = delta[col] > lim
        t1 = delta[col] > -lim
        count = np.logical_and(t0, t1).sum()
        rg = np.array([delta[col].min(), delta[col].max()])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T





    #
    # volumes = ['Sphere', 'Axial_Cylinder', 'RT_Cylinder', 'Axial_Cone', 'RT_Cone']
    # files = ['/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_20_0.dcm',
    #          '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/Cylinder_20_0.dcm',
    #          '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/RtCylinder_20_0.dcm',
    #          '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/Cone_20_0.dcm',
    #          '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_20_0.dcm']
    #
    # doses = ['/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm',
    #          '/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm']
    #
    # sheet = 'Sphere'
    #
    # dose_file = '/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_2mm_Aligned.dcm'
    # dose = ScoringDicomParser(filename=dose_file)
    # up = (0.4, 0.4, 0.2)
    # st = 2
    #
    # for i in range(len(volumes)):
    #     sheet = volumes[i]
    #     f = files[i]
    #     df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx', sheetname=sheet)
    #     adose = df['Dose (cGy)'].values
    #     advh = df['SI 2 mm'].values
    #     # structure
    #     struc = ScoringDicomParser(filename=f)
    #     structures = struc.GetStructures()
    #     structure = structures[st]
    #
    #     dicompyler_dvh = get_dvh(structure, dose)
    #     struc_teste = Structure(structure, end_cap=True)
    #     dhist, chist = struc_teste.calculate_dvh(dose, bin_size=1, upsample=True, delta_cm=up)
    #     plt.figure()
    #     plt.plot(dhist, chist, label='Up Sampled Structure')
    #     plt.plot(dhist, struc_teste.smoothed_dvh, label='Smooth')
    #     plt.hold(True)
    #     plt.plot(dicompyler_dvh['data'], label='Not up sampled')
    #     plt.plot(adose, advh, label='Analytical')
    #     plt.title('Structure Name: %s - volume (cc) %1.3f' % (struc_teste.name, struc_teste.volume_cc))
    #     plt.legend(loc='best')
    #
    # plt.show()
    #
    # rs = '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/RtCylinder_10_0.dcm'
    # dose_file = '/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_1mm_Aligned.dcm'
    # dose = ScoringDicomParser(filename=dose_file)
    #
    # sheet = 'RT_Cylinder'
    # df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx', sheetname=sheet)
    # adose = df['Dose (cGy)'].values
    # advh = df['SI 1 mm'].values
    # st = 2
    # up = (0.4, 0.4, 0.1)
    #
    # # structure
    # struc = ScoringDicomParser(filename=rs)
    # structures = struc.GetStructures()
    # structure = structures[st]
    #
    # dicompyler_dvh = get_dvh(structure, dose)
    # struc_teste = Structure(structure, end_cap=True)
    # dhist, chist = struc_teste.calculate_dvh(dose, bin_size=1, upsample=True, delta_cm=up)
    # plt.figure()
    # plt.plot(dhist, chist, label='Up Sampled Structure')
    # # plt.plot(dhist, struc_teste.smoothed_dvh, label='Smooth')
    # plt.hold(True)
    # plt.plot(dicompyler_dvh['data'], label='Not up sampled')
    # plt.plot(adose, advh, label='Analytical')
    # plt.title('Structure Name: %s - volume (cc) %1.3f' % (struc_teste.name, struc_teste.volume_cc))
    # plt.legend(loc='best')
    #
    # fx = itp.interp1d(adose, advh, kind='linear')
    #
    # calc_interp = itp.interp1d(dhist, chist, kind='linear')
    #
    # # points comparison
    #
    # eval_doses = np.arange(0, len(chist), 10)
    #
    # ref_dvh = fx(eval_doses)
    # calc_dvh = calc_interp(eval_doses)
    #
    # plt.plot(eval_doses, ref_dvh)
    # plt.plot(eval_doses, calc_dvh)
    #
    # slc = -1
    # dt = (calc_dvh[:slc] - ref_dvh[:slc]) / ref_dvh.max() * 100
    #
    # plt.plot(dhist, chist)
    # plt.plot(dhist, fx(dhist))
    #
    # cmp = CurveCompare(adose, advh, dhist, chist)
    # cmp.stats()
    # cmp.eval_range()
