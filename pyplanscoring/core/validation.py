from __future__ import division

import configparser
import logging
import os
import re
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.interpolate as itp
from joblib import Parallel
from joblib import delayed
from matplotlib import pyplot as plt

from pyplanscoring.core.dicomparser import ScoringDicomParser
from pyplanscoring.core.dosimetric import read_scoring_criteria, constrains, Competition2016
from pyplanscoring.core.dvhcalculation import Structure, prepare_dvh_data, calc_dvhs_upsampled, save_dicom_dvhs, load
from pyplanscoring.core.dvhdoses import get_dvh_max
from pyplanscoring.core.geometry import get_axis_grid, get_interpolated_structure_planes
from pyplanscoring.core.scoring import DVHMetrics, Scoring, Participant
from pyplanscoring.lecacy_dicompyler.dvhcalc import calc_dvhs
from pyplanscoring.lecacy_dicompyler.dvhcalc import load

logger = logging.getLogger('validation')

logging.basicConfig(filename='plan_competition_2016_no_dicom_DVH.log', level=logging.DEBUG)

# Get calculation defaults
folder = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core'
config = configparser.ConfigParser()
config.read(os.path.join(folder, 'validation.ini'))
calculation_options = dict()
calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
calculation_options['use_tps_structures'] = config.getboolean('DEFAULT', 'use_tps_structures')
calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']


# TODO extract constrains from analytical curves

class CurveCompare(object):
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
        self.sampling_size = 10
        self.dose_samples = np.arange(0, len(calc_dvh), self.sampling_size)  # The DVH curve sampled at every 10 cGy
        self.ref_dvh = itp.interp1d(a_dose, a_dvh, fill_value='extrapolate')
        self.calc_dvh = itp.interp1d(calc_dose, calc_dvh, fill_value='extrapolate')
        self.delta_dvh = self.calc_dvh(self.dose_samples) - self.ref_dvh(self.dose_samples)
        self.delta_dvh_pp = (self.delta_dvh / a_dvh[0]) * 100
        # prepare data dict
        self.calc_dvh_dict = prepare_dvh_data(self.dose_samples, self.calc_dvh(self.dose_samples))
        self.ref_dvh_dict = prepare_dvh_data(self.dose_samples, self.ref_dvh(self.dose_samples))
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

    def get_constrains(self, constrains_dict):
        ref_constrains = eval_constrains_dict(self.ref_dvh_dict, constrains_dict)
        calc_constrains = eval_constrains_dict(self.calc_dvh_dict, constrains_dict)

        return ref_constrains, calc_constrains

    def eval_range(self, lim=0.2):
        t1 = self.delta_dvh < -lim
        t2 = self.delta_dvh > lim
        ok = np.sum(np.logical_or(t1, t2))

        pp = ok / len(self.delta_dvh) * 100
        print('pp %1.2f - %i of %i ' % (pp, ok, self.delta_dvh.size))

    def plot_results(self):
        fig, ax = plt.subplots()
        ref = self.ref_dvh(self.dose_samples)
        calc = self.calc_dvh(self.dose_samples)
        ax.plot(self.dose_samples, ref, label='Analytical')
        ax.plot(self.dose_samples, calc, label='App')
        ax.set_ylabel('Volume (cc)')
        ax.set_xlabel('Dose (cGy)')
        txt = self.structure_name + ' ' + self.doseF_grid + ' mm ' + self.gradient
        ax.set_title(txt)
        ax.legend(loc='best')


def test_real_dvh():
    rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/RS.1.2.246.352.71.4.584747638204.248648.20170123083029.dcm'
    rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/RD.1.2.246.352.71.7.584747638204.1750110.20170123082607.dcm'
    rp = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/RP.1.2.246.352.71.5.584747638204.952069.20170122155706.dcm'
    # dvh_file = r'/media/victor/TOURO Mobile/COMPETITION 2017/Send to Victor - Jan10 2017/Norm Res with CT Images/RD.1.2.246.352.71.7.584747638204.1746016.20170110164605.dvh'

    f = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/PlanIQ Criteria TPS PlanIQ matched str names - TXT Fromat - Last mod Jan23.txt'
    constrains_all, scores_all, criteria = read_scoring_criteria(f)

    dose = ScoringDicomParser(filename=rd_file)
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()

    ecl_DVH = dose.GetDVHs()
    plt.style.use('ggplot')
    st = time.time()
    dvhs = {}

    for structure in structures.values():
        for end_cap in [False]:
            if structure['id'] in ecl_DVH:
                # if structure['id'] in [37, 38]:
                if structure['name'] in list(scores_all.keys()):
                    ecl_dvh = ecl_DVH[structure['id']]['data']
                    ecl_dmax = ecl_DVH[structure['id']]['max'] * 100  # to cGy
                    struc_teste = Structure(structure, end_cap=end_cap)
                    # struc['planes'] = struc_teste.planes
                    # dicompyler_dvh = get_dvh(structure, dose)
                    fig, ax = plt.subplots()
                    fig.set_figheight(12)
                    fig.set_figwidth(20)
                    dhist, chist = struc_teste.calculate_dvh(dose, up_sample=True)
                    max_dose = get_dvh_max(chist)

                    ax.plot(dhist, chist, label='Up sampled - Dmax: %1.1f cGy' % max_dose)
                    fig.hold(True)
                    ax.plot(ecl_dvh, label='Eclipse - Dmax: %1.1f cGy' % ecl_dmax)
                    dvh_data = prepare_dvh_data(dhist, chist)

                    txt = structure['name'] + ' volume (cc): %1.1f - end_cap: %s ' % (
                        ecl_dvh[0], str(end_cap))
                    ax.set_title(txt)
                    # nup = get_dvh_max(dicompyler_dvh['data'])
                    # plt.plot(dicompyler_dvh['data'], label='Software DVH - Dmax: %1.1f cGy' % nup)
                    ax.legend(loc='best')
                    ax.set_xlabel('Dose (cGy)')
                    ax.set_ylabel('Volume (cc)')
                    fname = txt + '.png'
                    fig.savefig(fname, format='png', dpi=100)
                    dvhs[structure['name']] = dvh_data

    end = time.time()

    print('Total elapsed Time (min):  ', (end - st) / 60)


def test_spacing(root_path):
    """
        # TEST PLANIQ RS-DICOM DATA if z planes are not equal spaced.

    :param root_path: root path
    """

    root_path = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/STRUCTURES'

    structure_files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
                       name.endswith(('.dcm', '.DCM'))]

    eps = 0.001

    test_result = {}
    for f in structure_files:
        structures = ScoringDicomParser(filename=f).GetStructures()
        for key in structures:
            try:
                all_z = np.array([z for z in structures[key]['planes'].keys()], dtype=float)
                all_sorted_diff = np.diff(np.sort(all_z))
                test = (abs((all_sorted_diff - all_sorted_diff[0])) > eps).any()
                test_result[structures[key]['name']] = test
            except:
                print('Error in key:', key)

    b = {key: value for key, value in test_result.items() if value == True}

    return test_result


def test_planes_spacing(sPlanes):
    eps = 0.001
    all_z = np.array([z for z in sPlanes], dtype=float)
    all_sorted_diff = np.diff(np.sort(all_z))
    test = (abs((all_sorted_diff - all_sorted_diff[0])) > eps).any()
    return test, all_sorted_diff


def test_upsampled_z_spacing(sPlanes):
    z = 0.1
    ordered_keys = [z for z, sPlane in sPlanes.items()]
    ordered_keys.sort(key=float)
    ordered_planes = np.array(ordered_keys, dtype=float)
    z_interp_positions, dz = get_axis_grid(z, ordered_planes)
    hi_res_structure = get_interpolated_structure_planes(sPlanes, z_interp_positions)

    ordered_keys = [z for z, sPlane in hi_res_structure.items()]
    ordered_keys.sort(key=float)
    t, p = test_planes_spacing(hi_res_structure)

    assert t is False


def eval_constrains_dict(dvh_data_tmp, constrains_dict):
    mtk = DVHMetrics(dvh_data_tmp)
    values_tmp = OrderedDict()
    for ki in constrains_dict.keys():
        cti = mtk.eval_constrain(ki, constrains_dict[ki])
        values_tmp[ki] = cti

    return values_tmp


def get_analytical_curve(an_curves_obj, file_structure_name, column):
    an_curve_i = an_curves_obj[file_structure_name.split('_')[0]]
    dose_an = an_curve_i['Dose (cGy)'].values
    an_dvh = an_curve_i[column].values  # check nonzero

    idx = np.nonzero(an_dvh)  # remove 0 volumes from DVH
    dose_range, cdvh = dose_an[idx], an_dvh[idx]

    return dose_range, cdvh


def calc_data(row, dose_files_dict, structure_dict, constrains, calculation_options):
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
    structure = structures[2]

    # set end cap by 1/2 slice thickness
    calculation_options['end_cap'] = structure['thickness'] / 2.0

    # set up sampled structure
    struc_teste = Structure(structure, calculation_options)
    dhist, chist = struc_teste.calculate_dvh(dicom_dose)
    dvh_data = struc_teste.get_dvh_data()

    # Setup DVH metrics class and get DVH DATA
    metrics = DVHMetrics(dvh_data)
    values_constrains = OrderedDict()
    for k in constrains.keys():
        ct = metrics.eval_constrain(k, constrains[k])
        values_constrains[k] = ct
    values_constrains['Gradient direction'] = gradient

    # Get data

    return pd.Series(values_constrains, name=voxel), s_name


def calc_data_all(row, dose_files_dict, structure_dict, constrains, an_curves, col_grad_dict, delta_mm=(0.2, 0.2, 0.2),
                  end_cap=True, up_sample=True):
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
    structure = structures[2]

    # set up sampled structure
    struc_teste = Structure(structure)
    struc_teste.set_delta(delta_mm)
    dhist, chist = struc_teste.calculate_dvh(dicom_dose)

    # get its columns from spreadsheet
    column = col_grad_dict[gradient][voxel]
    adose_range, advh = get_analytical_curve(an_curves, s_name, column)

    # use CurveCompare class to eval similarity from calculated and analytical curves

    cmp = CurveCompare(adose_range, advh, dhist, chist, s_name, voxel, gradient)

    ref_constrains, calc_constrains = cmp.get_constrains(constrains)

    ref_constrains['Gradient direction'] = gradient
    calc_constrains['Gradient direction'] = gradient
    ref_series = pd.Series(ref_constrains, name=voxel)
    calc_series = pd.Series(calc_constrains, name=voxel)

    return ref_series, calc_series, s_name, cmp


def test11(delta_mm=(0.2, 0.2, 0.1), plot_curves=False):
    # TEST DICOM DATA
    structure_files = ['/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/Cylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/RtCylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/Cone_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_02_0.dcm']

    structure_name = ['Sphere_02_0', 'Cylinder_02_0', 'RtCylinder_02_0', 'Cone__02_0', 'RtCone_02_0']

    dose_files = [
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_1mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_2mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_1mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_2mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm']

    # Structure Dict

    structure_dict = dict(zip(structure_name, structure_files))

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'0.4x0.2x0.4': dose_files[0], '1': dose_files[1], '2': dose_files[2], '3': dose_files[3]},
        'Y(SI)': {'0.4x0.2x0.4': dose_files[4], '1': dose_files[5], '2': dose_files[6], '3': dose_files[7]}}

    sheets = ['Sphere', 'Cylinder', 'RtCylinder', 'Cone', 'RtCone']
    col_grad_dict = {'Z(AP)': {'0.4x0.2x0.4': 'AP 0.2 mm', '1': 'AP 1 mm', '2': 'AP 2 mm', '3': 'AP 3 mm'},
                     'Y(SI)': {'0.4x0.2x0.4': 'SI 0.2 mm', '1': 'SI 1 mm', '2': 'SI 2 mm', '3': 'SI 3 mm'}}

    # grab analytical data
    sheet = 'Analytical'
    ref_path = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_data.xlsx'
    df = pd.read_excel(ref_path, sheetname=sheet)
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

    # Get all analytical curves
    out = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_dvh.obj'
    an_curves = load(out)

    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(calc_data_all)(row,
                               dose_files_dict,
                               structure_dict,
                               constrains,
                               an_curves,
                               col_grad_dict,
                               delta_mm=delta_mm) for row in df.iterrows())

    ref_results = [d[0] for d in res]
    calc_results = [d[1] for d in res]
    sname = [d[2] for d in res]
    curves = [d[3] for d in res]

    df_ref_results = pd.concat(ref_results, axis=1).T.reset_index()
    df_calc_results = pd.concat(calc_results, axis=1).T.reset_index()
    df_ref_results['Structure name'] = sname
    df_calc_results['Structure name'] = sname

    ref_num = df_ref_results[df_ref_results.columns[1:-2]]
    calc_num = df_calc_results[df_calc_results.columns[1:-2]]

    delta = ((calc_num - ref_num) / ref_num) * 100

    res = OrderedDict()
    lim = 3
    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 2), round(delta[col].max(), 2)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T
    print(test_table)

    if plot_curves:
        for c in curves:
            c.plot_results()
    plt.show()


def test22(delta_mm=(0.1, 0.1, 0.1), up_sample=True, plot_curves=True):
    ref_data = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_data.xlsx'

    struc_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/STRUCTURES'
    dose_grid_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS'
    #
    # ref_data = r'D:\Dropbox\Plan_Competit
    st = 2

    snames = ['Sphere_10_0', 'Sphere_20_0', 'Sphere_30_0',
              'Cylinder_10_0', 'Cylinder_20_0', 'Cylinder_30_0',
              'RtCylinder_10_0', 'RtCylinder_20_0', 'RtCylinder_30_0',
              'Cone_10_0', 'Cone_20_0', 'Cone_30_0',
              'RtCone_10_0', 'RtCone_20_0', 'RtCone_30_0']

    structure_path = [os.path.join(struc_dir, f + '.dcm') for f in snames]

    structure_dict = dict(zip(snames, structure_path))

    dose_files = [os.path.join(dose_grid_dir, f) for f in [
        'Linear_AntPost_1mm_Aligned.dcm',
        'Linear_AntPost_2mm_Aligned.dcm',
        'Linear_AntPost_3mm_Aligned.dcm',
        'Linear_SupInf_1mm_Aligned.dcm',
        'Linear_SupInf_2mm_Aligned.dcm',
        'Linear_SupInf_3mm_Aligned.dcm']]

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'1': dose_files[0], '2': dose_files[1], '3': dose_files[2]},
        'Y(SI)': {'1': dose_files[3], '2': dose_files[4], '3': dose_files[5]}}

    col_grad_dict = {'Z(AP)': {'0.4x0.2x0.4': 'AP 0.2 mm', '1': 'AP 1 mm', '2': 'AP 2 mm', '3': 'AP 3 mm'},
                     'Y(SI)': {'0.4x0.2x0.4': 'SI 0.2 mm', '1': 'SI 1 mm', '2': 'SI 2 mm', '3': 'SI 3 mm'}}

    # grab analytical data
    out = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_dvh.obj'
    an_curves = load(out)

    df = pd.read_excel(ref_data, sheetname='Analytical')

    dfi = df.ix[40:]
    mask0 = dfi['Structure Shift'] == 0
    dfi = dfi.loc[mask0]

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

    # GET CALCULATED DATA
    # backend = 'threading'
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(calc_data_all)(row,
                               dose_files_dict,
                               structure_dict,
                               constrains,
                               an_curves,
                               col_grad_dict,
                               delta_mm=delta_mm,
                               up_sample=up_sample) for row in dfi.iterrows())

    ref_results = [d[0] for d in res]
    calc_results = [d[1] for d in res]
    sname = [d[2] for d in res]
    curves = [d[3] for d in res]

    df_ref_results = pd.concat(ref_results, axis=1).T.reset_index()
    df_calc_results = pd.concat(calc_results, axis=1).T.reset_index()
    df_ref_results['Structure name'] = sname
    df_calc_results['Structure name'] = sname

    ref_num = df_ref_results[df_ref_results.columns[1:-2]]
    calc_num = df_calc_results[df_calc_results.columns[1:-2]]

    delta = ((calc_num - ref_num) / ref_num) * 100

    res = OrderedDict()
    lim = 3
    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 2), round(delta[col].max(), 2)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T
    print(test_table)

    # plot_curves = True
    if plot_curves:
        for c in curves:
            c.plot_results()

    plt.show()


def test1(lim=3, save_data=False):
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
    Total Volume (cc)     0  [-0.7, 0.5]
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

    structure_files = ['/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/Cylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cylinders/RtCylinder_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/Cone_02_0.dcm',
                       '/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Cones/RtCone_02_0.dcm']

    structure_name = ['Sphere_02_0', 'Cylinder_02_0', 'RtCylinder_02_0', 'Cone__02_0', 'RtCone_02_0']

    dose_files = [
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_1mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_2mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_1mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_2mm_Aligned.dcm',
        r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm']

    # Structure Dict

    structure_dict = dict(zip(structure_name, structure_files))

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'0.4x0.2x0.4': dose_files[0], '1': dose_files[1], '2': dose_files[2], '3': dose_files[3]},
        'Y(SI)': {'0.4x0.2x0.4': dose_files[4], '1': dose_files[5], '2': dose_files[6], '3': dose_files[7]}}

    # grab analytical data
    sheet = 'Analytical'
    df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_data.xlsx',
                       sheetname=sheet)
    mask = df['CT slice spacing (mm)'] == '0.2mm'
    df = df.loc[mask]

    # Constrains to get data
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

    # GET CALCULATED DATA
    # backend = 'threading'
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(calc_data)(row,
                           dose_files_dict,
                           structure_dict,
                           constrains,
                           calculation_options) for row in df.iterrows())

    # aggregating data
    df_concat = [d[0] for d in res]
    sname = [d[1] for d in res]

    result = pd.concat(df_concat, axis=1).T.reset_index()
    result['Structure name'] = sname

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']
    num_col = ['Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = df[num_col]
    result_num = result[result.columns[1:-2]]
    result_num.columns = df_num.columns
    delta = ((result_num - df_num) / df_num) * 100

    # print table

    res = OrderedDict()
    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 1), round(delta[col].max(), 1)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T
    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core/validation_paper'
    if save_data:
        result.to_excel(os.path.join(dest, 'Test_1_result.xls'))
        test_table.to_excel(os.path.join(dest, 'test_1_table_paper.xls'))

    print(test_table)
    return test_table


def test2(lim=3):
    """
                              count         range
        Total Volume (cc)     2   [-3.9, 0.6]
        Dmin                  0   [-0.2, 2.6]
        Dmax                  0   [-0.4, 0.0]
        Dmean                 0   [-0.8, 0.7]
        D99                   8  [-14.4, 5.2]
        D95                   2   [-4.2, 3.2]
        D5                    0   [-0.7, 0.9]
        D1                    0   [-1.1, 2.7]
        D0.03cc              11   [0.2, 10.0]


    """
    ref_data = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_data.xlsx'
    struc_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/STRUCTURES'
    dose_grid_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS'

    snames = ['Sphere_10_0', 'Sphere_20_0', 'Sphere_30_0',
              'Cylinder_10_0', 'Cylinder_20_0', 'Cylinder_30_0',
              'RtCylinder_10_0', 'RtCylinder_20_0', 'RtCylinder_30_0',
              'Cone_10_0', 'Cone_20_0', 'Cone_30_0',
              'RtCone_10_0', 'RtCone_20_0', 'RtCone_30_0']

    structure_path = [os.path.join(struc_dir, f + '.dcm') for f in snames]

    structure_dict = dict(zip(snames, structure_path))

    dose_files = [os.path.join(dose_grid_dir, f) for f in [
        'Linear_AntPost_1mm_Aligned.dcm',
        'Linear_AntPost_2mm_Aligned.dcm',
        'Linear_AntPost_3mm_Aligned.dcm',
        'Linear_SupInf_1mm_Aligned.dcm',
        'Linear_SupInf_2mm_Aligned.dcm',
        'Linear_SupInf_3mm_Aligned.dcm']]

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'1': dose_files[0], '2': dose_files[1], '3': dose_files[2]},
        'Y(SI)': {'1': dose_files[3], '2': dose_files[4], '3': dose_files[5]}}

    # grab analytical data

    df = pd.read_excel(ref_data, sheetname='Analytical')

    dfi = df.ix[40:]
    mask0 = dfi['Structure Shift'] == 0
    dfi = dfi.loc[mask0]

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

    # GET CALCULATED DATA
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(calc_data)(row,
                           dose_files_dict,
                           structure_dict,
                           constrains,
                           calculation_options) for row in dfi.iterrows())

    # aggregating data
    df_concat = [d[0] for d in res]
    sname = [d[1] for d in res]

    result = pd.concat(df_concat, axis=1).T.reset_index()
    result['Structure name'] = sname

    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core/validation_paper'
    result.to_excel(os.path.join(dest, 'Test_2_result.xls'))

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    num_col = ['Total Volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = dfi[num_col]

    result_num = result[result.columns[1:-2]]
    result_num.columns = df_num.columns
    result_num.index = df_num.index

    delta = ((result_num - df_num) / df_num) * 100

    pcol = ['Total Volume (cc)', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1']

    res = OrderedDict()

    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 1), round(delta[col].max(), 1)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T

    test_table.to_excel(os.path.join(dest, 'test_2_table_paper.xls'))

    print(test_table)


def test3(plot_curves=True):
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

    ref_data = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_data.xlsx'

    struc_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/STRUCTURES'
    dose_grid_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS'
    st = 2

    snames = ['Sphere_10_0', 'Sphere_30_0',
              'Cylinder_10_0', 'Cylinder_30_0',
              'RtCylinder_10_0', 'RtCylinder_30_0',
              'Cone_10_0', 'Cone_30_0',
              'RtCone_10_0', 'RtCone_30_0']

    structure_path = [os.path.join(struc_dir, f + '.dcm') for f in snames]

    structure_dict = dict(zip(snames, structure_path))

    dose_files = [os.path.join(dose_grid_dir, f) for f in [
        'Linear_AntPost_1mm_Aligned.dcm',
        'Linear_AntPost_2mm_Aligned.dcm',
        'Linear_AntPost_3mm_Aligned.dcm',
        'Linear_SupInf_1mm_Aligned.dcm',
        'Linear_SupInf_2mm_Aligned.dcm',
        'Linear_SupInf_3mm_Aligned.dcm']]

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'1': dose_files[0], '2': dose_files[1], '3': dose_files[2]},
        'Y(SI)': {'1': dose_files[3], '2': dose_files[4], '3': dose_files[5]}}

    test_files = {}
    for s_name in structure_dict:
        grad_files = {}
        for grad in dose_files_dict:
            tick = str(int(int(re.findall(r'\d+', s_name)[0]) / 10))
            grad_files[grad] = dose_files_dict[grad][tick]

        test_files[s_name] = grad_files

    result = OrderedDict()
    for sname in snames:
        struc_path = structure_dict[sname]
        # set structure's object
        struc = ScoringDicomParser(filename=struc_path)
        structures = struc.GetStructures()
        structure = structures[st]

        # set end cap by 1/2 slice thickness
        calculation_options['end_cap'] = structure['thickness'] / 2.0

        # set up sampled structure
        struc_teste = Structure(structure, calculation_options)
        str_result = {}
        test_data = test_files[sname]
        for k in test_data:
            # get dose
            dose_file = test_data[k]
            dicom_dose = ScoringDicomParser(filename=dose_file)
            dhist, chist = struc_teste.calculate_dvh(dicom_dose)
            dvh_data = struc_teste.get_dvh_data()
            str_result[k] = dvh_data

        result[sname] = str_result

    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/test3_ref_dvh.obj'
    # save(an_data, dest)
    an_data = load(dest)

    teste = []
    curve_compare = []
    for s in result:
        for g in result[s]:
            adata = an_data[s][g]
            calc_data = result[s][g]
            cmp = CurveCompare(adata['dose_axis'], adata['data'], calc_data['dose_axis'], calc_data['data'])
            curve_stats = cmp.stats_paper
            curve_stats['Resolution (mm)'] = str(int(int(re.findall(r'\d+', s)[0]) / 10))
            curve_stats['Gradient'] = g
            curve_compare.append(cmp)
            tmp = pd.DataFrame(curve_stats, index=[s])
            teste.append(tmp)

    df_final = pd.concat(teste)

    mask0 = np.logical_and(df_final['Resolution (mm)'] == '1', df_final['Gradient'] == 'Y(SI)')
    mask1 = np.logical_and(df_final['Resolution (mm)'] == '1', df_final['Gradient'] == 'Z(AP)')
    mask2 = np.logical_and(df_final['Resolution (mm)'] == '3', df_final['Gradient'] == 'Y(SI)')
    mask3 = np.logical_and(df_final['Resolution (mm)'] == '3', df_final['Gradient'] == 'Z(AP)')

    # Row 0
    r0 = pd.DataFrame(['Y(SI)'], index=['Average (N = 5)'], columns=['Gradient'])
    r0['Resolution (mm)'] = '1'
    ri = pd.DataFrame(df_final[mask0].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r0 = r0.join(ri)

    # Row 1
    r1 = pd.DataFrame(['Z(AP)'], index=['Average (N = 5)'], columns=['Gradient'])
    r1['Resolution (mm)'] = '1'
    ri = pd.DataFrame(df_final[mask1].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r1 = r1.join(ri)

    # Row 2
    r2 = pd.DataFrame(['Y(SI)'], index=['Average (N = 5)'], columns=['Gradient'])
    r2['Resolution (mm)'] = '3'
    ri = pd.DataFrame(df_final[mask2].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r2 = r2.join(ri)

    # Row 3
    r3 = pd.DataFrame(['Z(AP)'], index=['Average (N = 5)'], columns=['Gradient'])
    r3['Resolution (mm)'] = '3'
    ri = pd.DataFrame(df_final[mask3].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r3 = r3.join(ri)
    result_df = pd.concat([df_final, r0, r1, r2, r3])

    print(result_df)
    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core/validation_paper'
    result_df.to_excel(os.path.join(dest, 'test_3_table.xls'))
    #
    # result_df.to_excel('test_3_table.xls')

    if plot_curves:
        # for c in curve_compare:
        #     c.plot_results()
        for grad in ['Z(AP)', 'Y(SI)']:
            for s_key in result:
                adata = an_data[s_key][grad]
                calc_data = result[s_key][grad]
                fig, ax = plt.subplots()
                ax.plot(adata['dose_axis'], adata['data'], label='Analytical')
                ax.plot(calc_data['dose_axis'], calc_data['data'], label='Software')
                ax.legend(loc='best')
                ax.set_xlabel('Dose (cGy)')
                ax.set_ylabel('Volume (cc)')
                title = s_key + ' Dose Gradient ' + grad + '.png'
                ax.set_title(title)
                fig.savefig(os.path.join(dest, title), format='png', dpi=100)
        plt.show()

    return curve_compare, result_df


def test_eval_competition_data():
    # TODO EVAL FILE ERRORS
    root_path = r'I:\Plan_competition_data\Final Reports'
    rs_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'

    obj = EvalCompetition(root_path, rs_file, constrains, scores)
    obj.set_data()
    res = obj.calc_scores()
    data = obj.comp_data
    sc = [i for i in res if isinstance(i, tuple)]

    data_name = data.set_index(0)
    data_name = data_name.groupby(data_name.index).first()
    df = pd.DataFrame(sc).set_index(0)
    plan_iq = data_name.ix[df.index]['plan_iq_scores']

    comp = pd.concat([plan_iq, df], axis=1)
    comp['delta'] = comp[1] - comp['plan_iq_scores']
    comp = comp.rename(columns={1: 'py_score'})
    comp.to_excel('Plan_IQ_versus_Python_BODY_DMAX.xls')


def get_competition_data(root_path):
    files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    report_files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
                    name.endswith(('.pdf', '.PDF'))]

    filtered_files = OrderedDict()
    for f in files:
        try:
            obj = ScoringDicomParser(filename=f)
            rt_type = obj.GetSOPClassUID()
            if rt_type == 'rtdose':
                tmp = f.split(os.path.sep)[-2].split()
                name = tmp[0].split('-')[0]
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
            if rt_type == 'rtplan':
                tmp = f.split(os.path.sep)[-2].split()
                name = tmp[0].split('-')[0]
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
        except:
            logger.exception('Error in file %s' % f)

    data = pd.DataFrame(filtered_files).T

    plan_iq_scores = []
    for f in report_files:
        p, r = os.path.split(f)
        s = re.findall('\d+\.\d+', r)
        plan_iq_scores.append(s * 2)

    plan_iq_scores = np.ravel(plan_iq_scores).astype(float)
    data['plan_iq_scores'] = plan_iq_scores

    return data.reset_index()


def batch_call_dvh(root_path, rs_file, clean_files=False):
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    data = get_competition_data(root_path)

    if clean_files:
        dvh_files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
                     name.endswith('.dvh')]
        for dv in dvh_files:
            os.remove(dv)

    mask = data[1] == 'rtdose'
    rd_files = data['index'][mask].values
    names = data[0][mask].values

    rtss = ScoringDicomParser(filename=rs_file)
    structures = rtss.GetStructures()

    i = 0
    for f, n in zip(rd_files, names):
        p = os.path.splitext(f)
        out_file = p[0] + '.dvh'
        dest, df = os.path.split(f)
        if not os.path.exists(out_file):
            print('Iteration: %i' % i)
            print('processing file: %s' % f)
            calcdvhs = calc_dvhs(n, rs_file, f, out_file=out_file)
            i += 1
            print('processing file done %s' % f)

            fig, ax = plt.subplots()
            fig.set_figheight(12)
            fig.set_figwidth(20)

            for key, structure in structures.items():
                sname = structure['name']
                ax.plot(calcdvhs[sname]['data'] / calcdvhs[sname]['data'][0] * 100,
                        label=sname, linewidth=2.0, color=np.array(structure['color'], dtype=float) / 255)
                ax.legend(loc=7, borderaxespad=-5)
                ax.set_ylabel('Vol (%)')
                ax.set_xlabel('Dose (cGy)')
                ax.set_title(n + ':' + df)
                fig_name = os.path.join(dest, n + '_RD_calc_DVH.png')
                fig.savefig(fig_name, format='png', dpi=100)

            plt.close('all')


class EvalCompetition(object):
    def __init__(self, root_path, rs_file, constrains, scores):
        self.root_path = root_path
        self.rs_file = rs_file
        self.constrains = constrains
        self.scores = scores
        self.comp_data = None
        self.dvh_files = []
        self.results = []
        self.comp_data = get_competition_data(root_path)

    def save_reports(self):
        pass

    def save_dvh_all(self, clean_files=False, end_cap=False, dicom_dvh=False):

        # TODO implement saving TPS information, constrain and scoring report on dvh file encapsulated on Participant Class
        data = get_competition_data(self.root_path)
        self.comp_data = data

        if clean_files:
            dvh_files = [os.path.join(root, name) for root, dirs, files in os.walk(self.root_path) for name in files if
                         name.endswith('.dvh')]
            for dv in dvh_files:
                os.remove(dv)

        mask = data[1] == 'rtdose'
        rd_files = data['index'][mask].values
        names = data[0][mask].values

        rtss = ScoringDicomParser(filename=self.rs_file)
        structures = rtss.GetStructures()

        i = 0
        for f, n in zip(rd_files, names):
            p = os.path.splitext(f)
            out_file = p[0] + '.dvh'
            dest, df = os.path.split(f)
            if not os.path.exists(out_file):
                print('Iteration: %i' % i)
                print('processing file: %s' % f)
                if dicom_dvh:
                    try:
                        calcdvhs = save_dicom_dvhs(n, self.rs_file, f, out_file=out_file)
                    except:
                        rt_dose = ScoringDicomParser(filename=f)
                        k = rt_dose.get_tps_data()
                        txt = 'No DVH in file %s \n TPS info:' % f
                        txt += ', '.join("{!s}={!r}".format(key, val) for (key, val) in k.items())
                        logger.debug(txt)
                else:
                    calcdvhs = calc_dvhs_upsampled(n, self.rs_file, f, self.scores.keys(), out_file=out_file,
                                                   end_cap=end_cap)
                i += 1
                print('processing file done %s' % f)

    def set_data(self):
        self.comp_data = get_competition_data(self.root_path)

        self.dvh_files = [os.path.join(root, name) for root, dirs, files in os.walk(self.root_path) for name in files if
                          name.endswith('.dvh')]

    def calc_scores(self):
        res = Parallel(n_jobs=-1, verbose=11)(
            delayed(self.get_score)(dvh_file) for dvh_file in self.dvh_files)
        self.results = res
        return res

    def get_score(self, dvh_file):
        rd_file, rp_file, name = self.get_dicom_data(self.comp_data, dvh_file)
        try:
            obj = Scoring(rd_file, self.rs_file, rp_file, self.constrains, self.scores)
            obj.set_dvh_data(dvh_file)
            print('Score:', name, obj.get_total_score())
            return name, obj.get_total_score()
        except:
            logger.exception('Error in file: %s' % dvh_file)
            try:
                obj = Scoring(rd_file, self.rs_file, rp_file, self.constrains, self.scores)
                obj.set_dicom_dvh_data()
                print('Score:', name, obj.get_total_score())
                return name, obj.get_total_score()
            except:
                logger.exception('No DVH data in file  %s' % rd_file)
                return rd_file

    @staticmethod
    def get_dicom_data(data, dvh_file):
        if __name__ == '__main__':
            try:
                dvh = load(dvh_file)
                name = dvh['participant']
                p_files = data[data[0] == name].set_index(1)
                rd_file = p_files.ix['rtdose']['index']
                rp_file = p_files.ix['rtplan']['index']
                return rd_file, rp_file, name
            except:
                logger.exception('error on file %s' % dvh_file)

                # TODO wrap DICOM-RT data to eval scores


def read_planiq_dvh(f):
    """
        Reads plan IQ exported txt DVH data
    :param f: path to file txt
    :return: Pandas Dataframe with DVH data in cGy and vol cc
    """
    with open(f, 'r') as io:
        txt = io.readlines()

    struc_header = [t.strip() for t in txt[0].split('\t') if len(t.strip()) > 0]
    data = np.asarray([t.split('\t') for t in txt[2:]], dtype=float)
    dose_axis = data[:, 0] * 100  # cGy
    idx = np.arange(1, data.shape[1], 2)
    vol = data[:, idx]

    plan_iq = pd.DataFrame(vol, columns=struc_header, index=dose_axis)

    return plan_iq


def test_eval_scores():
    cdata = Competition2016()
    root = r'D:\PLAN_TESTING_DATA'
    rs = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    obj = EvalCompetition(root_path=root, rs_file=rs, constrains=cdata.constrains, scores=cdata.scores)
    # obj.save_dvh_all(clean_files=True, end_cap=True)
    obj.set_data()
    res = obj.calc_scores()
    df_new = pd.DataFrame(res, columns=['name', 'py_score_new']).set_index('name')

    # val = r'/home/victor/Dropbox/Plan_Competition_Project/validation/Python_2016_last_update.xls'
    # df = pd.DataFrame(sc, columns=['name', 'py_score_new'])
    # df.set_index('name').sort_index().to_excel(val)
    # dfi = df.set_index(0).sort_index()

    ref = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\validation\Plan_IQ_versus_Python_DONE_diff_greater_than_4_TPS.xls'
    ndata = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\validation\Python_2016_last_update.xls'
    df_ref = pd.read_excel(ref)
    df_new = pd.read_excel(ndata).set_index('name')
    df_comp = df_ref.join(df_new)
    df_comp['delta'] = df_comp['py_score_new'] - df_comp['PLANIQ_DMAX_BODY_SCORE']
    df_comp['delta_py'] = df_comp['py_score_new'] - df_comp['py_score']

    mask = df_comp['delta'].abs() > 4

    result_col = ['PLANIQ_DMAX_BODY_SCORE', 'py_score_new', 'delta', 'TPS']
    print(df_comp[result_col].loc[mask])


def test_upsampling_paper(plot_grads=False):
    rd = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RD.1.2.246.352.71.7.584747638204.1758320.20170210154830.dcm'
    rs = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RS.1.2.246.352.71.4.584747638204.248648.20170209152429.dcm'
    rd1 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Saad RapidArc Eclipse/Saad RapidArc Eclipse/RD.Saad-Eclipse-RapidArc.dcm'
    rs1 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Saad RapidArc Eclipse/Saad RapidArc Eclipse/RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'

    dicom_data = [(rd, rs), (rd1, rs1)]

    gradient_stats = []
    for data in dicom_data:
        rdi = data[0]
        rsi = data[1]

        rtss = ScoringDicomParser(filename=rsi)
        dicom_dose = ScoringDicomParser(filename=rdi)
        structures = rtss.GetStructures()

        for key, structure in structures.items():
            struc = Structure(structure)
            df_rectangles = struc.boundary_rectangles(dicom_dose)
            med = df_rectangles['Boundary gradient'].median()
            # get stats
            sts = [structure['name'], struc.volume_original, med]
            gradient_stats.append(sts)

            if plot_grads:
                rec = ['internal', 'bounding', 'external']
                fig, ax = plt.subplots()
                df_rectangles[rec].plot(ax=ax)
                ax.set_ylabel('Dmax - Dmin (cGy)')
                ax.set_xlabel('Z - slice position (mm)')
                ax.set_title(structure['name'])
                fig, ax = plt.subplots()
                df_rectangles['Boundary gradient'].plot(ax=ax)
                ax.set_ylabel('Boundary gradient (cGy)')
                ax.set_xlabel('Z - slice position (mm)')

                title = 'Boundary gradient ' + structure['name'] + \
                        ' Median: %1.2f (cGy)' % med
                ax.set_title(title)
                plt.show()

    gradient_df = pd.DataFrame(gradient_stats, columns=['Name', 'Volume', 'Median'])
    fig, ax = plt.subplots()
    ax.plot(gradient_df['Volume'], np.log10(gradient_df['Median']), '.')
    ax.set_xlabel('Volume (cc)')
    ax.set_ylabel('Median boundary gradient (cGy)')
    ax.set_xlim([0, 1000])
    plt.show()


def calc_gradients_pp(structure_dict, dose):
    structure_obj = Structure(structure_dict)
    df_rect = structure_obj.boundary_rectangles(dose, up_sample=True)
    # get stats
    return [structure_dict['name'], structure_obj.volume_original, df_rect['Boundary gradient'].median()]


def upsampling_paper_pp():
    rd = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RD.1.2.246.352.71.7.584747638204.1758320.20170210154830.dcm'
    rs = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RS.1.2.246.352.71.4.584747638204.248648.20170209152429.dcm'
    rd1 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Saad RapidArc Eclipse/Saad RapidArc Eclipse/RD.Saad-Eclipse-RapidArc.dcm'
    rs1 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Saad RapidArc Eclipse/Saad RapidArc Eclipse/RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    rd2 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Venessa IMRT Eclipse/RD-Eclipse-Venessa-IMRTDose.dcm'
    dicom_data = [(rd, rs), (rd1, rs1), (rd2, rs1)]

    gradient_stats = []
    for data in dicom_data:
        rdi = data[0]
        rsi = data[1]
        rtss = ScoringDicomParser(filename=rsi)
        dicom_dose = ScoringDicomParser(filename=rdi)
        structures = rtss.GetStructures()
        res = Parallel(n_jobs=4, verbose=11)(
            delayed(calc_gradients_pp)(structure, dicom_dose) for key, structure in structures.items())

        gradient_stats += res

    gradient_df = pd.DataFrame(gradient_stats, columns=['Name', 'Volume', 'Median'])
    fig, ax = plt.subplots()
    ax.plot(gradient_df['Volume'], gradient_df['Median'], '.')
    ax.set_xlabel('Volume (cc)')
    ax.set_ylabel('Median boundary gradient (cGy)')
    ax.set_xlim([0, 2000])
    plt.show()


def timimg_evaluation():
    rd = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RD.1.2.246.352.71.7.584747638204.1758320.20170210154830.dcm'
    rs = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RS.1.2.246.352.71.4.584747638204.248648.20170209152429.dcm'
    rd1 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Saad RapidArc Eclipse/Saad RapidArc Eclipse/RD.Saad-Eclipse-RapidArc.dcm'
    rs1 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Saad RapidArc Eclipse/Saad RapidArc Eclipse/RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    rd2 = '/home/victor/Dropbox/Plan_Competition_Project/Competition_2016/Eclipse Plans/Venessa IMRT Eclipse/RD-Eclipse-Venessa-IMRTDose.dcm'
    # dicom_data = [(rd, rs), (rd1, rs1), (rd2, rs1)]

    dicom_data = [(rd, rs), (rd1, rs1)]

    deltas = [1, 0.5, 0.25, 0.1]
    timing_stats = []
    for d in deltas:
        for data in dicom_data:
            rdi = data[0]
            rsi = data[1]

            rtss = ScoringDicomParser(filename=rsi)
            dicom_dose = ScoringDicomParser(filename=rdi)
            structures = rtss.GetStructures()

            for key, structure in structures.items():
                struc = Structure(structure)
                if struc.volume_original < 100:
                    struc.set_delta((d, d, d))
                    res = struc.calculate_dvh(dicom_dose, timing=True)
                    timing_stats.append(res)

    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/validation/timings.obj'
    save(timing_stats, dest)
    # timing_stats = load(dest)

    df = pd.DataFrame(timing_stats, columns=['Structure', 'Volume', 'voxels', 'grid_mm', 'timing'])
    mask = df['grid_mm'] == 0.1

    plt.plot(df[['Volume']][mask], df[['timing']][mask], '.')
    plt.show()


def compare_dvh_planIQ():
    plt.style.use('ggplot')
    from pyplanscoring.core.scoring import get_participant_folder_data, Participant

    folder = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/ref_plan'
    plan_iq = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/ref_plan/PlanIQ TXT DVH Feb 5 2017.txt'

    calculation_options = dict()
    calculation_options['end_cap'] = 1.5
    calculation_options['use_tps_dvh'] = False
    calculation_options['use_tps_structures'] = False
    calculation_options['up_sampling'] = True
    calculation_options['maximum_upsampled_volume_cc'] = 100.0
    calculation_options['voxel_size'] = 0.5
    calculation_options['num_cores'] = 8
    calculation_options['save_dvh_figure'] = True
    calculation_options['save_dvh_data'] = True
    calculation_options['mp_backend'] = 'multiprocessing'

    df = read_planiq_dvh(plan_iq)
    flag, files_data = get_participant_folder_data('ref_plan', folder)
    rd = files_data.reset_index().set_index(1).ix['rtdose']['index']
    rp = files_data.reset_index().set_index(1).ix['rtplan']['index']
    rs = files_data.reset_index().set_index(1).ix['rtss']['index']
    participant = Participant(rp, rs, rd, calculation_options=calculation_options)
    participant.set_participant_data('ref_plan')
    participant.set_structure_names(df.columns)
    cdvh = participant.calculate_dvh(df.columns)

    for key in cdvh.keys():
        fig, ax = plt.subplots()
        fig.set_figheight(12)
        fig.set_figwidth(20)

        ax.plot(cdvh[key]['dose_axis'], cdvh[key]['data'] / cdvh[key]['data'][0] * 100, label='PyPlanScoring')
        ax.plot(df.index, df[key] / df[key][0] * 100, label='PlanIQ')
        ax.set_title(key)
        ax.set_ylabel('Volume (%)')
        ax.set_xlabel('Dose (cGy)')
        ax.set_xlim([cdvh[key]['dose_axis'][0], cdvh[key]['dose_axis'][-1]])
        ax.legend()

    plt.show()


def get_plans_data(root_path):
    files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    filtered_files = OrderedDict()
    for f in files:
        try:
            obj = ScoringDicomParser(filename=f)
            rt_type = obj.GetSOPClassUID()
            if rt_type == 'rtdose':
                # tmp = f.split(os.path.sep)[-2].split()
                p, name = os.path.split(f)
                # name = tmp[0].split('-')[0]
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
            if rt_type == 'rtplan':
                # tmp = f.split(os.path.sep)[-2].split()
                # name = tmp[0].split('-')[0]
                p, name = os.path.split(f)
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
        except:
            logger.exception('Error in file %s' % f)

    data = pd.DataFrame(filtered_files).T
    return data


def batch_calc_2017():
    config = configparser.ConfigParser()
    conf_file = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/PyPlanScoring.ini'

    config.read(conf_file)
    calculation_options = dict()
    calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
    calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
    calculation_options['use_tps_structures'] = config.getboolean('DEFAULT', 'use_tps_structures')
    calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
    calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
    calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
    calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
    calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
    calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
    calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']

    root_path = '/media/victor/TOURO Mobile/COMPETITION 2017/plans/Patch_calculation/Patch Calculation Test'
    rs_file = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/DICOM/RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm'
    rp_file = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad Nobah/RP.1.2.246.352.71.5.584747638204.955801.20170210152428.dcm'
    comp_data = get_plans_data(root_path)

    criteria_file = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/Scoring Criteria.txt'
    constrains, scores, criteria = read_scoring_criteria(criteria_file)

    banner_path = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/2017 Plan Comp Banner.jpg'

    mask = comp_data[1] == 'rtdose'
    rd_files = comp_data.index[mask]
    names = comp_data[0][mask].values

    i = 0
    for f, n in zip(rd_files, names):
        p = os.path.splitext(f)
        out_file = p[0] + '.dvh'
        dest, df = os.path.split(f)
        if not os.path.exists(out_file):
            try:
                print('Iteration: %i' % i)
                print('processing file: %s' % f)
                participant = Participant(rp_file, rs_file, f, calculation_options=calculation_options)
                participant.set_participant_data(n)
                val = participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria)
                report_path = p[0] + '_plan_scoring_report.xlsx'
                participant.save_score(report_path, banner_path=banner_path, report_header=n)
            except:
                logger.exception('Error in file: %s' % f)


if __name__ == '__main__':

    config = configparser.ConfigParser()
    conf_file = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/PyPlanScoring.ini'

    config.read(conf_file)
    calculation_options = dict()
    calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
    calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
    calculation_options['use_tps_structures'] = config.getboolean('DEFAULT', 'use_tps_structures')
    calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
    calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
    calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
    calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
    calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
    calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
    calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']

    root_path = '/media/victor/TOURO Mobile/COMPETITION 2017/plans/Patch_calculation/Patch Calculation Test'
    rs_file = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/DICOM/RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm'
    rp_file = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad Nobah/RP.1.2.246.352.71.5.584747638204.955801.20170210152428.dcm'
    comp_data = get_plans_data(root_path)

    criteria_file = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/Scoring Criteria.txt'
    constrains, scores, criteria = read_scoring_criteria(criteria_file)

    banner_path = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/2017 Plan Comp Banner.jpg'

    mask = comp_data[1] == 'rtdose'
    rd_files = comp_data.index[mask]
    names = comp_data[0][mask].values

    i = 0
    for f, n in zip(rd_files, names):
        # f = rd_files[-3]
        p = os.path.splitext(f)
        out_file = p[0] + '.dvh'

        dest, df = os.path.split(f)
        # if not os.path.exists(out_file):
        try:
            print('Iteration: %i' % i)
            print('processing file: %s' % f)
            participant = Participant(rp_file, rs_file, f, calculation_options=calculation_options)
            participant.set_participant_data(n)
            val = participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria)
            report_path = p[0] + '_plan_scoring_report.xlsx'
            participant.save_score(report_path, banner_path=banner_path, report_header=n)
        except:
            logger.exception('Error in file: %s' % f)

        i += 1
