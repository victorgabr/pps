import os

import pandas as pd
import pytest

from core.calculation import PyStructure
from core.dicom_reader import PyDicomParser

benchmark_data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'benchmark_data',
)

struc_dir = os.path.join(benchmark_data_dir, 'DVH-Analysis-Data-Etc' + os.sep + 'STRUCTURES')
dose_grid_dir = os.path.join(benchmark_data_dir, 'DVH-Analysis-Data-Etc' + os.sep + 'DOSE GRIDS')


@pytest.fixture()
def results_folder():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')


# calculate end cap

@pytest.fixture(scope='session')
def analytical_data():
    sheet = 'Analytical'
    path_xlsx = os.path.join(benchmark_data_dir, 'analytical_data.xlsx')
    df = pd.read_excel(path_xlsx, sheet_name=sheet)

    return df


def test1_data():
    structure_name = ['Sphere_02_0.dcm', 'Cylinder_02_0.dcm', 'RtCylinder_02_0.dcm', 'Cone_02_0.dcm', 'RtCone_02_0.dcm']

    dose_files_name = ['Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm',
                       'Linear_AntPost_1mm_Aligned.dcm',
                       'Linear_AntPost_2mm_Aligned.dcm',
                       'Linear_AntPost_3mm_Aligned.dcm',
                       'Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm',
                       'Linear_SupInf_1mm_Aligned.dcm',
                       'Linear_SupInf_2mm_Aligned.dcm',
                       'Linear_SupInf_3mm_Aligned.dcm']

    structure_files = [os.path.join(struc_dir, n) for n in structure_name]
    dose_files = [os.path.join(dose_grid_dir, n) for n in dose_files_name]

    structures_dicom = [PyDicomParser(filename=f).GetStructures()[2] for f in structure_files]
    structures_py = [PyStructure(s, end_cap=s['thickness'] / 2.) for s in structures_dicom]

    doses_3d = [PyDicomParser(filename=d).get_dose_3d() for d in dose_files]

    structure_dict = dict(zip(structure_name, structures_py))

    dose_files_dict = {
        'Z(AP)': {'0.4x0.2x0.4': doses_3d[0], '1': doses_3d[1], '2': doses_3d[2], '3': doses_3d[3]},
        'Y(SI)': {'0.4x0.2x0.4': doses_3d[4], '1': doses_3d[5], '2': doses_3d[6], '3': doses_3d[7]}}

    return structure_dict, dose_files_dict


@pytest.fixture(scope='session')
def test1_calc_data():
    structure_name = ['Sphere_02_0.dcm', 'Cylinder_02_0.dcm', 'RtCylinder_02_0.dcm', 'Cone_02_0.dcm', 'RtCone_02_0.dcm']

    dose_files_name = ['Linear_AntPost_0-4_0-2_0-4_mm_Aligned.dcm',
                       'Linear_AntPost_1mm_Aligned.dcm',
                       'Linear_AntPost_2mm_Aligned.dcm',
                       'Linear_AntPost_3mm_Aligned.dcm',
                       'Linear_SupInf_0-4_0-2_0-4_mm_Aligned.dcm',
                       'Linear_SupInf_1mm_Aligned.dcm',
                       'Linear_SupInf_2mm_Aligned.dcm',
                       'Linear_SupInf_3mm_Aligned.dcm']

    structure_files = [os.path.join(struc_dir, n) for n in structure_name]
    dose_files = [os.path.join(dose_grid_dir, n) for n in dose_files_name]

    structures_dicom = [PyDicomParser(filename=f).GetStructures()[2] for f in structure_files]
    structures_py = [PyStructure(s, end_cap=s['thickness'] / 2.) for s in structures_dicom]

    doses_3d = [PyDicomParser(filename=d).get_dose_3d() for d in dose_files]

    structure_dict = dict(zip(structure_name, structures_py))

    dose_files_dict = {
        'Z(AP)': {'0.4x0.2x0.4': doses_3d[0], '1': doses_3d[1], '2': doses_3d[2], '3': doses_3d[3]},
        'Y(SI)': {'0.4x0.2x0.4': doses_3d[4], '1': doses_3d[5], '2': doses_3d[6], '3': doses_3d[7]}}

    sheet = 'Analytical'
    path_xlsx = os.path.join(benchmark_data_dir, 'analytical_data.xlsx')
    df = pd.read_excel(path_xlsx, sheetname=sheet)

    mask = df['CT slice spacing (mm)'] == '0.2mm'
    dfi = df.loc[mask]

    return structure_dict, dose_files_dict, dfi


@pytest.fixture(scope='session')
def constraints_data():
    constraints_query = ['Min[cGy]',
                         'Max[cGy]',
                         'Mean[cGy]',
                         'D99%[cGy]',
                         'D95%[cGy]',
                         'D5%[cGy]',
                         'D1%[cGy]',
                         'D0.03cc[cGy]']
    return constraints_query


@pytest.fixture(scope='session')
def test2_calc_data():
    # structure_name = ['Sphere_02_0.dcm', 'Cylinder_02_0.dcm', 'RtCylinder_02_0.dcm', 'Cone_02_0.dcm', 'RtCone_02_0.dcm']

    structure_name = ['Sphere_10_0', 'Sphere_20_0', 'Sphere_30_0',
                      'Cylinder_10_0', 'Cylinder_20_0', 'Cylinder_30_0',
                      'RtCylinder_10_0', 'RtCylinder_20_0', 'RtCylinder_30_0',
                      'Cone_10_0', 'Cone_20_0', 'Cone_30_0',
                      'RtCone_10_0', 'RtCone_20_0', 'RtCone_30_0']

    structure_name = [s + '.dcm' for s in structure_name]

    dose_files_name = ['Linear_AntPost_1mm_Aligned.dcm',
                       'Linear_AntPost_2mm_Aligned.dcm',
                       'Linear_AntPost_3mm_Aligned.dcm',
                       'Linear_SupInf_1mm_Aligned.dcm',
                       'Linear_SupInf_2mm_Aligned.dcm',
                       'Linear_SupInf_3mm_Aligned.dcm']

    structure_files = [os.path.join(struc_dir, n) for n in structure_name]
    dose_files = [os.path.join(dose_grid_dir, n) for n in dose_files_name]

    structures_dicom = [PyDicomParser(filename=f).GetStructures()[2] for f in structure_files]
    structures_py = [PyStructure(s, end_cap=s['thickness'] / 2.) for s in structures_dicom]

    doses_3d = [PyDicomParser(filename=d).get_dose_3d() for d in dose_files]

    structure_dict = dict(zip(structure_name, structures_py))

    dose_files_dict = {
        'Z(AP)': {'1': doses_3d[0], '2': doses_3d[1], '3': doses_3d[2]},
        'Y(SI)': {'1': doses_3d[3], '2': doses_3d[4], '3': doses_3d[5]}}

    sheet = 'Analytical'
    path_xlsx = os.path.join(benchmark_data_dir, 'analytical_data.xlsx')
    df = pd.read_excel(path_xlsx, sheetname=sheet)

    dfi = df.ix[40:]
    mask0 = dfi['Structure Shift'] == 0
    dfi = dfi.loc[mask0]

    return structure_dict, dose_files_dict, dfi


@pytest.fixture(scope='session')
def test3_calc_data():
    # structure_name = ['Sphere_02_0.dcm', 'Cylinder_02_0.dcm', 'RtCylinder_02_0.dcm', 'Cone_02_0.dcm', 'RtCone_02_0.dcm']

    structure_name = ['Sphere_10_0', 'Sphere_30_0',
                      'Cylinder_10_0', 'Cylinder_30_0',
                      'RtCylinder_10_0', 'RtCylinder_30_0',
                      'Cone_10_0', 'Cone_30_0',
                      'RtCone_10_0', 'RtCone_30_0']

    structure_name = [s + '.dcm' for s in structure_name]

    dose_files_name = ['Linear_AntPost_1mm_Aligned.dcm',
                       'Linear_AntPost_3mm_Aligned.dcm',
                       'Linear_SupInf_1mm_Aligned.dcm',
                       'Linear_SupInf_3mm_Aligned.dcm']

    structure_files = [os.path.join(struc_dir, n) for n in structure_name]
    dose_files = [os.path.join(dose_grid_dir, n) for n in dose_files_name]

    structures_dicom = [PyDicomParser(filename=f).GetStructures()[2] for f in structure_files]
    structures_py = [PyStructure(s, end_cap=s['thickness'] / 2.) for s in structures_dicom]

    doses_3d = [PyDicomParser(filename=d).get_dose_3d() for d in dose_files]

    structure_dict = dict(zip(structure_name, structures_py))

    dose_files_dict = {
        'Z(AP)': {'1': doses_3d[0], '3': doses_3d[1]},
        'Y(SI)': {'1': doses_3d[2], '3': doses_3d[3]}}

    return structure_dict, dose_files_dict


@pytest.fixture(scope='session')
def analytical_curves():
    path_xlsx = os.path.join(benchmark_data_dir, 'analytical_data.xlsx')
    sheet_names = ['Sphere', 'Cylinder', 'RtCylinder', 'Cone', 'RtCone']

    return {name: pd.read_excel(path_xlsx, sheetname=name) for name in sheet_names}


@pytest.fixture(scope='session')
def test_dicompyler_data():
    # structure_name = ['Sphere_02_0.dcm', 'Cylinder_02_0.dcm', 'RtCylinder_02_0.dcm', 'Cone_02_0.dcm', 'RtCone_02_0.dcm']

    structure_name = ['Sphere_10_0', 'Sphere_30_0',
                      'Cylinder_10_0', 'Cylinder_30_0',
                      'RtCylinder_10_0', 'RtCylinder_30_0',
                      'Cone_10_0', 'Cone_30_0',
                      'RtCone_10_0', 'RtCone_30_0']

    structure_name = [s + '.dcm' for s in structure_name]

    dose_files_name = ['Linear_AntPost_1mm_Aligned.dcm',
                       'Linear_AntPost_3mm_Aligned.dcm',
                       'Linear_SupInf_1mm_Aligned.dcm',
                       'Linear_SupInf_3mm_Aligned.dcm']

    structure_files = [os.path.join(struc_dir, n) for n in structure_name]
    dose_files = [os.path.join(dose_grid_dir, n) for n in dose_files_name]

    structures_dicom = [PyDicomParser(filename=f).GetStructures()[2] for f in structure_files]
    structures_py = [PyStructure(s, end_cap=s['thickness'] / 2.) for s in structures_dicom]

    doses_3d = [PyDicomParser(filename=d).get_dose_3d() for d in dose_files]

    structure_dict = dict(zip(structure_name, structures_py))
    structure_files_dict = dict(zip(structure_name, structure_files))

    doses_3d_dict = {
        'Z(AP)': {'1': doses_3d[0], '3': doses_3d[1]},
        'Y(SI)': {'1': doses_3d[2], '3': doses_3d[3]}}

    doses_files_dict = {
        'Z(AP)': {'1': dose_files[0], '3': dose_files[1]},
        'Y(SI)': {'1': dose_files[2], '3': dose_files[3]}}

    return structure_dict, doses_3d_dict, structure_files_dict, doses_files_dict
