import os
import pytest
import pandas as pd

from core.calculation import PyStructure
from core.dicom_reader import PyDicomParser
from core.types import Dose3D, DoseUnit

benchmark_data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'benchmark_data',
)

struc_dir = os.path.join(benchmark_data_dir, os.sep + 'DVH-Analysis-Data-Etc' + os.sep + 'STRUCTURES')
dose_grid_dir = os.path.join(benchmark_data_dir, os.sep + 'DVH-Analysis-Data-Etc' + os.sep + 'DOSE GRIDS')


def get_dose_3d(rd):
    dose_values = PyDicomParser(filename=rd).get_dose_matrix()
    grid = PyDicomParser(filename=rd).get_grid_3d()

    return Dose3D(dose_values, grid, DoseUnit.Gy)


@pytest.fixture()
def analytical_data():
    sheet = 'Analytical'
    path_xlsx = os.path.join(benchmark_data_dir, 'analytical_data.xlsx')
    df = pd.read_excel(path_xlsx, sheetname=sheet)

    return df


@pytest.fixture()
def test1_data():
    structure_name = ['Sphere_02_0', 'Cylinder_02_0', 'RtCylinder_02_0', 'Cone__02_0', 'RtCone_02_0']

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

    structure_dict = dict(zip(structure_name, structure_files))

    dose_files_dict = {
        'Z(AP)': {'0.4x0.2x0.4': dose_files[0], '1': dose_files[1], '2': dose_files[2], '3': dose_files[3]},
        'Y(SI)': {'0.4x0.2x0.4': dose_files[4], '1': dose_files[5], '2': dose_files[6], '3': dose_files[7]}}

    return structure_dict, dose_files_dict


@pytest.fixture()
def test1_calc_data():
    structure_name = ['Sphere_02_0', 'Cylinder_02_0', 'RtCylinder_02_0', 'Cone__02_0', 'RtCone_02_0']

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

    structures_dicom = [PyDicomParser(filename=f) for f in structure_files]
    structures_py = [PyStructure(s) for s in structures_dicom]
    grids = [(0.1, 0.1, 0.1)] * len(structures_py)
    doses_3d = [get_dose_3d(d) for d in dose_files]
    # calculate end cap






def constraints_data():
    constraints_query = ['Dmin[Gy]',
                         'Dmax[Gy]',
                         'Dmean[Gy]',
                         'D99%[Gy]',
                         'D95%[Gy]',
                         'D5%[Gy]',
                         'D1%[Gy]',
                         'D0.03cc[Gy]']
    return constraints_query
