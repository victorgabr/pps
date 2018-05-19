import matplotlib.pyplot as plt
import os

from gui.api.tools import read_eclipse_dvh, plot_dvh
from pyplanscoring.core.calculation import PyStructure, DVHCalculation
from pyplanscoring.core.dicom_reader import PyDicomParser
from pyplanscoring.core.types import DoseAccumulation


def get_dicom_data(root_path):
    """
        Provide all participant required files (RP,RS an RD DICOM FILES)
    :param root_path: participant folder
    :return: Pandas DataFrame containing path to files
    """
    files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    filtered_files = []
    for f in files:
        obj = PyDicomParser(filename=f)
        rt_type = obj.GetSOPClassUID()
        # fix halcyon SOP class UI
        if rt_type is None:
            rt_type = obj.ds.Modality.lower()

        if rt_type in ['rtdose', 'rtplan', 'rtss']:
            filtered_files.append([rt_type, f])

    return filtered_files


if __name__ == '__main__':

    # slicer/RT DVH
    file_path = r'/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/left_chest_wall/photon_electron/Absolute DVH Photon and electron plan sum.txt'
    eclipse_dvh = read_eclipse_dvh(file_path)

    # path to dicom files
    folder = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/left_chest_wall/photon_electron'
    dicom_data = get_dicom_data(folder)
    # filter rtdose
    rd_files = [d[1] for d in dicom_data if d[0] == 'rtdose']

    dcm_objs = [PyDicomParser(filename=rd_file) for rd_file in rd_files]
    doses_3d = [obj.get_dose_3d() for obj in dcm_objs]

    # Sum DVHs
    acc = DoseAccumulation(doses_3d)
    plan_sum = acc.get_plan_sum()

    # TODO compare with slicerRT plan sum DVH
    rs_file = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/left_chest_wall/photon_electron/RS.1.2.246.352.71.4.584747638204.283643.20180405155645.dcm'

    structures = PyDicomParser(filename=rs_file).GetStructures()
    structures_py = [PyStructure(s, end_cap=s['thickness']) for k, s in structures.items()]

    dvh_pyplan = {}
    for s in structures_py:
        if s.name in eclipse_dvh.keys():
            if s.volume < 100:
                dvh_calci = DVHCalculation(s, plan_sum, calc_grid=(0.5, 0.5, 0.5))
            else:
                dvh_calci = DVHCalculation(s, plan_sum, calc_grid=None)
            dvh_l = dvh_calci.calculate(verbose=True)
            dvh_pyplan[dvh_l['name']] = dvh_l
            plot_dvh(dvh_l, dvh_l['name'])
            plt.plot(eclipse_dvh[s.name][:, 0], eclipse_dvh[s.name][:, 1], label='Eclipse')
            plt.legend()
            plt.show()
