"""
    Module to encapsulate the public interface

"""
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .core.calculation import DVHCalculation, PyStructure
from .core.dicom_reader import PyDicomParser


def plot_dvh(dvh, title=''):
    """
        Plots an absolute DVH
    :param dvh:
    :param title:
    """
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
    plt.plot(x, dvh['data'], label=dvh['name'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_dvhs(dvhs, title):
    """
        Plots relative volume DVHs
    :param dvhs:
    :param title:
    """
    x_label = 'Dose [Gy]'
    y_label = 'Volume [%]'
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for k, dvh in dvhs.items():
        x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
        y = dvh['data'] / dvh['data'][0] * 100
        ax.plot(x, y, label=dvh['name'])

    plt.legend()
    plt.show()


class PyPlanScoringAPI:
    """
        Class to Calculate a DVH from DICOM RT data
    """

    def __init__(self, rs_file_path: str, rd_file_path: str) -> None:
        self._structures = None
        self._dose_3d = None
        self._dvhs = {}
        self.rs_dcm = PyDicomParser(filename=rs_file_path)
        self.rd_dcm = PyDicomParser(filename=rd_file_path)

        # setters
        self.structures = self.rs_dcm
        self.dose_3d = self.rd_dcm

    @property
    def structures(self):
        return self._structures

    @structures.setter
    def structures(self, value):
        self._structures = value.GetStructures()

    @property
    def dose_3d(self):
        return self._dose_3d

    @dose_3d.setter
    def dose_3d(self, value):
        self._dose_3d = value.get_dose_3d()

    @property
    def dvhs(self):
        return self._dvhs

    def get_structure_dvh(self,
                          roi_number: int,
                          end_cap: float = None,
                          calc_grid: Tuple = None,
                          verbose=False) -> Dict:
        """
            Helper method to calculate a structure DVH from DICOM dataset
        :param roi_number: strucure id - 1,2,3..N
        :param end_cap: end cap value in mm - e.g, half slice size.
        :param calc_grid: (dx,dy,dz) up-sampling grid delta in mm - Voxel size
        :return:
        """
        if roi_number in self.structures:
            # setup Structure object that encapsulates end-cap and oversampling
            py_struc = PyStructure(
                self.structures[roi_number], end_cap=end_cap)

            # Setup DVH calculator
            dvh_calc_obj = DVHCalculation(
                py_struc, self.dose_3d, calc_grid=calc_grid)
            structure_dvh = dvh_calc_obj.calculate(verbose=verbose)

            return structure_dvh

        else:
            raise ValueError(
                "Structure of DICOM-ID: %s not found on DICOM-RTSS dataset" %
                roi_number)

    def calc_dvhs(self,
                  end_cap: float = None,
                  calc_grid: Tuple = None,
                  verbose=False) -> Dict:
        """
            Calculates all DVHs
        """

        # TODO implement auto grid size selecting

        for roi_number, contour in self.structures.items():
            self._dvhs[roi_number] = self.get_structure_dvh(
                roi_number=roi_number,
                end_cap=end_cap,
                calc_grid=calc_grid,
                verbose=verbose)

        return self._dvhs
