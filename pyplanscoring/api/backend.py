"""
Module to implement Façade design pattern to control pyplanscoring's functionality.

"""

import abc

import os

from complexity.PyComplexityMetric import PyComplexityMetric
from constraints.metrics import PlanEvaluation, RTCase, PyPlanningItem
from core.calculation import DVHCalculator, get_calculation_options, timeit
from core.dicom_reader import PyDicomParser
from core.io import get_participant_folder_data, IOHandler, save_formatted_report
from core.types import Dose3D, DoseUnit


class BackEnd(abc.ABC):
    # todo later implement as Abstract base class

    def setup_case(self, rs_file_path, file_path, sheet_name):
        return NotImplementedError

    def parse_dicom_folder(self, plan_folder):
        return NotImplementedError

    def setup_dvh_calculation(self, ini_file):
        return NotImplementedError

    def setup_planing_item(self):
        return NotImplementedError

    def calculate_dvh(self):
        return NotImplementedError

    def calc_plan_score(self):
        return NotImplementedError

    def calc_plan_complexity(self):
        return NotImplementedError

    def save_dvh_data(self):
        return NotImplementedError

    def save_report_data(self):
        return NotImplementedError


class Observer(abc.ABC):

    def update(self, obj, *args, **kwargs):
        raise NotImplementedError


class Observable:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def notify_observer(self, *args, **kwargs):
        for observer in self._observers:
            observer.update(self, *args, **kwargs)


class DVHCalculatorObservable(DVHCalculator, Observable):
    _initialized = False

    def __init__(self, rt_case=None, calculation_options=None):
        super().__init__(rt_case, calculation_options)
        Observable.__init__(self)

    def __getattribute__(self, name):
        return Observable.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if not self._initialized:
            Observable.__setattr__(self, name, value)
        else:
            setattr(self, name, value)
            self.notify_observer(key=name, value=value)


class PyPlanScoringKernel(BackEnd):

    def __init__(self):
        self._dcm_files = None
        self._plan_eval = PlanEvaluation()
        self._case = None
        self._dvh_calculator = DVHCalculator()
        self._planning_item = None
        self._dvh_data = {}
        self._report_data_frame = None
        self._total_score = 0.
        self._complexity = PyComplexityMetric()
        self._plan_complexity = 0
        self._io = IOHandler()

    @property
    def total_score(self):
        return self._total_score

    @property
    def plan_complexity(self):
        return self._plan_complexity

    @property
    def report(self):
        return self._report_data_frame

    @property
    def dvh_data(self):
        return self._dvh_data

    @property
    def planning_item(self):
        return self._planning_item

    @property
    def dvh_calculator(self):
        return self._dvh_calculator

    @property
    def case(self):
        return self._case

    @property
    def dcm_files(self):
        return self._dcm_files

    def parse_dicom_folder(self, plan_folder):
        dcm_files, flag = get_participant_folder_data(plan_folder)
        self._dcm_files = dcm_files
        return dcm_files, flag

    def setup_case(self, rs_file_path, file_path, sheet_name):
        structures = PyDicomParser(filename=rs_file_path).GetStructures()
        self._plan_eval.read(file_path, sheet_name=sheet_name)
        # todo implement setup case by id
        self._case = RTCase(sheet_name, 1, structures, self._plan_eval.criteria)

    def setup_dvh_calculation(self, ini_file):
        if self.case is not None:
            setup_calculation_options = get_calculation_options(ini_file)
            self._dvh_calculator.rt_case = self.case
            self._dvh_calculator.calculation_options = setup_calculation_options

    def setup_planing_item(self):

        if self.dcm_files is not None and self.case is not None:
            if self.dvh_calculator is not None:
                plan_dcm = PyDicomParser(filename=self.dcm_files['rtplan'])
                dose_dcm = PyDicomParser(filename=self.dcm_files['rtdose'])
                plan_dict = plan_dcm.GetPlan()
                dose_values = dose_dcm.get_dose_matrix()
                grid = dose_dcm.get_grid_3d()
                dose_3d = Dose3D(dose_values, grid, DoseUnit.Gy)
                self._planning_item = PyPlanningItem(plan_dict, self.case, dose_3d, self.dvh_calculator)

    def calculate_dvh(self):
        if self.planning_item is not None:
            self.planning_item.calculate_dvh()
            self._dvh_data = self.planning_item.dvh_data

    def calc_plan_score(self):
        if self._dvh_data:
            df = self._plan_eval.eval_plan(self.planning_item)
            self._total_score = df['Raw score'].sum()
            self._report_data_frame = df

    def calc_plan_complexity(self):
        if self.planning_item is not None:
            complexity_metric = self._complexity.CalculateForPlan(None, self.planning_item.plan_dict)
            self._plan_complexity = complexity_metric

    def save_dvh_data(self):
        if self._dvh_data:
            diretory, filename = os.path.split(self.dcm_files['rtdose'])
            dvh_file = os.path.join(diretory, filename + '.dvh')
            self._io.dvh_data = self._dvh_data
            self._io.to_json_file(dvh_file)

    def save_report_data(self):
        if self._report_data_frame is not None:
            diretory, filename = os.path.split(self.dcm_files['rtdose'])
            report_data_file = os.path.join(diretory, filename + '.csv')
            self._report_data_frame.to_csv(report_data_file)
            save_formatted_report(self.report,os.path.join(diretory, filename + '.xls'))
