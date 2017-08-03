import logging

import numba as nb
import numpy as np

from pyplanscoring.complexity.ApertureMetric import Aperture, EdgeMetricBase
from pyplanscoring.complexity.EsapiApertureMetric import ComplexityMetric, MetersetsFromMetersetWeightsCreator
from pyplanscoring.complexity.dicomrt import RTPlan

logger = logging.getLogger('PyApertureMetric.py')

logging.basicConfig(filename='complexity_reports.log', level=logging.DEBUG)


class PyAperture(Aperture):
    def __init__(self, leaf_positions, leaf_widths, jaw, gantry_angle):
        super().__init__(leaf_positions, leaf_widths, jaw)
        self.gantry_angle = gantry_angle

    @property
    def GantryAngle(self):
        return self.gantry_angle

    @GantryAngle.setter
    def GantryAngle(self, value):
        self.gantry_angle = value

    def __repr__(self):
        txt = "Aperture - Gantry: %1.1f" % self.GantryAngle
        return txt


class PyAperturesFromBeamCreator:
    def Create(self, beam):

        apertures = []
        leafWidths = self.GetLeafWidths(beam)
        jaw = self.CreateJaw(beam)
        for controlPoint in beam['ControlPointSequence']:
            gantry_angle = float(controlPoint.GantryAngle) if 'GantryAngle' in controlPoint else beam['GantryAngle']
            leafPositions = self.GetLeafPositions(controlPoint)
            apertures.append(PyAperture(leafPositions, leafWidths, jaw, gantry_angle))
        return apertures

    @staticmethod
    def CreateJaw(beam):
        """
            but the Aperture class expects cartesian y axis
        :param beam:
        :return:
        """

        # if there is no X jaws, consider open 400 mm
        left = float(beam['ASYMX'][0]) if 'ASYMX' in beam else -200.0
        right = float(beam['ASYMX'][1]) if 'ASYMX' in beam else 200.0
        top = float(beam['ASYMY'][0]) if 'ASYMY' in beam else -200.0
        bottom = float(beam['ASYMY'][1]) if 'ASYMY' in beam else 200.0

        # invert y axis to match apperture class -top, -botton
        return [left, -top, right, -bottom]

    def GetLeafWidths(self, beam_dict):
        """
            Get MLCX leaf width from  BeamLimitingDeviceSequence
            (300a, 00be) Leaf Position Boundaries Tag
        :param beam_dict: Dicomparser Beam dict from plan_dict
        :return: MLCX leaf width
        """

        bs = beam_dict['BeamLimitingDeviceSequence']
        # the script only takes MLCX as parameter
        for b in bs:
            if b.RTBeamLimitingDeviceType == 'MLCX':
                return np.diff(b.LeafPositionBoundaries)

    def GetLeafTops(self, beam_dict):
        """
            Get MLCX leaf Tops from  BeamLimitingDeviceSequence
            (300a, 00be) Leaf Position Boundaries Tag
        :param beam_dict: Dicomparser Beam dict from plan_dict
        :return: MLCX leaf width
        """
        bs = beam_dict['BeamLimitingDeviceSequence']
        for b in bs:
            if b.RTBeamLimitingDeviceType == 'MLCX':
                return np.array(b.LeafPositionBoundaries[:-1], dtype=float)

    def GetLeafPositions(self, control_point):
        """
            Leaf positions are given from bottom to top by ESAPI,
            but the Aperture class expects them from top to bottom
            Leaf Positions are mechanical boundaries projected onto Isocenter plane
        :param control_point:
        """
        pos = control_point.BeamLimitingDevicePositionSequence[-1]
        mlc_open = pos.LeafJawPositions
        n_pairs = int(len(mlc_open) / 2)
        bank_a_pos = mlc_open[:n_pairs]
        bank_b_pos = mlc_open[n_pairs:]

        return np.vstack((bank_a_pos, bank_b_pos))


class PyEdgeMetricBase(EdgeMetricBase):
    def Calculate(self, aperture):
        return self.DivisionOrDefault(aperture.side_perimeter(), aperture.Area())

    @staticmethod
    def DivisionOrDefault(a, b):
        return a / b if b != 0 else 0


class PyComplexityMetric(ComplexityMetric):
    # TODO add unit tests
    def CalculateForPlan(self, patient=None, plan=None):
        """
            Returns the complexity metric of a plan, calculated as
            the weighted sum of the individual metrics for each beam
        :param patient: Patient Class
        :param plan: Plan class
        :return: metric
        """
        weights = self.GetWeightsPlan(plan)
        metrics = self.GetMetricsPlan(patient, plan)

        return self.WeightedSum(weights, metrics)

    def GetWeightsPlan(self, plan):
        """
             Returns the weights of a plan's beams
             by default, the weights are the meterset values per beam
        :param plan: DicomParser plan dict
        """
        return self.GetMeterSetsPlan(plan)

    def GetMeterSetsPlan(self, plan):
        """
            Returns the total metersets of a plan's beams
        :param plan: DicomParser plan dictionaty
        :return: metersets of a plan's beams
        """
        return [float(beam['MU']) for k, beam in plan['beams'].items() if 'MU' in beam]

    def CalculateForPlanPerBeam(self, patient, plan):
        """
            Returns the unweighted metrics of a plan's non-setup beams
        :param patient:
        :param plan:
        :return:
        """
        values = []
        for k, beam in plan['beams'].items():
            # check if treatment beam
            if 'MU' in beam:
                v = self.CalculateForBeam(patient, plan, beam)
                values.append(v)

        return values

    def CalculatePerAperture(self, apertures):
        metric = PyEdgeMetricBase()
        return [metric.Calculate(aperture) for aperture in apertures]

    def CalculateForBeamPerAperture(self, patient, plan, beam):
        apertures = self.CreateApertures(patient, plan, beam)
        return self.CalculatePerAperture(apertures)

    def CreateApertures(self, patient, plan, beam):
        """
            Added default parameter to meet Liskov substitution principle
        :param patient:
        :param plan:
        :param beam:
        :return:
        """
        return PyAperturesFromBeamCreator().Create(beam)


class PyMetersetsFromMetersetWeightsCreator:
    def Create(self, beam):
        if beam['PrimaryDosimeterUnit'] != 'MU':
            return None

        metersetWeights = self.GetMetersetWeights(beam['ControlPointSequence'])
        metersets = self.ConvertMetersetWeightsToMetersets(beam['MU'], metersetWeights)

        return self.UndoCummulativeSum(metersets)

    @staticmethod
    def GetMetersetWeights(ControlPoints):
        return np.array([cp.CumulativeMetersetWeight for cp in ControlPoints], dtype=float)

    @staticmethod
    def ConvertMetersetWeightsToMetersets(beamMeterset, metersetWeights):
        return beamMeterset * metersetWeights / metersetWeights[-1]

    @staticmethod
    def UndoCummulativeSum(cummulativeSum):
        """
            Returns the values whose cummulative sum is "cummulativeSum"
        :param cummulativeSum:
        :return:
        """
        return undo_cumulative_sum(cummulativeSum)


@nb.njit
def undo_cumulative_sum(cummulativeSum):
    values = np.zeros(len(cummulativeSum))
    delta_prev = 0.0
    for i in range(len(values) - 1):
        delta_curr = cummulativeSum[i + 1] - cummulativeSum[i]
        values[i] = 0.5 * delta_prev + 0.5 * delta_curr
        delta_prev = delta_curr

    values[-1] = 0.5 * delta_prev

    return values


def test_metersets_cp_creator_numba():
    plan_file = r"D:\Final_Plans\ECPLIPSE_VMAT\Friedemann Herberth  - FANTASY - 21 APRIL FINAL - 100.0\RP.1.2.246.352.71.5.29569967170.312423.20170420161749.dcm"
    plan_info = RTPlan(filename=plan_file)
    plan_dict = plan_info.get_plan()
    # complexity_metric = PyComplexityMetric().CalculateForPlan(None, plan_dict)
    # # PyComplexityMetric().CalculateForBeam(None, plan_dict, )
    # print('Friedemann Herberth  - FANTASY - 21 APRIL FINAL - 100')
    # print('complexity Metric: ', complexity_metric)
    # print('complexity threshold of 0.18')

    # test beam creator
    beam = plan_dict['beams'][1]
    a = MetersetsFromMetersetWeightsCreator().Create(beam)
    mt = MetersetsFromMetersetWeightsCreator()
    metersetWeights = mt.GetMetersetWeights(beam['ControlPointSequence'])
    metersets = mt.ConvertMetersetWeightsToMetersets(beam['MU'], metersetWeights)
    cs = mt.UndoCummulativeSum(metersets)

    mt.Create(beam)

    pymt = PyMetersetsFromMetersetWeightsCreator()
    metersetWeights1 = pymt.GetMetersetWeights(beam['ControlPointSequence'])
    metersets1 = pymt.ConvertMetersetWeightsToMetersets(beam['MU'], metersetWeights1)
    cs1 = pymt.UndoCummulativeSum(metersets1)

    np.testing.assert_array_almost_equal(metersetWeights, metersetWeights1)
    np.testing.assert_array_almost_equal(metersets, metersets1)
    np.testing.assert_array_almost_equal(mt.Create(beam), pymt.Create(beam))


def batch_calc_complexity():
    import logging
    # from pyplanscoring.complexity.PyApertureMetric import get_score_complexity

    logger = logging.getLogger('test.py')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('ggplot')

    if __name__ == '__main__':
        participant_folder = r'D:\Final_Plans\ECPLIPSE_VMAT'
        res = get_score_complexity(participant_folder)

        score_complexity = np.array(list(filter(lambda x: x is not None, res)))

        plt.figure()
        plt.plot(score_complexity[:, 0], score_complexity[:, 1], '.')
        plt.xlabel('Score')
        plt.ylabel('complexity factor mm-1')
        plt.axhline(0.18, color='b')
        plt.title('Eclipse™ - VMAT')

        plt.figure()
        plt.plot(score_complexity[:, 0], score_complexity[:, 2], '.')
        plt.xlabel('Score')
        plt.ylabel('MU/cGy')
        plt.title('Eclipse™ - VMAT')

        plt.figure()
        plt.plot(score_complexity[:, 2], score_complexity[:, 1], '.')
        plt.xlabel('MU/cGy')
        plt.ylabel('complexity factor mm-1')
        plt.title('Eclipse™ - VMAT')
        plt.show()


if __name__ == '__main__':
    participant_folder = r'D:\Dropbox\Plan_Competition_Project\competition_2017\plans\final_reports\plans_folder'
    # data = get_dicom_data(participant_folder)
    # files_data = data[1]
    # rp = files_data.reset_index().set_index(1).loc['rtplan']['index']

    # for plan_file in rp:
    plan_file = r'D:\Dropbox\Plan_Competition_Project\competition_2017\plans\final_reports\plans_folder\Abdul Qadir Jangda - Eclipse - IMRT - 23 MARCH FINAL - 50.4\RP.1.2.246.352.71.5.225097321.535249.20170322121553.dcm'
    plan_info = RTPlan(filename=plan_file)
    plan_dict = plan_info.get_plan()
    pm = PyComplexityMetric()
    complexity_metric = pm.CalculateForPlan(None, plan_dict)
    print(complexity_metric)

    ap = PyAperturesFromBeamCreator().Create(plan_dict['beams'][2])
