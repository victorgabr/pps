"""

Classes to encapsulate NTCP calculation

"""
import numpy as np

from scipy import integrate

from . import njit
from ..core.dvhdoses import get_ddvh


class NTCPLKBModel:
    """
        This class represents a NTCP(Normal Tissue Complication Probability) LKB model
        (Lyman 1985, Kutcher and Burman 1989)

    """
    def __init__(self, cdvh, parameter_vector):
        self._cdvh = {}
        self._dose_array = None
        self._volume_array = None
        self._m = None
        self._n = None
        self._td50 = None
        self._parameter_vector = None
        self._parameter_map = {}
        # setters
        self.cdvh = cdvh
        self.parameter_vector = parameter_vector
        self.td50 = self.parameter_vector[0]
        self.m = self.parameter_vector[1]
        self.n = self.parameter_vector[2]
        self.parameter_map = self.get_parameter_map()

    @property
    def dose_array(self):
        return self._dose_array

    @property
    def volume_array(self):
        return self._volume_array

    @property
    def cdvh(self):
        return self._cdvh

    @cdvh.setter
    def cdvh(self, value):
        if not isinstance(value, dict):
            raise TypeError("DVH value should be dict")

        if "type" not in value:
            raise ValueError(
                "DVH type should have type key - (CUMULATIVE or DIFFERENTIAL)")

        if value["type"] == "CUMULATIVE":
            # convert it to differential
            ddvh = get_ddvh(value["data"], value["scaling"])
            self._cdvh = value.copy()
            self._cdvh["data"] = ddvh
            self._cdvh["type"] = "DIFFERENTIAL"
            # set private variables
            self._dose_array = np.arange(value['bins']) * value['scaling']
            self._volume_array = ddvh
            if value['doseunits'] == 'cGY':
                # dirty fix to cGy data
                self._dose_array /= 100

        if value["type"] == "DIFFERENTIAL":
            self._cdvh = value
            self._dose_array = np.arange(value['bins']) * value['scaling']
            self._volume_array = value["data"]

    @property
    def parameter_vector(self):
        return self._parameter_vector

    @parameter_vector.setter
    def parameter_vector(self, value):
        if len(value) != 3:
            raise ValueError(
                "parameter_vector invalid: size must be 3! [TD50, m, n]")
        self._parameter_vector = value

    @property
    def m(self):
        """
            m is a measure of the slope of the sigmoid curve
        :rtype: float
        """
        return self._m

    @m.setter
    def m(self, value):
        if not isinstance(value, float):
            raise TypeError("the slope of the sigmoid curve should be float")

        self._m = value

    @property
    def n(self):
        """
            n is the volume effect parameter
        """
        return self._n

    @n.setter
    def n(self, value):
        if not isinstance(value, float):
            raise TypeError("volume effect parameter should be float")

        self._n = value

    @property
    def td50(self):
        """
            TD 50 is the uniform dose given to the entire organ
            that results in 50% complication risk
        :return:
        :rtype float
        """

        return self._td50

    @td50.setter
    def td50(self, value):
        if not isinstance(value, float):
            raise TypeError("TD 50 parameter should be float")
        self._td50 = value

    @property
    def dvh(self):
        return self._dvh

    @dvh.setter
    def dvh(self, value):
        # should be differential DVH
        self._dvh = value

    @property
    def parameter_map(self):
        return self._parameter_map

    @parameter_map.setter
    def parameter_map(self, value):
        if not isinstance(value, dict):
            raise TypeError(
                "Parameter map should be dictionary type wih keys: TD50, m, n")

        if isinstance(value, dict):
            for k in ["TD50", "m", "n"]:
                if k not in value:
                    raise ValueError("Missing key: {}".format(k))

            self._parameter_map = value

    def get_parameter_map(self):
        pMap = {"TD50": self.td50, "m": self.m, "n": self.n}
        return pMap

    def calc_model(self):
        # TODO debug EUD
        deff = calc_eud(self.dose_array, self.volume_array, self.n)
        ntcp = self.lkb(deff, self.td50, self.m)
        return ntcp

    def lkb(self, dose_eff, td_50, m):
        """
            Calculates NTCP via LKB movel

        :param dose_eff: gEUD
        :param td_50: TD 50 is the uniform dose given to the entire organ
        :param m:  m is a measure of the slope of the sigmoid curve
        :return:
        """
        t = (dose_eff - td_50) / (m * td_50)
        return self.integrate_lkb(t)

    @staticmethod
    def integrate_lkb(t):
        norm_factor = np.sqrt(2 * np.pi)
        # f = lambda x: np.exp(-x ** 2 / 2)
        # res = integrate.quad(f, -np.inf, t)
        res = integrate.quad(lambda t: np.exp(-t**2 / 2), -np.inf, t)

        return res[0] / norm_factor

    def __str__(self):
        return "LKB model - TD50: {:.2f} m: {:.2f} n: {:.2f}".format(
            self.td50, self.m, self.n)

    def __repr__(self):
        return self.__str__()


@njit
def calc_eud(dose, volume, n):
    """
        GENERALIZED EQUIVALENT UNIFORM DOSE
    :param dose: Dose axis
    :param volume: dDVH in cc
    :param n: n is the volume effect parameter
    :return: gEUD
    """
    eud = 0.
    # n_voxels = len(dose)
    a = 1. / n
    delta_dose = dose[1] - dose[0]
    for i in range(len(dose)):
        di = (i + 0.5) * delta_dose
        eud += np.power(di, a) * volume[i]

    eud = np.power(eud, 1. / a) / np.sum(volume)

    return eud
