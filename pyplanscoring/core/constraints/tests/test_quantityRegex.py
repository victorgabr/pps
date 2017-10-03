from unittest import TestCase

from pyplanscoring.core.constraints.types import QuantityRegex, VolumePresentation, DoseUnit

strings = ["cc",
           "CC",
           "cm3",
           "CM3",
           "cM3",
           "cGy",
           "cGY",
           "CGY",
           "cgy",
           "gy",
           "Gy",
           "GY",
           "%",
           "NA",
           ""]

qtd_target = [VolumePresentation.absolute_cm3,
              VolumePresentation.absolute_cm3,
              VolumePresentation.absolute_cm3,
              VolumePresentation.absolute_cm3,
              VolumePresentation.absolute_cm3,
              DoseUnit.cGy,
              DoseUnit.cGy,
              DoseUnit.cGy,
              DoseUnit.cGy,
              DoseUnit.Gy,
              DoseUnit.Gy,
              DoseUnit.Gy,
              DoseUnit.Percent,
              DoseUnit.Unknown,
              DoseUnit.Unknown]


class TestQuantityRegex(TestCase):
    def test_string_to_quantity(self):
        qtd = [QuantityRegex.string_to_quantity(s) for s in strings]
        for i in range(len(qtd)):
            self.assertEqual(qtd[i], qtd_target[i])
