from unittest import TestCase

from core.types import DoseValue, DoseUnit, DoseValuePresentation


class TestDoseValue(TestCase):
    def test_value(self):
        a = DoseValue(10, DoseUnit.cGy)
        b = DoseValue(15, DoseUnit.cGy)
        c = a + b
        self.assertAlmostEqual(c, DoseValue(25, DoseUnit.cGy))

        # test different units
        a = DoseValue(1, DoseUnit.Gy)
        b = DoseValue(15, DoseUnit.cGy)
        c = a + b
        self.assertAlmostEqual(c, DoseValue(115, DoseUnit.cGy))

        # test sub equal quantity
        a = DoseValue(100, DoseUnit.cGy)
        b = DoseValue(10, DoseUnit.cGy)
        c = a - b
        self.assertAlmostEqual(c, DoseValue(90, DoseUnit.cGy))

        # test sub diff quantity - result always cGy
        a = DoseValue(1, DoseUnit.Gy)
        b = DoseValue(10, DoseUnit.cGy)
        c = a - b
        self.assertAlmostEqual(c, DoseValue(90, DoseUnit.cGy))

        # tests mul by integer - results at the same unit
        a = DoseValue(200, DoseUnit.cGy)
        b = 35
        c = a * b
        self.assertAlmostEqual(c, DoseValue(7000, DoseUnit.cGy))
        self.assertAlmostEqual(c, DoseValue(70, DoseUnit.Gy))

        # tests mult by dose result always in Gy
        a = DoseValue(2, DoseUnit.Gy)
        b = DoseValue(10, DoseUnit.cGy)
        c = a * b
        self.assertAlmostEqual(c, 0.2)

        # equal units
        a = DoseValue(2, DoseUnit.Gy)
        b = DoseValue(10, DoseUnit.Gy)
        c = a * b
        self.assertAlmostEqual(c, 20)

        a = DoseValue(200, DoseUnit.cGy)
        b = DoseValue(1000, DoseUnit.cGy)
        c = a * b
        self.assertAlmostEqual(c, 200000.0)

        # test division by integer
        a = DoseValue(7000, DoseUnit.cGy)
        b = 35
        c = a / b
        self.assertAlmostEqual(c, DoseValue(2, DoseUnit.Gy))
        self.assertAlmostEqual(c, DoseValue(200, DoseUnit.cGy))

        # test division by Dose
        a = DoseValue(7000, DoseUnit.cGy)
        b = DoseValue(200, DoseUnit.cGy)
        c = a / b
        self.assertAlmostEqual(c, DoseValue(35.0, DoseUnit.Unknown))

    def test_operators(self):
        # test operators
        # test major
        a = DoseValue(200.0, DoseUnit.cGy)
        b = DoseValue(1, DoseUnit.cGy)

        assert a > b

        # test minor
        a = DoseValue(200.0, DoseUnit.cGy)
        b = DoseValue(1, DoseUnit.cGy)

        assert b < a

        # test equality
        a = DoseValue(200.0, DoseUnit.cGy)
        b = DoseValue(200.0, DoseUnit.cGy)
        assert a == b

    def test_get_presentation(self):
        a = DoseValue(200.0, DoseUnit.cGy)
        b = DoseValue(200.0, DoseUnit.Gy)
        c = DoseValue(200.0, DoseUnit.Percent)
        d = DoseValue(200.0, DoseUnit.Unknown)
        assert a.get_presentation() == DoseValuePresentation.Absolute
        assert b.get_presentation() == DoseValuePresentation.Absolute
        assert c.get_presentation() == DoseValuePresentation.Relative
        assert d.get_presentation() == DoseValuePresentation.Unknown
