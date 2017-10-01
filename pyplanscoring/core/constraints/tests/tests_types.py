# from .types import DoseValue, DoseUnit
import os

import numpy.testing as npt

from pyplanscoring.core.constraints.types import DoseValue, DoseUnit, DVHData


def test_dose_value():
    # test sum equal quantity
    """
        Test DoseValue Class and its objects
    """
    a = DoseValue(10, DoseUnit.cGy)
    b = DoseValue(15, DoseUnit.cGy)
    c = a + b
    npt.assert_approx_equal(c.value, 25.0)

    # test sum diff quantity - result always cGy
    a = DoseValue(1, DoseUnit.Gy)
    b = DoseValue(15, DoseUnit.cGy)
    c = a + b
    npt.assert_approx_equal(c.value, 115)

    # test sub equal quantity
    a = DoseValue(100, DoseUnit.cGy)
    b = DoseValue(10, DoseUnit.cGy)
    c = a - b
    npt.assert_approx_equal(c.value, 90.0)

    # test sub diff quantity - result always cGy
    a = DoseValue(1, DoseUnit.Gy)
    b = DoseValue(10, DoseUnit.cGy)
    c = a - b
    npt.assert_approx_equal(c.value, 90)

    # tests mul by integer - results at the same unit
    a = DoseValue(200, DoseUnit.cGy)
    b = 35
    c = a * b
    npt.assert_approx_equal(c.value, 7000)

    a = DoseValue(2, DoseUnit.Gy)
    b = 35
    c = a * b
    npt.assert_approx_equal(c.value, 70)

    # tests mult by dose result always in Gy
    a = DoseValue(2, DoseUnit.Gy)
    b = DoseValue(10, DoseUnit.cGy)
    c = a * b
    npt.assert_approx_equal(c, 0.2)

    # equal units
    a = DoseValue(2, DoseUnit.Gy)
    b = DoseValue(10, DoseUnit.Gy)
    c = a * b
    npt.assert_approx_equal(c, 20.0)

    a = DoseValue(200, DoseUnit.cGy)
    b = DoseValue(1000, DoseUnit.cGy)
    c = a * b
    npt.assert_approx_equal(c, 20.0)

    # test division by integer
    a = DoseValue(7000, DoseUnit.cGy)
    b = 35
    c = a / b
    npt.assert_approx_equal(c.value, 200.0)
    a = DoseValue(70, DoseUnit.Gy)
    b = 35
    c = a / b
    npt.assert_approx_equal(c.value, 2.0)

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


def test_DVHDataExtensions():
    DATA_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'test_data',

    )
    filename = os.path.join(DATA_DIR, 'RD.dcm')

    from pyplanscoring.core.dicomparser import ScoringDicomParser
    import quantities as pq
    # GETTING dvh DATA FROM DOSE

    filename = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data\RD.dcm'
    rd_dcm = ScoringDicomParser(filename=filename)
    dvh_all = rd_dcm.GetDVHs()
    dvh = dvh_all[61]
    dvh_metrics = DVHData(dvh)

    # test get dose at volume
    res = dvh_metrics.get_dose_at_volume(100 * pq.percent)
    assert res == dvh_metrics.min_dose
    res1 = dvh_metrics.get_dose_at_volume(dvh_metrics.volume)
    assert res1 == dvh_metrics.min_dose

    # test get volume at dose
    dose0 = DoseValue(7000, DoseUnit.cGy)
    res = dvh_metrics.get_volume_at_dose(dose0, )

    # check dose in Gy query
    dose0 = DoseValue(70, DoseUnit.Gy)
    res1 = dvh_metrics.get_volume_at_dose(dose0, )
    assert res == res1

    # check dose > max_dose
    dose0 = DoseValue(700, DoseUnit.Gy)
    res1 = dvh_metrics.get_volume_at_dose(dose0, )
    assert res1 == 0

    # check dose < min_dose
    dose0 = DoseValue(1, DoseUnit.Gy)
    res1 = dvh_metrics.get_volume_at_dose(dose0, )
    assert res1 == dvh_metrics.volume
