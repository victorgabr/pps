from pyplanscoring.core.types import DoseValue, DoseUnit, VolumePresentation
import numpy as np
from unittest import TestCase


def test_get_volume_at_dose(dvh):
    # test get volume at dose
    dose0 = DoseValue(7000, DoseUnit.cGy)
    res = dvh.get_volume_at_dose(dose0, VolumePresentation.absolute_cm3)

    # check dose in Gy query
    dose0 = DoseValue(70, DoseUnit.Gy)
    res1 = dvh.get_volume_at_dose(dose0, VolumePresentation.absolute_cm3)
    assert res == res1

    # check dose > max_dose
    dose0 = DoseValue(700, DoseUnit.Gy)
    res1 = dvh.get_volume_at_dose(dose0, VolumePresentation.absolute_cm3)
    assert res1 == 0

    # check dose < min_dose
    dose0 = DoseValue(1, DoseUnit.Gy)
    res1 = dvh.get_volume_at_dose(dose0, VolumePresentation.absolute_cm3)
    assert res1 == dvh.volume


def test_get_compliment_volume_at_dose(dvh):
    # test get volume at dose
    dose_c = DoseValue(7000, DoseUnit.cGy)
    dvh.get_compliment_volume_at_dose(dose_c, VolumePresentation.absolute_cm3)

    # test get c volume at dose
    dose_c1 = DoseValue(70000, DoseUnit.cGy)
    dvh.get_compliment_volume_at_dose(dose_c1, VolumePresentation.absolute_cm3)


def test_get_dose_at_volume(dvh, query_extensions):
    query_str = 'D100%[cGy]'
    # read query into que object
    query_extensions.read(query_str)

    # execute the static method
    md = query_extensions.query_dose(dvh, query_extensions)
    assert md == dvh.min_dose

    # Do 100 from absolute values
    query_str = 'D689.501173445633cc[cGy]'
    query_extensions.read(query_str)

    # execute the static method
    md = query_extensions.query_dose(dvh, query_extensions)
    assert md == dvh.min_dose

    query_str = 'D95%[cGy]'
    # read query into que object
    query_extensions.read(query_str)
    # execute the static method
    md = query_extensions.query_dose(dvh, query_extensions)

    np.testing.assert_almost_equal(md.value, 6103.854532025905)

    # Do 95% from absolute values
    query_str = 'D655.0261147733513cc[cGy]'
    query_extensions.read(query_str)

    # execute the static method
    md = query_extensions.query_dose(dvh, query_extensions)
    np.testing.assert_almost_equal(md.value, 6103.854532025905)
    #


def test_get_dose_compliment(query_extensions, dvh):
    query_str = 'DC95%[cGy]'
    # read query into que object
    query_extensions.read(query_str)
    # execute the static method
    md = query_extensions.query_dose_compliment(dvh, query_extensions)
    np.testing.assert_almost_equal(md.value, 7401.78624315853)

    # Do 95% from absolute values
    query_str = 'DC655.0261147733513cc[cGy]'
    query_extensions.read(query_str)
    # execute the static method
    md = query_extensions.query_dose_compliment(dvh, query_extensions)
    np.testing.assert_almost_equal(md.value, 7401.78624315853)


def test_to_relative(dvh, query_extensions):
    test_case = TestCase()

    prescribed_dose = 7000.0
    dvh.to_relative_dose(DoseValue(7000, DoseUnit.cGy))
    assert dvh.dose_unit == DoseUnit.Percent

    # Test min dose %
    min_dose_pp = dvh.min_dose
    min_dose_target = DoseValue(3457.0138887638896 / prescribed_dose * 100, DoseUnit.Percent)
    test_case.assertAlmostEqual(min_dose_pp, min_dose_target)

    # Test mean dose %
    mean_dose_pp = dvh.mean_dose
    mean_dose = 6949.34536891202
    mean_dose_target = DoseValue(mean_dose / prescribed_dose * 100, DoseUnit.Percent)
    test_case.assertAlmostEqual(mean_dose_pp, mean_dose_target)

    # Test max dose %
    max_dose_pp = dvh.max_dose
    max_dose = 7631.604166666671
    max_dose_target = DoseValue(max_dose / prescribed_dose * 100, DoseUnit.Percent)
    test_case.assertAlmostEqual(max_dose_pp, max_dose_target)

    # test dose D95%[%]
    query_str = 'D95%[%]'
    # read query into que object
    query_extensions.read(query_str)
    # execute the static method
    md = query_extensions.query_dose(dvh, query_extensions)
    test_case.assertAlmostEqual(md.value, 6103.854532025905 / 7000.0 * 100)


def test_set_volume_focused_data(dvh):
    v = dvh.volume_focused_format
    d = dvh.dose_focused_format
    assert len(v) == len(d)
