import os
from unittest import TestCase

from pyplanscoring.core.constraints.query import QueryExtensions
from pyplanscoring.core.constraints.types import DVHData, DoseValue, DoseUnit, VolumePresentation
from pyplanscoring.core.dicomparser import ScoringDicomParser

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)
filename = os.path.join(DATA_DIR, 'RD.dcm')

# GETTING dvh DATA FROM DOSE
rd_dcm = ScoringDicomParser(filename=filename)
dvh_all = rd_dcm.GetDVHs()

dvh = DVHData(dvh_all[61])

rd = QueryExtensions()


class TestDVHData(TestCase):
    def test_get_volume_at_dose(self):
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

    def test_get_compliment_volume_at_dose(self):
        # test get volume at dose
        dose_c = DoseValue(7000, DoseUnit.cGy)
        dvh.get_compliment_volume_at_dose(dose_c, VolumePresentation.absolute_cm3)

        # test get c volume at dose
        dose_c1 = DoseValue(70000, DoseUnit.cGy)
        dvh.get_compliment_volume_at_dose(dose_c1, VolumePresentation.absolute_cm3)

    def test_get_dose_at_volume(self):
        query_str = 'D100%[cGy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, dvh.min_dose.value)
        # Do 100 from absolute values
        query_str = 'D689.501173445633cc[cGy]'
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, dvh.min_dose.value)

        query_str = 'D95%[cGy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 6103.854532025905)
        # Do 95% from absolute values
        query_str = 'D655.0261147733513cc[cGy]'
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 6103.854532025905)

    def test_get_dose_compliment(self):
        query_str = 'DC95%[cGy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose_compliment(dvh, rd)
        self.assertAlmostEqual(md.value, 7401.78624315853)

        # Do 95% from absolute values
        query_str = 'DC655.0261147733513cc[cGy]'
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose_compliment(dvh, rd)
        self.assertAlmostEqual(md.value, 7401.78624315853)

    def test_to_relative(self):
        prescribed_dose = 7000.0
        dvh.to_relative_dose(DoseValue(7000, DoseUnit.cGy))
        assert dvh.dose_unit == DoseUnit.Percent

        # Test min dose %
        min_dose_pp = dvh.min_dose
        min_dose_target = DoseValue(3457.0138887638896 / prescribed_dose * 100, DoseUnit.Percent)
        self.assertAlmostEqual(min_dose_pp, min_dose_target)

        # Test mean dose %
        mean_dose_pp = dvh.mean_dose
        mean_dose = 6949.34536891202
        mean_dose_target = DoseValue(mean_dose / prescribed_dose * 100, DoseUnit.Percent)
        self.assertAlmostEqual(mean_dose_pp, mean_dose_target)

        # Test max dose %
        max_dose_pp = dvh.max_dose
        max_dose = 7631.604166666671
        max_dose_target = DoseValue(max_dose / prescribed_dose * 100, DoseUnit.Percent)
        self.assertAlmostEqual(max_dose_pp, max_dose_target)

        # test dose D95%[%]
        query_str = 'D95%[%]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 6103.854532025905 / 7000.0 * 100)

    def test_set_volume_focused_data(self):
        v = dvh.volume_focused_format
        d = dvh.dose_focused_format
        a = 1
        assert len(v) == 31
        assert len(d) == 31
        # import matplotlib.pyplot as plt
        # plt.plot(d, v)
        # plt.show()
        pass
