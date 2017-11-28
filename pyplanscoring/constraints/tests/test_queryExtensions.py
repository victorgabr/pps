import os
from unittest import TestCase

from constraints import DVHData, VolumePresentation, DoseValue, DoseUnit
from constraints import PlanningItem
from constraints import QueryExtensions
from core.dicom_reader import ScoringDicomParser

# GETTING dvh DATA FROM DOSE
DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# DATA_DIR = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data'

rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')

rp_dcm = ScoringDicomParser(filename=rp)
rs_dcm = ScoringDicomParser(filename=rs)
rd_dcm = ScoringDicomParser(filename=rd)

# initializing the objects
dvh_all = rd_dcm.GetDVHs()
dvh = DVHData(dvh_all[61])
rd = QueryExtensions()

pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)


class TestQueryExtensions(TestCase):
    def test_get_dose_presentation(self):
        # Dose at % volume Gy
        query = 'D90%[Gy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        query = 'DC90%[Gy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        # Dose at % volume cGy
        query = 'D90%[cGy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        # Dose at cc volume cGy
        query = 'D0.1cc[cGy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        # volume at % dose
        query = 'V95%[%]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 0
        # volume at cGy dose
        query = 'V95%[cGy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 0
        # volume at cGy dose
        query = 'V20Gy[cGy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        # mean dose
        query = 'Mean[cGy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        # min dose
        query = 'Min[cGy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        # max dose
        query = 'Max[cGy]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        # CI
        query = 'CI47Gy[]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        query = 'CI47.5Gy[]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1
        query = 'HI47.5Gy[]'
        mq = rd.read(query)
        assert rd.get_dose_presentation(mq) == 1

    def test_get_dose_unit(self):
        # Dose at % volume Gy
        query = 'D90%[Gy]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'Gy'
        # Dose at % volume cGy
        query = 'D90%[cGy]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'cGy'
        # Dose at cc volume cGy
        query = 'D0.1cc[cGy]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'cGy'
        # volume at % dose
        query = 'V95%[%]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == '%'

        # volume at cGy dose
        query = 'V95%[cc]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == '%'

        # volume at cGy dose
        query = 'V20Gy[%]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'Gy'
        # mean dose
        query = 'Mean[cGy]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'cGy'
        # min dose
        query = 'Min[Gy]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'Gy'
        # max dose
        query = 'Max[cGy]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'cGy'
        # CI
        query = 'CI47Gy[]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'Gy'
        # HI
        query = 'HI47Gy[]'
        mq = rd.read(query)
        un = rd.get_dose_unit(mq)
        assert un.symbol == 'Gy'

    def test_get_volume_presentation(self):
        query = 'D90%[Gy]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == '%'
        query = 'D90cc[Gy]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == 'cc'
        query = 'Min[Gy]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == 'dimensionless'
        query = 'V95%[cc]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == 'cc'
        query = 'V95%[cc]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == 'cc'
        query = 'V95%[%]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == '%'

        query = 'CI47Gy[]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == 'dimensionless'

        query = 'CV47Gy[cc]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == 'cc'
        query = 'CV47Gy[%]'
        mq = rd.read(query)
        un = rd.get_volume_presentation(mq)
        assert un.symbol == '%'

    def test_query_dose(self):
        query_str = 'D95%[cGy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 6103.854532025905)

        query_str = 'D95%[Gy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 61.03854532025905)

        query_str = 'D100%[Gy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 34.5701388876389)

        # absolute volume D100%
        query_str = 'D689.501173445633cc[Gy]'
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 34.5701388876389)

        # Do 95% from absolute values
        query_str = 'D655.0261147733513cc[Gy]'
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 61.03854532025905)

        # # TODO query percent doses normalized
        # query_str = 'D95%[%]'
        # # read query into que object
        # rd.read(query_str)
        # # execute the static method
        # md = rd.query_dose(dvh, rd)
        # self.assertAlmostEqual(md.value, 6103.854532025905)

    def test_query_dose_compliment(self):
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

        query_str = 'DC95%[Gy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose_compliment(dvh, rd)
        self.assertAlmostEqual(md.value, 74.0178624315853)

        query_str = 'DC100%[Gy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose_compliment(dvh, rd)
        self.assertAlmostEqual(md.value, 76.31604166666671)

        # absolute volume D100%
        query_str = 'DC689.501173445633cc[Gy]'
        rd.read(query_str)
        # execute the static method
        md = rd.query_dose_compliment(dvh, rd)
        self.assertAlmostEqual(md.value, 76.3160416666667)

    def test_query_max_dose(self):
        query_str = 'Max[cGy]'
        # read query into que object
        rd.read(query_str)
        md = rd.query_max_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 7631.604166666671)

        query_str = 'Max[Gy]'
        # read query into que object
        rd.read(query_str)
        md = rd.query_max_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 76.31604166666671)

    def test_query_min_dose(self):
        query_str = 'Min[cGy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_min_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 3457.0138887638896)

        query_str = 'Min[Gy]'
        # read query into que object
        rd.read(query_str)
        md = rd.query_min_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 34.570138887638896)

    def test_query_mean_dose(self):
        query_str = 'Mean[cGy]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_mean_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 6949.34536891202)

        query_str = 'Mean[Gy]'
        # read query into que object
        rd.read(query_str)
        md = rd.query_mean_dose(dvh, rd)
        self.assertAlmostEqual(md.value, 69.4934536891202)

        # Todo test relative dose and volume ?

    def test_query_volume(self):
        query_str = 'V6103.854532025905cGy[%]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_volume(dvh, rd)
        self.assertAlmostEqual(md, 95.0 * VolumePresentation.relative)

        query_str = 'V6103.854532025905cGy[cc]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_volume(dvh, rd)
        self.assertAlmostEqual(md, 655.0261147733513 * VolumePresentation.absolute_cm3)

        # Test Query volumes at relative doses
        local_dvh = dvh

        query_str = 'V87.1979218860843%[%]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        local_dvh.to_relative_dose(DoseValue(7000, DoseUnit.cGy))
        md_pp = rd.query_volume(local_dvh, rd)
        self.assertAlmostEqual(md_pp, 95.0 * VolumePresentation.relative)

        query_str = 'V87.19792188608436%[cc]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md_cc = rd.query_volume(local_dvh, rd)
        self.assertAlmostEqual(md_cc, 655.0261147733513 * VolumePresentation.absolute_cm3)

    def test_query_compliment_volume(self):
        query_str = 'CV6103.854532025905cGy[%]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_compliment_volume(dvh, rd)
        self.assertAlmostEqual(md, 5 * VolumePresentation.relative)

        query_str = 'CV6103.854532025905cGy[cc]'
        # read query into que object
        rd.read(query_str)
        # execute the static method
        md = rd.query_compliment_volume(dvh, rd)
        self.assertAlmostEqual(md, 34.47505867228165 * VolumePresentation.absolute_cm3)
