# GETTING dvh DATA FROM DOSE
import os
from unittest import TestCase

import numpy as np
import numpy.testing as npt

from constraints.metrics import PlanningItem
from constraints.tests import rp_dcm, rs_dcm, rd_dcm, DATA_DIR
from core.types import DoseValue, DoseUnit, VolumePresentation, DoseValuePresentation
from pyplanscoring.core.dvhcalculation import load

dvh_path = os.path.join(DATA_DIR, 'PyPlanScoring_dvh.dvh')

pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)

pyplan_dvh = load(dvh_path)
dvh = pyplan_dvh['DVH']

names = ['PTV70 GH',
         'PTV63 GH',
         'PTV56-63 3MM',
         'BODY',
         'BRACHIAL PLEXUS',
         'Brain',
         'BRAINSTEM',
         'BRAINSTEM PRV',
         'CombinedParotids',
         'CTV 56',
         'CTV 63',
         'CTV 70',
         'ESOPHAGUS',
         'GTV',
         'LARYNX',
         'LIPS',
         'ANT CHAMBER LT',
         'COCHLEA LT',
         'EYE LT',
         'LACRIMAL G. LT',
         'LENS LT',
         'OPTIC N. LT',
         'PAROTID LT',
         'MANDIBLE',
         'OPTIC CHIASM',
         'OPTIC CHIASM PRV',
         'ORAL CAVITY',
         'POST NECK',
         'POST-CRICOID',
         'PTV56',
         'PTV63',
         'PTV70',
         'ANT CHAMBER RT',
         'COCHLEA RT',
         'EYE RT',
         'LACRIMAL G. RT',
         'LENS RT',
         'OPTIC N. RT',
         'PAROTID RT',
         'SPINAL CORD',
         'SPINAL CORD PRV',
         'THYROID GLAND',
         'OPTIC N. RT PRV',
         'PTV63-70 3MM',
         'PTV70-BR.PLX 4MM',
         'OPTIC N. LT PRV',
         'PTV63-BR.PLX 1MM']


class TestPlanningItem(TestCase):
    def test_approval_status(self):
        ap = pi.approval_status
        self.assertEqual(ap, "UNAPPROVED")

    def test_plan(self):
        assert pi.plan

    def test_dose_data(self):
        assert pi.dose_data

    def test_beams(self):
        # test property beams
        assert len(pi.beams) > 0

    def test_dose_value_presentation(self):
        pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        assert pi.dose_value_presentation == 1

    def test_total_prescribed_dose(self):
        self.assertAlmostEqual(pi.total_prescribed_dose.value, 70.0)

    def test_treatment_orientation(self):
        target = np.array(['1', '0', '0', '0', '1', '0'], dtype=float)
        npt.assert_array_almost_equal(pi.treatment_orientation, target)

    def test_get_structures(self):
        assert len(pi.get_structures()) > 0

    def test_contains_structure(self):
        # pi.contains_structure('spinal cord')
        m, sm = pi.contains_structure('Spinal Cord')
        assert m
        m, sm = pi.contains_structure('PTV70 GH')
        assert m
        m, sm = pi.contains_structure('P')
        assert not m
        m, sm = pi.contains_structure('O')
        assert not m
        m, sm = pi.contains_structure('PTV70-BR.PLX 4MM')
        assert m

    def test_get_structure(self):
        struc = pi.get_structure('PTV56')
        assert struc

        struc1 = pi.get_structure('Spinal Cord')
        assert struc1

        struc2 = pi.get_structure('spinal cord')
        assert struc2

        struc3 = pi.get_structure('SPINAL CORD')
        assert struc3

        struc4 = pi.get_structure('SPINAL Coord')
        assert struc4

        struc = pi.get_structure('XSUGUA')
        assert struc == "Structure XSUGUA not found"

    def test_creation_date_time(self):
        a_dt = pi.creation_date_time
        assert a_dt == '20170331'

    def test_get_dvh_cumulative_data(self):
        struc_name = 'PTV70'
        dvh_dose_abs = pi.get_dvh_cumulative_data(struc_name, DoseValuePresentation.Absolute)
        dvh_dose_rel = pi.get_dvh_cumulative_data(struc_name, DoseValuePresentation.Relative)

        # check relative and absolute representations
        assert dvh_dose_abs.dose_unit == DoseUnit.Gy
        assert dvh_dose_rel.dose_unit == DoseUnit.Percent

    def test_get_dose_at_volume(self):
        struc_name = 'PTV_70_3mm'
        # query_str = 'D95%[cGy]'
        target_dose = DoseValue(6103.854532025905, DoseUnit.cGy)
        volume_pp = 95 * VolumePresentation.relative
        dose_0 = pi.get_dose_at_volume(struc_name, volume_pp,
                                       VolumePresentation.relative,
                                       DoseValuePresentation.Absolute)

        self.assertAlmostEqual(dose_0, target_dose)

        # query_str = 'D95%[%]'
        target_dose_pp = DoseValue(6103.854532025905 / 7000.0 * 100, DoseUnit.Percent)
        volume_pp = 95 * VolumePresentation.relative
        dose_1 = pi.get_dose_at_volume(struc_name, volume_pp,
                                       VolumePresentation.relative,
                                       DoseValuePresentation.Relative)
        self.assertAlmostEqual(dose_1, target_dose_pp)

        # query_str = 'D655.0261147733513cc[%]'
        volume = 655.0261147733513 * VolumePresentation.absolute_cm3
        dose_1 = pi.get_dose_at_volume(struc_name, volume,
                                       VolumePresentation.absolute_cm3,
                                       DoseValuePresentation.Relative)
        self.assertAlmostEqual(dose_1, target_dose_pp)

        # query_str = 'D655.0261147733513cc[Gy]'
        dose_1 = pi.get_dose_at_volume(struc_name, volume,
                                       VolumePresentation.absolute_cm3,
                                       DoseValuePresentation.Absolute)
        self.assertAlmostEqual(dose_1, target_dose)

    def test_get_dose_compliment_at_volume(self):
        struc_name = 'PTV_70_3mm'
        query_str = 'DC95%[cGy]'
        target_dose = DoseValue(7401.78624315853, DoseUnit.cGy)
        target_dose_pp = DoseValue(7401.78624315853 / 7000.0 * 100, DoseUnit.Percent)
        volume_pp = 95 * VolumePresentation.relative
        dose_0 = pi.get_dose_compliment_at_volume(struc_name, volume_pp,
                                                  VolumePresentation.relative,
                                                  DoseValuePresentation.Absolute)

        dose_1 = pi.get_dose_compliment_at_volume(struc_name, volume_pp,
                                                  VolumePresentation.relative,
                                                  DoseValuePresentation.Relative)

        self.assertAlmostEqual(dose_0, target_dose)
        self.assertAlmostEqual(dose_1, target_dose_pp)

        # Do 95% from absolute values
        query_str = 'DC655.0261147733513cc[cGy]'
        volume = 655.0261147733513 * VolumePresentation.absolute_cm3
        dose_0 = pi.get_dose_compliment_at_volume(struc_name, volume,
                                                  VolumePresentation.relative,
                                                  DoseValuePresentation.Absolute)

        dose_1 = pi.get_dose_compliment_at_volume(struc_name, volume,
                                                  VolumePresentation.relative,
                                                  DoseValuePresentation.Relative)

        self.assertAlmostEqual(dose_0, target_dose)
        self.assertAlmostEqual(dose_1, target_dose_pp)

    def test_get_volume_at_dose(self):
        struc_name = 'PTV_70_3mm'
        query_str = 'V6103.854532025905cGy[%]'
        dv = DoseValue(6103.854532025905, DoseUnit.cGy)
        v0 = pi.get_volume_at_dose(struc_name, dv, VolumePresentation.relative)
        self.assertAlmostEqual(v0, 95.0 * VolumePresentation.relative)

        query_str = 'V6103.854532025905cGy[cc]'
        dv = DoseValue(6103.854532025905, DoseUnit.cGy)
        v1 = pi.get_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
        self.assertAlmostEqual(v1, 655.0261147733513 * VolumePresentation.absolute_cm3)

        query_str = 'V87.1979218860843%[%]'
        dv = DoseValue(87.1979218860843, DoseUnit.Percent)
        v3 = pi.get_volume_at_dose(struc_name, dv, VolumePresentation.relative)
        self.assertAlmostEqual(v3, 95.0 * VolumePresentation.relative)

        query_str = 'V87.1979218860843%[cc]'
        dv = DoseValue(87.1979218860843, DoseUnit.Percent)
        v3 = pi.get_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
        self.assertAlmostEqual(v3, 655.0261147733513 * VolumePresentation.absolute_cm3)

    def test_get_compliment_volume_at_dose(self):
        struc_name = 'PTV_70_3mm'
        query_str = 'CV6103.854532025905cGy[%]'
        dv = DoseValue(6103.854532025905, DoseUnit.cGy)
        v0 = pi.get_compliment_volume_at_dose(struc_name, dv, VolumePresentation.relative)
        self.assertAlmostEqual(v0, 5 * VolumePresentation.relative)

        query_str = 'CV6103.854532025905cGy[cc]'
        dv = DoseValue(6103.854532025905, DoseUnit.cGy)
        v1 = pi.get_compliment_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
        self.assertAlmostEqual(v1, 34.47505867228165 * VolumePresentation.absolute_cm3)

    def test_execute_query(self):
        # Dose at volume
        struc_name = 'PTV_70_3mm'
        mayo_format_query = 'D95%[cGy]'
        dose_0 = pi.execute_query(mayo_format_query, struc_name)
        target_dose = DoseValue(6103.854532025905, DoseUnit.cGy)
        self.assertAlmostEqual(dose_0, target_dose)

        # Volume at dose
        mayo_format_query = 'V6103.854532025905cGy[%]'
        volume = pi.execute_query(mayo_format_query, struc_name)
        target_volume = 95 * VolumePresentation.relative
        self.assertAlmostEqual(volume, target_volume)

        # Dose compliment
        mayo_format_query = 'CV6103.854532025905cGy[%]'
        dose_1 = pi.execute_query(mayo_format_query, struc_name)
        self.assertAlmostEqual(dose_1, 5 * VolumePresentation.relative)

        # Test point dose constrains
        mayo_format_query = 'Max[Gy]'
        # read query into que object
        dm = pi.execute_query(mayo_format_query, struc_name)
        target_dose = DoseValue(76.31604166666671, DoseUnit.Gy)
        self.assertAlmostEqual(dm, target_dose)

        mayo_format_query = 'Max[%]'
        # read query into que object
        dm = pi.execute_query(mayo_format_query, struc_name)
        target_dose = DoseValue(109.02291666666673, DoseUnit.Percent)
        self.assertAlmostEqual(dm, target_dose)

        mayo_format_query = 'Mean[Gy]'
        # read query into que object
        dm = pi.execute_query(mayo_format_query, struc_name)
        target_dose = DoseValue(6949.34536891202, DoseUnit.cGy)
        self.assertAlmostEqual(dm, target_dose)

        mayo_format_query = 'Mean[%]'
        # read query into que object
        dm = pi.execute_query(mayo_format_query, struc_name)
        target_dose = DoseValue(99.27636241302885, DoseUnit.Percent)
        self.assertAlmostEqual(dm, target_dose)

        mayo_format_query = 'Min[Gy]'
        # read query into que object
        dm = pi.execute_query(mayo_format_query, struc_name)
        target_dose = DoseValue(34.570138887638896, DoseUnit.Gy)
        self.assertAlmostEqual(dm, target_dose)

        mayo_format_query = 'Min[%]'
        # read query into que object
        dm = pi.execute_query(mayo_format_query, struc_name)
        target_dose = DoseValue(49.385912696626995, DoseUnit.Percent)
        self.assertAlmostEqual(dm, target_dose)

        struc_name = 'PTV70-BR.PLX 4MM'
        # teste HI index
        mayo_format_query = 'HI70Gy[]'
        target = 0.143276785714286
        dm = pi.execute_query(mayo_format_query, struc_name)
        self.assertAlmostEqual(dm, target, places=3)

        # test CI
        mayo_format_query = 'CI66.5Gy[]'
        target = 0.684301239322868
        dm = pi.execute_query(mayo_format_query, struc_name)
        self.assertAlmostEqual(dm, target, places=1)

        # teste GI
        mayo_format_query = 'GI66.5Gy[]'
        target = 0.684301239322868
        dm = pi.execute_query(mayo_format_query, struc_name)

        self.assertAlmostEqual(dm, dm, places=1)
