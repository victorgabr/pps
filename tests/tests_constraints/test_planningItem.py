# GETTING dvh DATA FROM DOSE
import os

import numpy as np

from constraints.metrics import PlanningItem
from core.dicom_reader import PyDicomParser
from core.types import DoseUnit, DoseValuePresentation, VolumePresentation, DoseValue
from tests.conftest import DATA_DIR, rp

# given
rd_dvh = os.path.join(DATA_DIR, 'RD_dvh.dcm')
rs_dvh = os.path.join(DATA_DIR, 'RS_dvh.dcm')
planning_item = PlanningItem(PyDicomParser(filename=rp),
                             PyDicomParser(filename=rs_dvh),
                             PyDicomParser(filename=rd_dvh))


def test_planning_item_general_info(test_case):
    ap = planning_item.approval_status
    test_case.assertEqual(ap, "UNAPPROVED")
    a_dt = planning_item.creation_date_time
    assert a_dt == '20171128'

    assert planning_item.plan
    assert planning_item.dose_data
    # test property beams
    assert len(planning_item.beams) > 0
    assert planning_item.dose_value_presentation == 1
    test_case.assertAlmostEqual(planning_item.total_prescribed_dose.value, 70.0)

    target = np.array(['1', '0', '0', '0', '1', '0'], dtype=float)
    np.testing.assert_array_almost_equal(planning_item.treatment_orientation, target)

    assert len(planning_item.get_structures()) > 0


def test_contains_structure(test_case):
    # pi.contains_structure('spinal cord')
    m, sm = planning_item.contains_structure('Spinal Cord')
    assert m
    m, sm = planning_item.contains_structure('PTV70 GH')
    assert m
    m, sm = planning_item.contains_structure('P')
    assert not m
    m, sm = planning_item.contains_structure('O')
    assert not m
    m, sm = planning_item.contains_structure('PTV70-BR.PLX 4MM')
    assert m


def test_get_structure(test_case):
    struc = planning_item.get_structure('PTV56')
    assert struc

    struc1 = planning_item.get_structure('Spinal Cord')
    assert struc1

    struc2 = planning_item.get_structure('spinal cord')
    assert struc2

    struc3 = planning_item.get_structure('SPINAL CORD')
    assert struc3

    struc4 = planning_item.get_structure('SPINAL Coord')
    assert struc4

    struc = planning_item.get_structure('XSUGUA')
    assert struc == "Structure XSUGUA not found"


def test_get_dvh_cumulative_data():
    struc_name = 'PTV70'
    dvh_dose_abs = planning_item.get_dvh_cumulative_data(struc_name, DoseValuePresentation.Absolute)
    dvh_dose_rel = planning_item.get_dvh_cumulative_data(struc_name, DoseValuePresentation.Relative)

    # check relative and absolute representations
    assert dvh_dose_abs.dose_unit == DoseUnit.Gy
    assert dvh_dose_rel.dose_unit == DoseUnit.Percent


def test_get_dose_at_volume(test_case):
    struc_name = 'PTV_70_3mm'
    # query_str = 'D95%[cGy]'
    target_dose = DoseValue(6103.854532025905, DoseUnit.cGy)
    volume_pp = 95 * VolumePresentation.relative
    dose_0 = planning_item.get_dose_at_volume(struc_name, volume_pp,
                                              VolumePresentation.relative,
                                              DoseValuePresentation.Absolute)

    test_case.assertAlmostEqual(dose_0, target_dose)

    # query_str = 'D95%[%]'
    target_dose_pp = DoseValue(6103.854532025905 / 7000.0 * 100, DoseUnit.Percent)
    volume_pp = 95 * VolumePresentation.relative
    dose_1 = planning_item.get_dose_at_volume(struc_name, volume_pp,
                                              VolumePresentation.relative,
                                              DoseValuePresentation.Relative)
    test_case.assertAlmostEqual(dose_1, target_dose_pp)

    # query_str = 'D655.0261147733513cc[%]'
    volume = 655.0261147733513 * VolumePresentation.absolute_cm3
    dose_1 = planning_item.get_dose_at_volume(struc_name, volume,
                                              VolumePresentation.absolute_cm3,
                                              DoseValuePresentation.Relative)
    test_case.assertAlmostEqual(dose_1, target_dose_pp)

    # query_str = 'D655.0261147733513cc[Gy]'
    dose_1 = planning_item.get_dose_at_volume(struc_name, volume,
                                              VolumePresentation.absolute_cm3,
                                              DoseValuePresentation.Absolute)
    test_case.assertAlmostEqual(dose_1, target_dose)


def test_get_dose_compliment_at_volume(test_case):
    struc_name = 'PTV_70_3mm'
    query_str = 'DC95%[cGy]'
    target_dose = DoseValue(7401.78624315853, DoseUnit.cGy)
    target_dose_pp = DoseValue(7401.78624315853 / 7000.0 * 100, DoseUnit.Percent)
    volume_pp = 95 * VolumePresentation.relative
    dose_0 = planning_item.get_dose_compliment_at_volume(struc_name, volume_pp,
                                                         VolumePresentation.relative,
                                                         DoseValuePresentation.Absolute)

    dose_1 = planning_item.get_dose_compliment_at_volume(struc_name, volume_pp,
                                                         VolumePresentation.relative,
                                                         DoseValuePresentation.Relative)

    test_case.assertAlmostEqual(dose_0, target_dose)
    test_case.assertAlmostEqual(dose_1, target_dose_pp)

    # Do 95% from absolute values
    query_str = 'DC655.0261147733513cc[cGy]'
    volume = 655.0261147733513 * VolumePresentation.absolute_cm3
    dose_0 = planning_item.get_dose_compliment_at_volume(struc_name, volume,
                                                         VolumePresentation.relative,
                                                         DoseValuePresentation.Absolute)

    dose_1 = planning_item.get_dose_compliment_at_volume(struc_name, volume,
                                                         VolumePresentation.relative,
                                                         DoseValuePresentation.Relative)

    test_case.assertAlmostEqual(dose_0, target_dose)
    test_case.assertAlmostEqual(dose_1, target_dose_pp)


def test_get_volume_at_dose(test_case):
    struc_name = 'PTV_70_3mm'
    query_str = 'V6103.854532025905cGy[%]'
    dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    v0 = planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.relative)
    test_case.assertAlmostEqual(v0, 95.0 * VolumePresentation.relative)

    query_str = 'V6103.854532025905cGy[cc]'
    dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    v1 = planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
    test_case.assertAlmostEqual(v1, 655.0261147733513 * VolumePresentation.absolute_cm3)

    query_str = 'V87.1979218860843%[%]'
    dv = DoseValue(87.1979218860843, DoseUnit.Percent)
    v3 = planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.relative)
    test_case.assertAlmostEqual(v3, 95.0 * VolumePresentation.relative)

    query_str = 'V87.1979218860843%[cc]'
    dv = DoseValue(87.1979218860843, DoseUnit.Percent)
    v3 = planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
    test_case.assertAlmostEqual(v3, 655.0261147733513 * VolumePresentation.absolute_cm3)


def test_get_compliment_volume_at_dose(test_case):
    struc_name = 'PTV_70_3mm'
    query_str = 'CV6103.854532025905cGy[%]'
    dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    v0 = planning_item.get_compliment_volume_at_dose(struc_name, dv, VolumePresentation.relative)
    test_case.assertAlmostEqual(v0, 5 * VolumePresentation.relative)

    query_str = 'CV6103.854532025905cGy[cc]'
    dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    v1 = planning_item.get_compliment_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
    test_case.assertAlmostEqual(v1, 34.47505867228165 * VolumePresentation.absolute_cm3)


def test_execute_query(test_case):
    # Dose at volume
    struc_name = 'PTV_70_3mm'
    mayo_format_query = 'D95%[cGy]'
    dose_0 = planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(6103.854532025905, DoseUnit.cGy)
    test_case.assertAlmostEqual(dose_0, target_dose)

    # Volume at dose
    mayo_format_query = 'V6103.854532025905cGy[%]'
    volume = planning_item.execute_query(mayo_format_query, struc_name)
    target_volume = 95 * VolumePresentation.relative
    test_case.assertAlmostEqual(volume, target_volume)

    # Dose compliment
    mayo_format_query = 'CV6103.854532025905cGy[%]'
    dose_1 = planning_item.execute_query(mayo_format_query, struc_name)
    test_case.assertAlmostEqual(dose_1, 5 * VolumePresentation.relative)

    # Test point dose constrains
    mayo_format_query = 'Max[Gy]'
    # read query into que object
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(76.31604166666671, DoseUnit.Gy)
    test_case.assertAlmostEqual(dm, target_dose)

    mayo_format_query = 'Max[%]'
    # read query into que object
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(109.02291666666673, DoseUnit.Percent)
    test_case.assertAlmostEqual(dm, target_dose)

    mayo_format_query = 'Mean[Gy]'
    # read query into que object
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(6949.34536891202, DoseUnit.cGy)
    test_case.assertAlmostEqual(dm, target_dose)

    mayo_format_query = 'Mean[%]'
    # read query into que object
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(99.27636241302885, DoseUnit.Percent)
    test_case.assertAlmostEqual(dm, target_dose)

    mayo_format_query = 'Min[Gy]'
    # read query into que object
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(34.570138887638896, DoseUnit.Gy)
    test_case.assertAlmostEqual(dm, target_dose)

    mayo_format_query = 'Min[%]'
    # read query into que object
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(49.385912696626995, DoseUnit.Percent)
    test_case.assertAlmostEqual(dm, target_dose)

    struc_name = 'PTV70-BR.PLX 4MM'
    # teste HI index
    mayo_format_query = 'HI70Gy[]'
    target = 0.143276785714286
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    test_case.assertAlmostEqual(dm, target, places=3)

    # test CI
    mayo_format_query = 'CI66.5Gy[]'
    target = 0.684301239322868
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    test_case.assertAlmostEqual(dm, target, places=1)

    # teste GI
    # TODO add real GK case
    mayo_format_query = 'GI66.5Gy[]'
    target = 0.684301239322868
    dm = planning_item.execute_query(mayo_format_query, struc_name)
    test_case.assertAlmostEqual(dm, dm, places=1)
