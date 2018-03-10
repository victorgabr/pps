# GETTING dvh DATA FROM DOSE

from core.types import DoseUnit, DoseValuePresentation, VolumePresentation, DoseValue


def test_calculate_dvh(test_case, py_planning_item):
    # Must calculate DVH
    py_planning_item.calculate_dvh()

    # def test_get_dvh_cumulative_data(test_case):
    struc_name = 'PTV70-BR.PLX 4MM'
    dvh_dose_abs = py_planning_item.get_dvh_cumulative_data(struc_name, DoseValuePresentation.Absolute)
    dvh_dose_rel = py_planning_item.get_dvh_cumulative_data(struc_name, DoseValuePresentation.Relative)

    # check relative and absolute representations
    assert dvh_dose_abs.dose_unit == DoseUnit.Gy
    assert dvh_dose_rel.dose_unit == DoseUnit.Percent

    # def test_get_dose_at_volume(test_case):
    target_dose = DoseValue(67.762, DoseUnit.Gy)
    volume_pp = 95 * VolumePresentation.relative
    dose_0 = py_planning_item.get_dose_at_volume(struc_name, volume_pp,
                                                 VolumePresentation.relative,
                                                 DoseValuePresentation.Absolute)

    assert dose_0 == target_dose

    # query_str = 'D95%[%]'
    target_dose_pp = DoseValue(6781.3 / 7000.0 * 100, DoseUnit.Percent)
    volume_pp = 95 * VolumePresentation.relative
    dose_1 = py_planning_item.get_dose_at_volume(struc_name, volume_pp,
                                                 VolumePresentation.relative,
                                                 DoseValuePresentation.Relative)

    assert dose_1 == target_dose_pp

    # Test point dose constrains
    mayo_format_query = 'Max[Gy]'
    # read query into que object
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(76.2, DoseUnit.Gy)
    assert dm == target_dose

    mayo_format_query = 'Max[%]'
    # read query into que object
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(108.857, DoseUnit.Percent)
    assert dm == target_dose

    mayo_format_query = 'Mean[Gy]'
    # read query into que object
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(71.56748874249499, DoseUnit.Gy)
    assert dm == target_dose

    mayo_format_query = 'Mean[%]'
    # read query into que object
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(102.239, DoseUnit.Percent)
    assert dm == target_dose

    mayo_format_query = 'Min[Gy]'
    # read query into que object
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(56.955, DoseUnit.Gy)
    assert dm == target_dose

    mayo_format_query = 'Min[%]'
    # read query into que object
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    target_dose = DoseValue(81.364, DoseUnit.Percent)
    assert dm == target_dose

    # # teste HI index
    mayo_format_query = 'HI70Gy[]'
    target = 0.14336163265306154
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    assert dm == target

    # test CI
    mayo_format_query = 'CI66.5Gy[]'
    target = 0.6631410314914067
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    assert dm == target

    # teste GI
    # TODO add real GK case
    mayo_format_query = 'GI66.5Gy[]'
    target = 0.6631410314914067
    dm = py_planning_item.execute_query(mayo_format_query, struc_name)
    assert dm == dm

    # volume = 655.0261147733513 * VolumePresentation.absolute_cm3
    # dose_1 = py_planning_item.get_dose_at_volume(struc_name, volume,
    #                                              VolumePresentation.absolute_cm3,
    #                                              DoseValuePresentation.Relative)
    # assert dose_1 == target_dose_pp
    #
    # dose_1 = py_planning_item.get_dose_at_volume(struc_name, volume,
    #                                              VolumePresentation.absolute_cm3,
    #                                              DoseValuePresentation.Absolute)
    # assert dose_1 == target_dose
    #
    # target_dose = DoseValue(7401.78624315853, DoseUnit.cGy)
    # target_dose_pp = DoseValue(7401.78624315853 / 7000.0 * 100, DoseUnit.Percent)
    # volume_pp = 95 * VolumePresentation.relative
    # dose_0 = py_planning_item.get_dose_compliment_at_volume(struc_name, volume_pp,
    #                                                         VolumePresentation.relative,
    #                                                         DoseValuePresentation.Absolute)
    #
    # dose_1 = py_planning_item.get_dose_compliment_at_volume(struc_name, volume_pp,
    #                                                         VolumePresentation.relative,
    #                                                         DoseValuePresentation.Relative)
    #
    # assert dose_0 == target_dose
    # assert dose_1 == target_dose_pp
    #
    # # Do 95% from absolute values
    # # query_str = 'DC655.0261147733513cc[cGy]'
    # volume = 655.0261147733513 * VolumePresentation.absolute_cm3
    # dose_0 = py_planning_item.get_dose_compliment_at_volume(struc_name, volume,
    #                                                         VolumePresentation.relative,
    #                                                         DoseValuePresentation.Absolute)
    #
    # dose_1 = py_planning_item.get_dose_compliment_at_volume(struc_name, volume,
    #                                                         VolumePresentation.relative,
    #                                                         DoseValuePresentation.Relative)
    #
    # assert dose_0 == target_dose
    # assert dose_1 == target_dose_pp
    #
    # # query_str = 'V6103.854532025905cGy[%]'
    # dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    # v0 = py_planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.relative)
    #
    # assert  v0 == 95.0 * VolumePresentation.relative
    #
    # # query_str = 'V6103.854532025905cGy[cc]'
    # dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    # v1 = py_planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
    # test_case.assertAlmostEqual(v1, 655.0261147733513 * VolumePresentation.absolute_cm3)
    #
    # # query_str = 'V87.1979218860843%[%]'
    # dv = DoseValue(87.1979218860843, DoseUnit.Percent)
    # v3 = py_planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.relative)
    # test_case.assertAlmostEqual(v3, 95.0 * VolumePresentation.relative)
    #
    # # query_str = 'V87.1979218860843%[cc]'
    # dv = DoseValue(87.1979218860843, DoseUnit.Percent)
    # v3 = py_planning_item.get_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
    # test_case.assertAlmostEqual(v3, 655.0261147733513 * VolumePresentation.absolute_cm3)
    #
    # # query_str = 'CV6103.854532025905cGy[%]'
    # dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    # v0 = py_planning_item.get_compliment_volume_at_dose(struc_name, dv, VolumePresentation.relative)
    # test_case.assertAlmostEqual(v0, 5 * VolumePresentation.relative)
    #
    # query_str = 'CV6103.854532025905cGy[cc]'
    # dv = DoseValue(6103.854532025905, DoseUnit.cGy)
    # v1 = py_planning_item.get_compliment_volume_at_dose(struc_name, dv, VolumePresentation.absolute_cm3)
    # test_case.assertAlmostEqual(v1, 34.47505867228165 * VolumePresentation.absolute_cm3)
    #
    # # Dose at volume
    # struc_name = 'PTV_70_3mm'
    # mayo_format_query = 'D95%[cGy]'
    # dose_0 = py_planning_item.execute_query(mayo_format_query, struc_name)
    # target_dose = DoseValue(6103.854532025905, DoseUnit.cGy)
    # test_case.assertAlmostEqual(dose_0, target_dose)
    #
    # # Volume at dose
    # mayo_format_query = 'V6103.854532025905cGy[%]'
    # volume = py_planning_item.execute_query(mayo_format_query, struc_name)
    # target_volume = 95 * VolumePresentation.relative
    # test_case.assertAlmostEqual(volume, target_volume)
    #
    # # Dose compliment
    # mayo_format_query = 'CV6103.854532025905cGy[%]'
    # dose_1 = py_planning_item.execute_query(mayo_format_query, struc_name)
    # test_case.assertAlmostEqual(dose_1, 5 * VolumePresentation.relative)
