from pyplanscoring.core.calculation import PyStructure, DVHCalculationMP


def test_calc_data(lens, body, brain, ptv70, spinal_cord, dose_3d):
    grid_up = (0.2, 0.2, 0.2)
    structures_dicom = [lens, body, brain, ptv70, spinal_cord]
    structures_py = [PyStructure(s) for s in structures_dicom]
    grids = [grid_up, None, None, None, None]

    # # test not giving a list of structures
    # with pytest.raises(TypeError):
    #     calc_mp = DVHCalculationMP(dose_3d, (1, 2, 3), grids)
    #
    # # test init structures dose and grids
    # with pytest.raises(TypeError):
    #     calc_mp = DVHCalculationMP(dose_3d, structures_dicom, grids)
    #
    # # test init structures dose and grids
    # with pytest.raises(TypeError):
    #     calc_mp = DVHCalculationMP([], structures_py, grids)
    #
    # # test grid initialization same size
    # with pytest.raises(TypeError):
    #     calc_mp = DVHCalculationMP(dose_3d, structures_py, ())
    #
    # with pytest.raises(TypeError):
    #     calc_mp = DVHCalculationMP(dose_3d, structures_py, [0.1, None])
    #
    # with pytest.raises(TypeError):
    #     calc_mp = DVHCalculationMP(dose_3d, structures_py, ())
    #
    # with pytest.raises(ValueError):
    #     calc_mp = DVHCalculationMP(dose_3d, structures_py, [(0.1, 2), None])
    #     # test give correct grids type
    #
    # # test_ adding structures and grids with different sizes
    # with pytest.raises(ValueError):
    #     calc_mp = DVHCalculationMP(dose_3d, structures_py, grids[:-2])

    # assert calculation data
    calc_mp = DVHCalculationMP(dose_3d, structures_py, grids)
    assert calc_mp.calc_data

    # assert calculation data
    calc_mp = DVHCalculationMP(dose_3d, structures_py, grids)
    struc_dvh = calc_mp.calculate(structures_py[0], grids[0], dose_3d, True)
    assert struc_dvh
