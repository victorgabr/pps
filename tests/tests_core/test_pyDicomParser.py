'''
Test cases DICOM objs
'''


def test_get_tps_data(rp_dcm):
    # get tps data from dicom rt-plan
    tps = rp_dcm.get_tps_data()
    assert tps


def test_get_iso_position(rp_dcm):
    # get iso position from dicom rt_plan
    iso = rp_dcm.get_iso_position()
    assert any(iso)


def test_GetStructures(rs_dcm):
    # getting structures dict including centroid
    structures = rs_dcm.GetStructures()
    assert structures


def test_DoseRegularGridInterpolator(rd_dcm):
    dose_interp, (x, y, z), (fx, fy, fz) = rd_dcm.DoseRegularGridInterpolator()


def test_global_max(rd_dcm):
    assert rd_dcm.global_max


def test_GetDVHs(rd_dcm):
    assert rd_dcm.GetDVHs()


def test_GetSOPClassUID(rd_dcm, rs_dcm, rp_dcm):
    assert rd_dcm.GetSOPClassUID() == 'rtdose'
    assert rs_dcm.GetSOPClassUID() == 'rtss'
    assert rp_dcm.GetSOPClassUID() == 'rtplan'


def test_GetPlan(rp_dcm):
    assert rp_dcm.GetPlan()


def test_GetReferencedBeamsInFraction(rp_dcm):
    assert rp_dcm.GetReferencedBeamsInFraction()
