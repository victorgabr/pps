from unittest import TestCase

from core.tests import rs_dcm, rp_dcm, rd_dcm


class TestPyDicomParser(TestCase):
    def test_get_tps_data(self):
        # get tps data from dicom rt-plan
        tps = rp_dcm.get_tps_data()
        assert tps

    def test_get_iso_position(self):
        # get iso position from dicom rt_plan
        iso = rp_dcm.get_iso_position()
        assert any(iso)

    def test_GetStructures(self):
        # getting structures dict including centroid
        structures = rs_dcm.GetStructures()
        assert structures

    def test_DoseRegularGridInterpolator(self):
        dose_interp, (x, y, z), (fx, fy, fz) = rd_dcm.DoseRegularGridInterpolator()

    def test_global_max(self):
        assert rd_dcm.global_max

    def test_GetDVHs(self):
        assert rd_dcm.GetsDVHs()

    def test_GetSOPClassUID(self):
        assert rd_dcm.GetSOPClassUID() == 'rtdose'
        assert rs_dcm.GetSOPClassUID() == 'rtss'
        assert rp_dcm.GetSOPClassUID() == 'rtplan'

    def test_GetPlan(self):
        assert rp_dcm.GetPlan()

    def test_GetReferencedBeamsInFraction(self):
        assert rp_dcm.GetReferencedBeamsInFraction()
