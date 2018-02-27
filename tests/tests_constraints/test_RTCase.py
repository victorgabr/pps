import pytest


def test_case_info(rt_case):
    assert not rt_case.metrics.empty
    # self.fail()

    assert rt_case.structures

    assert rt_case.name

    assert rt_case.case_id


def test_calc_structures(rt_case):
    assert rt_case.calc_structures


def test_get_structure(rt_case):
    struc = rt_case.get_structure('PTV56')
    assert struc

    struc1 = rt_case.get_structure('Spinal Cord')
    assert struc1

    struc2 = rt_case.get_structure('spinal cord')
    assert struc2

    struc3 = rt_case.get_structure('SPINAL CORD')
    assert struc3

    struc4 = rt_case.get_structure('SPINAL Coord')
    assert struc4

    with pytest.raises(ValueError):
        struc = rt_case.get_structure('XSUGUA')



def test_get_external(rt_case):
    external = rt_case.get_external()
    assert external