from unittest import TestCase

import pytest

from constraints.query import MayoQueryReader



# instantiate the MayoReader class

rd = MayoQueryReader()


class TestMayoQueryReader(TestCase):
    def test_read(self):
        # Dose at % volume Gy
        query0 = 'D90%[Gy]'

        mq0 = rd.read(query0)
        assert mq0.query_type == 2
        assert mq0.query_value == 90.0
        assert mq0.query_units == 1
        assert mq0.units_desired == 2
        assert mq0.to_string() == query0

    def test_read_query_value(self):
        assert rd.read_query_value('D90%[Gy]') == 90.0
        assert rd.read_query_value('D0.01%[Gy]') == 0.01
        assert rd.read_query_value('Max[cGy]') is None
        assert rd.read_query_value('Mean[cGy]') is None
        assert rd.read_query_value('Min[cGy]') is None
        # Read values from CI and HI
        assert rd.read_query_value('CI50.5Gy') == 50.5
        assert rd.read_query_value('CI4500cGy') == 4500

    def test_is_valid(self):
        # Dose at vol %
        assert rd.is_valid('D90%[Gy]')
        # Dose at vol cc
        assert rd.is_valid('D1cc[cGy]')
        # Dose at vol %
        assert rd.is_valid('D90%[cGy]')

        # Dose at vol %
        assert rd.is_valid('DC90%[Gy]')
        # Dose at vol cc
        assert rd.is_valid('DC1cc[%]')
        # Dose at vol %
        assert rd.is_valid('DC90%[cGy]')

        # Volumes
        # volume cc at dose cGy
        assert rd.is_valid('V90cGy[cc]')
        # volume % dose Gy
        assert rd.is_valid('V90Gy[%]')
        # volume % dose Gy
        assert rd.is_valid('V90Gy[%]')
        # volume % dose %
        assert rd.is_valid('V90%[%]')

        # Compliment volumes
        # volume cc at dose cGy
        assert rd.is_valid('CV90cGy[cc]')
        # volume % dose Gy
        assert rd.is_valid('CV90Gy[%]')
        # volume % dose Gy
        assert rd.is_valid('CV90Gy[%]')
        # volume % dose %
        assert rd.is_valid('CV90%[%]')

        # Max mean min
        assert rd.is_valid('Max[cGy]')
        assert rd.is_valid('Mean[%]')
        assert rd.is_valid('Min[Gy]')
        # CI and HI
        assert rd.is_valid('CI65.4Gy[%]')
        # CI and HI
        assert rd.is_valid('HI65.4Gy[]')
        # CI and HI - no explicit unit

    def test_read_query_type(self):
        assert rd.read_query_type('V90%[cc]') == 0
        assert rd.read_query_type('CV90%[cc]') == 1
        assert rd.read_query_type('D90%[Gy]') == 2
        assert rd.read_query_type('DC90%[Gy]') == 3
        assert rd.read_query_type('Mean[Gy]') == 4
        assert rd.read_query_type('Min[Gy]') == 5
        assert rd.read_query_type('Max[Gy]') == 6
        assert rd.read_query_type('CI[Gy]') == 7
        assert rd.read_query_type('HI[Gy]') == 8

        with pytest.raises(ValueError):
            rd.read_query_type('IC[Gy]')

    def test_read_query_units(self):
        # test units
        # no query unit
        assert rd.read_query_units('Max[cGy]') == 4
        # cc
        assert rd.read_query_units('D90cc[cGy]') == 0
        # %
        assert rd.read_query_units('D90%[cGy]') == 1
        # no
        assert rd.read_query_units('D90[cGy]') == 4
        # v at dose
        assert rd.read_query_units('V40Gy[cGy]') == 2
        assert rd.read_query_units('V400cGy[cGy]') == 3
        # V no query unit
        assert rd.read_query_units('V40[cGy]') == 4
        # CI and HI
        # no
        assert rd.read_query_units('CI40') == 4
        assert rd.read_query_units('CI40Gy') == 2
        assert rd.read_query_units('CI40cGy') == 3
        assert rd.read_query_units('HI40') == 4
        assert rd.read_query_units('HI40Gy') == 2
        assert rd.read_query_units('HI40cGy') == 3

    def test_read_units_desired(self):
        # read units at []
        un = rd.read_units_desired('Max[cGy]')
        assert un == 3
        # read units
        un = rd.read_units_desired('Max[Gy]')
        assert un == 2
        # read units %
        un = rd.read_units_desired('Max[%]')
        assert un == 1
        # read volume at dose
        un = rd.read_units_desired('V90cGy[cc]')
        assert un == 0
        # read volume at dose
        un = rd.read_units_desired('V90Gy[%]')
        assert un == 1
        # test unit not set
        nu = rd.read_units_desired('CI65.4Gy')
        assert nu == 4
        # test unit not set
        nu = rd.read_units_desired('CI65.4Gy[]')
        assert nu == 4

    def test_convert_string_to_unit(self):
        un = rd.convert_string_to_unit('cGy')
        assert un == 3
        # read units
        un = rd.convert_string_to_unit('Gy')
        assert un == 2
        # read units %
        un = rd.convert_string_to_unit('%')
        assert un == 1
        # read volume at dose
        un = rd.convert_string_to_unit('cc')
        assert un == 0
        # read volume at dose
        un = rd.convert_string_to_unit('%')
        assert un == 1
        # test unit not set
        nu = rd.convert_string_to_unit('NA')
        assert nu == 4
        # test unit not set
        nu = rd.convert_string_to_unit('')
        assert nu == 4
