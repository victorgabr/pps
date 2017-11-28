from unittest import TestCase

from constraints import MayoQueryWriter, MayoQueryReader

rd = MayoQueryReader()
wt = MayoQueryWriter()


class TestMayoQueryWriter(TestCase):
    def test_write(self):
        # Dose at % volume Gy
        query0 = 'D90%[Gy]'
        mq0 = rd.read(query0)
        assert wt.write(mq0) == query0

        # Dose at % volume cGy
        query1 = 'D90%[cGy]'
        mq1 = rd.read(query1)
        assert wt.write(mq1) == query1

        # Dose at cc volume cGy
        query = 'D0.1cc[cGy]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        # volume at % dose
        query = 'V95%[%]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        # volume at cGy dose
        query = 'V95%[cGy]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        # mean dose
        query = 'Mean[cGy]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        # min dose
        query = 'Min[cGy]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        # max dose
        query = 'Max[cGy]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        # CI
        query = 'CI47Gy[]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        query = 'CI47.5Gy[]'
        mq = rd.read(query)
        assert wt.write(mq) == query

        query = 'HI47.5Gy[]'
        mq = rd.read(query)
        assert wt.write(mq) == query

    def test_get_type_string(self):
        assert wt.get_type_string(0) == 'V'
        assert wt.get_type_string(1) == 'CV'
        assert wt.get_type_string(2) == 'D'
        assert wt.get_type_string(3) == 'DC'
        assert wt.get_type_string(4) == 'Mean'
        assert wt.get_type_string(5) == 'Min'
        assert wt.get_type_string(6) == 'Max'
        assert wt.get_type_string(7) == 'CI'
        assert wt.get_type_string(8) == 'HI'

    def test_get_value_string(self):
        assert wt.get_value_string(1.0) == '1'
        assert wt.get_value_string(0.01) == '0.01'
        assert wt.get_value_string(65.5) == '65.5'
        assert wt.get_value_string(None) == ''

    def test_get_unit_string(self):
        assert wt.get_unit_string(0) == 'cc'
        assert wt.get_unit_string(1) == '%'
        assert wt.get_unit_string(2) == 'Gy'
        assert wt.get_unit_string(3) == 'cGy'
        assert wt.get_unit_string(4) == ''
