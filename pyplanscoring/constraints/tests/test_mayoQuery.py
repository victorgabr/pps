from unittest import TestCase

import pytest

from constraints.query import MayoQueryReader, MayoQuery

# initialize mayo query
q_obj = MayoQuery()


class TestMayoQuery(TestCase):
    def test_read(self):
        # Dose at % volume Gy
        query = 'D90%[Gy]'
        q_obj.read(query)
        with pytest.raises(ValueError):
            query = 'D90[Gy]'
            q_obj.read(query)

    def test_to_string(self):
        query0 = 'D90%[Gy]'
        q_obj.read(query0)

        assert q_obj.to_string() == query0
        assert str(q_obj) == 'D90%[Gy]'

    def test_query(self):
        query0 = 'D90%[Gy]'
        q_obj.read(query0)
        assert isinstance(q_obj.query, MayoQuery)


def test_MayoQueryReader():
    """
        Test class MayoQueryReader
        The Mayo format is broken down into the following components:

        Query Type Qt
        Query Value Qv (if necessary)
        Query Units Qu (if necessary)
        Units Desired Ud
        They are ordered as :

        QtQvQu[Ud]
    """
    rd = MayoQueryReader()

    # Dose at % volume Gy
    query0 = 'D90%[Gy]'

    mq0 = rd.read(query0)
    assert mq0.query_type == 2
    assert mq0.query_value == 90.0
    assert mq0.query_units == 1
    assert mq0.units_desired == 2
    assert mq0.to_string() == query0

    # Dose at % volume cGy
    query1 = 'D90%[cGy]'
    mq1 = rd.read(query1)
    assert mq1.query_type == 2
    assert mq1.query_value == 90.0
    assert mq1.query_units == 1
    assert mq1.units_desired == 3
    assert mq1.to_string() == query1

    # Dose at cc volume cGy
    query = 'D0.1cc[cGy]'
    mq = rd.read(query)
    assert mq.query_type == 2
    assert mq.query_value == 0.1
    assert mq.query_units == 0
    assert mq.units_desired == 3
    assert mq.to_string() == query

    # volume at % dose
    query1 = 'V95%[%]'
    mq = rd.read(query1)
    assert mq.query_type == 0
    assert mq.query_value == 95.0
    assert mq.query_units == 1
    assert mq.units_desired == 1
    assert mq.to_string() == query1

    # volume at cGy dose
    query1 = 'V95%[cGy]'
    mq = rd.read(query1)
    assert mq.query_type == 0
    assert mq.query_value == 95.0
    assert mq.query_units == 1
    assert mq.units_desired == 3
    assert mq.to_string() == query1

    # mean dose
    query1 = 'Mean[cGy]'
    mq = rd.read(query1)
    assert mq.query_type == 4
    assert mq.query_value is None
    assert mq.query_units == 4
    assert mq.units_desired == 3
    assert mq.to_string() == query1

    # min dose
    query1 = 'Min[cGy]'
    mq = rd.read(query1)
    assert mq.query_type == 5
    assert mq.query_value is None
    assert mq.query_units == 4
    assert mq.units_desired == 3
    assert mq.to_string() == query1

    # max dose
    query1 = 'Max[cGy]'
    mq = rd.read(query1)
    assert mq.query_type == 6
    assert mq.query_value is None
    assert mq.query_units == 4
    assert mq.units_desired == 3
    assert mq.to_string() == query1
