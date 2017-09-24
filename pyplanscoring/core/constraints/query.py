"""
Classes to implement Mayo DVH query format

based on:

https://rexcardan.github.io/ESAPIX/api/ESAPIX.Constraints.DVH.Query.html

"""
import re

from pyplanscoring.core.constraints.metrics import MaxDoseConstraint, MinDoseConstraint, MinMeanDoseConstraint, \
    MaxMeanDoseConstraint, MinDoseAtVolConstraint, MaxDoseAtVolConstraint, MinVolAtDoseConstraint, \
    MaxVolAtDoseConstraint, MinComplimentVolumeAtDose, MaxComplimentVolumeAtDose, MinComplimentDoseAtVolumeConstraint, \
    MaxComplimentDoseAtVolumeConstraint
from pyplanscoring.core.constraints.types import QueryType, Units, DoseUnit, Discriminator, VolumePresentation, \
    DoseValue, DiscriminatorConverter


class MayoRegex:
    UnitsDesired = r"\[(cc|%|(c?Gy))\]"
    QueryType = r"^(V|CV|DC|D|Mean|Max|Min)"
    QueryValue = r"\d+(\.?)(\d+)?"
    QueryUnits = r"((cc)|%|(c?Gy))"
    # Valid = "(((V|CV|DC|D)(%s%s))|(Mean|Max|Min))%s" % (query_value, query_units, units_desired)
    Valid = '(((V|CV|DC|D)(\d+(\.?)(\d+)?((cc)|%|(c?Gy))))|(Mean|Max|Min))\[(cc|%|(c?Gy))\]'


class MayoQuery:
    def __init__(self):
        self._query_type = None
        self._query_units = None
        self._query_value = None
        self._units_desired = None

    @staticmethod
    def read(query):
        return MayoQueryReader().read(query)

    @property
    def query_type(self):
        return self._query_type

    @query_type.setter
    def query_type(self, value):
        self._query_type = value

    @property
    def query_units(self):
        return self._query_units

    @query_units.setter
    def query_units(self, value):
        self._query_units = value

    @property
    def query_value(self):
        return self._query_value

    @query_value.setter
    def query_value(self, value):
        self._query_value = value

    @property
    def units_desired(self):
        return self._units_desired

    @units_desired.setter
    def units_desired(self, value):
        self._units_desired = value

    def to_string(self):
        return MayoQueryWriter().write(self)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()


class MayoQueryWriter(object):
    def write(self, mayo_query):
        """
             public static string Write(MayoQuery query)
                    {
                        var type = GetTypeString(query.query_type);
                        var qUnits = GetUnitString(query.query_units);
                        var qValue = query.query_value.to_string();
                        var dUnits = GetUnitString(query.units_desired);
                        return $"{type}{qValue}{qUnits}[{dUnits}]";
                    }
        """
        query_type = self.get_type_string(mayo_query.query_type)
        qUnits = self.get_unit_string(mayo_query.query_units)
        qValue = self.get_value_string(mayo_query.query_value)
        dUnits = self.get_unit_string(mayo_query.units_desired)

        return query_type + qValue + qUnits + '[' + dUnits + ']'

    @staticmethod
    def get_type_string(query_type):
        switch = {
            QueryType.COMPLIMENT_VOLUME: "CV",
            QueryType.DOSE_AT_VOLUME: "D",
            QueryType.DOSE_COMPLIMENT: "DC",
            QueryType.MAX_DOSE: "Max",
            QueryType.MEAN_DOSE: "Mean",
            QueryType.MIN_DOSE: "Min",
            QueryType.VOLUME_AT_DOSE: "V"
        }

        return switch.get(query_type)

    @staticmethod
    def get_value_string(query_value):
        if query_value:
            if query_value > 1.0:
                return str(int(query_value))
            else:
                return str(query_value)
        else:
            return ''

    @staticmethod
    def get_unit_string(query_units):
        switch = {
            Units.CC: "cc",
            Units.CGY: "cGy",
            Units.GY: "Gy",
            Units.PERC: "%",
            Units.NA: '',
        }
        return switch.get(query_units)


class MayoQueryReader(object):
    """
     Class with methods to read a DVH query in "Mayo Format" (https://www.ncbi.nlm.nih.gov/pubmed/26825250)
    """

    def read(self, query):
        """
             Reads a full Mayo query string and converts it to a MayoQuery object
        :param query: string mayo query
        :return: Mayo Query object
        """
        if not self.is_valid(query):
            print('Not a valid Mayo format')
            raise ValueError

        mq = MayoQuery()
        mq.query_type = self.read_query_type(query)
        mq.query_units = self.read_query_units(query)
        mq.units_desired = self.read_units_desired(query)
        mq.query_value = self.read_query_value(query)

        return mq

    @staticmethod
    def read_query_value(query):
        """
            Reads only the numerical value in the query (if one exists)
        :param query:
        :return:
        """
        match = re.search(MayoRegex.QueryValue, query)
        if not match:
            return None

        return float(match.group())

    @staticmethod
    def is_valid(query):
        # TODO debug it from Min, Max and Mean
        """
            Check if a query is valid
        :param query: Query string
        :return: boolean (True or False)
        """
        isMatch = re.search(MayoRegex.Valid, query, re.IGNORECASE)
        return bool(isMatch)

    @staticmethod
    def read_query_type(query):
        """
             read query type
        :param query: Query string
        :return: Query type
        """
        match = re.search(MayoRegex.QueryType, query)
        if not match:
            print('Not a valid query type: %s' % query)
            raise TypeError

        switcher = {"DC": QueryType.DOSE_COMPLIMENT,
                    "V": QueryType.VOLUME_AT_DOSE,
                    "D": QueryType.DOSE_AT_VOLUME,
                    "CV": QueryType.COMPLIMENT_VOLUME,
                    "Min": QueryType.MIN_DOSE,
                    "Max": QueryType.MAX_DOSE,
                    "Mean": QueryType.MEAN_DOSE}

        return switcher.get(match.group(), QueryType.VOLUME_AT_DOSE)

    def read_query_units(self, query):
        """
            read Query units
        :param query: Query string
        :return: Unit
        """
        filtered = re.sub(MayoRegex.UnitsDesired, '', query)
        match = re.search(MayoRegex.QueryUnits, filtered, re.IGNORECASE)
        if not match:
            return Units.NA
        return self.convert_string_to_unit(match.group())

    def read_units_desired(self, query):
        match = re.search(MayoRegex.UnitsDesired, query, re.IGNORECASE)
        if not match:
            print('Not valid units -> %s' % query)
            return TypeError
        return self.convert_string_to_unit(match.group().replace('[', '').replace(']', ''))

    @staticmethod
    def convert_string_to_unit(value):
        switcher = {"cc": Units.CC,
                    "CC": Units.CC,
                    "cGy": Units.CGY,
                    "cGY": Units.CGY,
                    "CGY": Units.CGY,
                    "cgy": Units.CGY,
                    "gy": Units.GY,
                    "Gy": Units.GY,
                    "GY": Units.GY,
                    "%": Units.PERC}
        return switcher.get(value, "Unknown query units!")


class MayoConstraint:
    def __init__(self):
        self._query = None
        self._discriminator = None
        self._constraintValue = None

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, value):
        self._query = value

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value

    @property
    def constraint_value(self):
        return self._constraintValue

    @constraint_value.setter
    def constraint_value(self, value):
        self._constraintValue = value

    def read(self, constraint):
        """
             Reads a constraint of the form {MayoQuery} {Discriminator} {ConstraintValue}
        :param constraint:
        """
        split = constraint.split()
        if len(split) != 3:
            raise ValueError(
                "Mayo constraints much be 3 parts separated by whitespace: "
                "{MayoQuery} {Discriminator} {ConstraintValue}")

        self._query = MayoQuery().read(split[0])
        self.discriminator = DiscriminatorConverter.read_discriminator(split[1])
        self._constraintValue = float(split[2])

    def write(self):
        query = MayoQueryWriter().write(self.query)
        discriminator = DiscriminatorConverter.write_discriminator(self.discriminator)
        constraint_value = str(self.constraint_value)
        return query + ' ' + discriminator + ' ' + constraint_value


class MayoConstraintConverter:
    def convert_to_dvh_constraint(self, structure_name, priority, mc):
        """
            Converts a Mayo constraint type to a DVH Constraint class
        :param structure_name: string structure_name
        :param priority:  PriorityType
        :param mc: class MayoConstraint
        :return: IConstraint
        """
        switch = {
            QueryType.MAX_DOSE:
                self.build_max_dose_constraint(mc, structure_name, priority),
            QueryType.MIN_DOSE:
                self.build_min_dose_constraint(mc, structure_name, priority),
            QueryType.MEAN_DOSE:
                self.build_mean_dose_constraint(mc, structure_name, priority),
            QueryType.DOSE_AT_VOLUME:
                self.build_dose_at_volume_constraint(mc, structure_name, priority),
            QueryType.VOLUME_AT_DOSE:
                self.build_volume_at_dose_constraint(mc, structure_name, priority),
            QueryType.DOSE_COMPLIMENT:
                self.build_dose_compliment_constraint(mc, structure_name, priority),
            QueryType.COMPLIMENT_VOLUME:
                self.build_compliment_volume_constraint(mc, structure_name, priority)
        }
        return switch.get(mc.query.query_type)

    @staticmethod
    def get_volume_units(mayo_unit):
        """

        :param mayo_unit: Units mayo_unit
        :return: VolumePresentation atribute
        """
        switcher = {Units.CC: VolumePresentation.absolute_cm3,
                    Units.PERC: VolumePresentation.relative}

        return switcher.get(mayo_unit, VolumePresentation.relative)

    @staticmethod
    def get_dose_units(mayo_unit):
        """
        :param mayo_unit: Units mayo_unit
        :return: DoseUnit
        """
        switcher = {Units.CGY: DoseUnit.cGy,
                    Units.GY: DoseUnit.Gy,
                    Units.PERC: DoseUnit.Percent}

        return switcher.get(mayo_unit, DoseUnit.Unknown)

    def build_max_dose_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: max_dose_constraint
        """
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)
        # constraint class
        c = MaxDoseConstraint()
        c.constraint_dose = dv
        c.structure_name = structure_name
        c.priority = priority

        return c

    def build_min_dose_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: min_dose_constraint
        """
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)
        # constraint class
        c = MinDoseConstraint()
        c.constraint_dose = dv
        c.structure_name = structure_name
        c.priority = priority
        return c

    def build_mean_dose_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: mean_dose_constraint
        """
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)
        # constraint classes
        min_mean = MinMeanDoseConstraint()
        min_mean.constraint_dose = dv
        min_mean.structure_name = structure_name
        min_mean.priority = priority

        # constraint classes
        max_mean = MaxMeanDoseConstraint()
        max_mean.constraint_dose = dv
        max_mean.structure_name = structure_name
        max_mean.priority = priority

        switch = {Discriminator.EQUAL: min_mean,
                  Discriminator.GREATER_THAN: min_mean,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_mean,
                  Discriminator.LESS_THAN: max_mean,
                  Discriminator.LESS_THAN_OR_EQUAL: max_mean
                  }

        return switch.get(mc.discriminator)

    def build_dose_at_volume_constraint(self, mc, structure_name, priority):
        """

        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: dose_at_volume_constraint
        """
        volume = mc.query.query_value
        volume_unit = self.get_dose_units(mc.query.query_units)
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_dv = MinDoseAtVolConstraint()
        min_dv.constraint_dose = dv
        min_dv.structure_name = structure_name
        min_dv.priority = priority
        min_dv.volume = volume
        min_dv.volume_type = volume_unit

        # constraint classes
        max_dv = MaxDoseAtVolConstraint()
        max_dv.constraint_dose = dv
        max_dv.structure_name = structure_name
        max_dv.priority = priority
        max_dv.volume = volume
        max_dv.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_dv,
                  Discriminator.GREATER_THAN: min_dv,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_dv,
                  Discriminator.LESS_THAN: max_dv,
                  Discriminator.LESS_THAN_OR_EQUAL: max_dv
                  }

        return switch.get(mc.discriminator)

    def build_volume_at_dose_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: volume_at_dose_constraint
        """
        volume = mc.query.query_value
        volume_unit = self.get_dose_units(mc.query.query_units)
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_vd = MinVolAtDoseConstraint()
        min_vd.constraint_dose = dv
        min_vd.structure_name = structure_name
        min_vd.priority = priority
        min_vd.volume = volume
        min_vd.volume_type = volume_unit

        # constraint classes
        max_vd = MaxVolAtDoseConstraint()
        max_vd.constraint_dose = dv
        max_vd.structure_name = structure_name
        max_vd.priority = priority
        max_vd.volume = volume
        max_vd.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_vd,
                  Discriminator.GREATER_THAN: min_vd,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_vd,
                  Discriminator.LESS_THAN: max_vd,
                  Discriminator.LESS_THAN_OR_EQUAL: max_vd
                  }

        return switch.get(mc.discriminator)

    def build_compliment_volume_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: compliment_volume_constraint
        """
        volume = mc.query.query_value
        volume_unit = self.get_dose_units(mc.query.query_units)
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_cv = MinComplimentVolumeAtDose()
        min_cv.constraint_dose = dv
        min_cv.structure_name = structure_name
        min_cv.priority = priority
        min_cv.volume = volume
        min_cv.volume_type = volume_unit

        # constraint classes
        max_cv = MaxComplimentVolumeAtDose()
        max_cv.constraint_dose = dv
        max_cv.structure_name = structure_name
        max_cv.priority = priority
        max_cv.volume = volume
        max_cv.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_cv,
                  Discriminator.GREATER_THAN: min_cv,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_cv,
                  Discriminator.LESS_THAN: max_cv,
                  Discriminator.LESS_THAN_OR_EQUAL: max_cv
                  }

        return switch.get(mc.discriminator)

    def build_dose_compliment_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: compliment_volume_constraint
        """
        volume = mc.query.query_value
        volume_unit = self.get_dose_units(mc.query.query_units)
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_dc = MinComplimentDoseAtVolumeConstraint()
        min_dc.constraint_dose = dv
        min_dc.structure_name = structure_name
        min_dc.priority = priority
        min_dc.volume = volume
        min_dc.volume_type = volume_unit

        # constraint classes
        max_dc = MaxComplimentDoseAtVolumeConstraint()
        max_dc.constraint_dose = dv
        max_dc.structure_name = structure_name
        max_dc.priority = priority
        max_dc.volume = volume
        max_dc.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_dc,
                  Discriminator.GREATER_THAN: min_dc,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_dc,
                  Discriminator.LESS_THAN: max_dc,
                  Discriminator.LESS_THAN_OR_EQUAL: max_dc
                  }

        return switch.get(mc.discriminator)


def test_MayoConstraint():
    """
        Test class MayoConstraint
    """

    constrain = 'D95%[cGy] > 7000'
    ctr = MayoConstraint()
    ctr.read(constrain)

    assert ctr.constraint_value == 7000.0
    assert ctr.discriminator == 2


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


if __name__ == '__main__':
    pass
