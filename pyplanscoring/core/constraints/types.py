"""
Classes to enumerate DVH types
based on:
https://rexcardan.github.io/ESAPIX/
"""


class QueryType:
    VOLUME_AT_DOSE = 0
    COMPLIMENT_VOLUME = 1
    DOSE_AT_VOLUME = 2
    DOSE_COMPLIMENT = 3
    MEAN_DOSE = 4
    MIN_DOSE = 5
    MAX_DOSE = 6


class Units:
    CC = 0
    PERC = 1
    GY = 2
    CGY = 3
    NA = 4


class DoseUnit:
    Unknown = 0
    Gy = 1
    cGy = 2
    Percent = 3


class DoseValuePresentation:
    Relative = 0
    Absolute = 1


class Discriminator:
    LESS_THAN = 0
    LESS_THAN_OR_EQUAL = 1
    GREATER_THAN = 2
    GREATHER_THAN_OR_EQUAL = 3
    EQUAL = 4


class VolumePresentation:
    relative = 0
    absolute_cm3 = 1


class PriorityType:
    IDEAL = 0
    ACCEPTABLE = 1
    MINOR_DEVIATION = 2
    MAJOR_DEVIATION = 3
    GOAL = 4
    OPTIONAL = 5
    REPORT = 6
    PRIORITY_1 = 7
    PRIORITY_2 = 8


class ResultType:
    PASSED = 0
    ACTION_LEVEL_1 = 1
    ACTION_LEVEL_2 = 2
    ACTION_LEVEL_3 = 3
    NOT_APPLICABLE = 4
    NOT_APPLICABLE_MISSING_STRUCTURE = 5
    NOT_APPLICABLE_MISSING_DOSE = 6
    INCONCLUSIVE = 7


class DICOMType:
    """
    Class that holds constant strings from the Eclipse Scripting API
    """
    PTV = "PTV"
    GTV = "GTV"
    CTV = "CTV"
    DOSE_REGION = "DOSE_REGION"
    NONE = ""
    CONSTRAST_AGENT = "CONSTRAST_AGENT"
    CAVITY = "CAVITY"
    AVOIDANCE = "AVOIDANCE"
    CONTROL = "CONTROL"
    FIXATION = "FIXATION"
    IRRAD_VOLUME = "IRRAD_VOLUME"
    ORGAN = "ORGAN"
    TREATED_VOLUME = "TREATED_VOLUME"
    EXTERNAL = "EXTERNAL"


class DoseValue:
    def __init__(self, dose_value, unit):
        self._value = dose_value
        self._unit = unit
        self._dose = dose_value

    @property
    def dose(self):
        return self._dose

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value


class DiscriminatorConverter:
    @staticmethod
    def read_discriminator(disc_string):
        """
         Reads a discriminator from a string
        :param disc_string: the string discriminator (eg <, <=, etc)

        """
        switcher = {
            "<=": Discriminator.LESS_THAN_OR_EQUAL,
            "<": Discriminator.LESS_THAN,
            ">=": Discriminator.GREATHER_THAN_OR_EQUAL,
            ">": Discriminator.GREATER_THAN,
            "=": Discriminator.EQUAL
        }

        return switcher.get(disc_string, "Not a valid comparitor (eg >=, =, <=...)")

    @staticmethod
    def write_discriminator(disc):
        """
             Writes a discriminator to a string
        :param disc: Discriminator
        :return:
        """
        switcher = {
            Discriminator.EQUAL: "=",
            Discriminator.LESS_THAN_OR_EQUAL: "<=",
            Discriminator.LESS_THAN: "<",
            Discriminator.GREATHER_THAN_OR_EQUAL: ">=",
            Discriminator.GREATER_THAN: ">",
        }

        return switcher.get(disc, "Not a valid discriminator!")


class TargetStat:
    CONFORMITY_INDEX_RTOG = 0
    CONFORMITY_INDEX_PADDICK = 1
    HOMOGENEITY_INDEX = 2
    VOXEL_BASED_HOMOGENEITY_INDEX = 3
