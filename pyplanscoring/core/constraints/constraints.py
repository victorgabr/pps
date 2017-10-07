"""
Classes to implement Mayo constraints

based on:

https://rexcardan.github.io/ESAPIX/api/ESAPIX.Constraints

Extended to use pydicom packages

"""
from pyplanscoring.core.constraints.query import MayoQueryWriter, QueryExtensions
from pyplanscoring.core.constraints.types import ResultType, QueryType, Units, VolumePresentation, DoseUnit, DoseValue, \
    Discriminator, PriorityType


class DoseStructureConstraint:
    def __init__(self):
        """
            abstract class DoseStructureConstraint

            The string structure name corresponding to the structure ID in Eclipse.
            Separate with '&' character for multiple structures
        """
        self._structure_name = None
        self._constraint_dose = None
        self._dose = None
        self._unit = None
        self._priority = None
        self._volume = None
        self._volume_type = None

    @property
    def volume(self):
        """
        :return: volume
        """
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def volume_type(self):
        """
        :return: VolumePresentation
        """
        return self._volume_type

    @volume_type.setter
    def volume_type(self, value):
        self._volume_type = value

    @property
    def structure_name(self):
        return self._structure_name

    @structure_name.setter
    def structure_name(self, value):
        self._structure_name = value

    @property
    def dose(self):
        """
             The dose value component of the constraint dose - Used for text storage
        :return:  dose
        """
        return self._dose

    @dose.setter
    def dose(self, value):
        self._dose = value

    @property
    def unit(self):
        """
            The dose unit component of the constraint dose - Used for text storage
        :return: unit
        """
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value

    @property
    def constraint_dose(self):
        return DoseValue(self.dose, self.unit)

    @constraint_dose.setter
    def constraint_dose(self, dose_value):
        self.dose = dose_value.value
        self.unit = dose_value.unit

    @property
    def structure_names(self):
        return self.structure_name.split('&')

    @property
    def name(self):
        return self.structure_name

    @property
    def priority(self):
        return self._priority

    @priority.setter
    def priority(self, value):
        self._priority = value

    def constrain(self, pi):
        """
        :return: ConstraintResult
        """
        return NotImplementedError

    def can_constrain(self, pi):
        """
        :return: ConstraintResult
        """
        if not pi.plan:
            return ConstraintResult(self, ResultType.NOT_APPLICABLE, "Plan is None")

        # Check structure exists
        structures = self.structure_name.split('&')
        for s in structures:
            valid = pi.contains_structure(s)
            if not valid:
                message = '%s is not contoured in plan'
                return ConstraintResult(self, ResultType.NOT_APPLICABLE_MISSING_STRUCTURE, message)

        # Check dose is calculated
        if not pi.dose_data:
            message = 'There is no dose calculated - DICOM-RD'
            return ConstraintResult(self, ResultType.NOT_APPLICABLE_MISSING_DOSE, message)

        return ConstraintResult(self, ResultType.PASSED, '')

    def get_failed_result_type(self):
        switch = {PriorityType.MAJOR_DEVIATION: ResultType.ACTION_LEVEL_3,
                  PriorityType.PRIORITY_1: ResultType.ACTION_LEVEL_3}

        return switch.get(self.priority, ResultType.ACTION_LEVEL_1)

    def get_structures(self, pi):
        """
            Implement it from DICOM RS files
        :return:
        """
        return pi.get_structure(self.structure_name)


class ConstraintResult:
    def __init__(self, constraint, result_type, message, value=''):
        """
            Encapsulates the results from an attempt to constrain a planning item
        :param constraint: IConstraint
        :param result_type: ResultType
        :param message: message
        :param value:
        """
        self._constraint = constraint
        self._is_success = result_type == ResultType.PASSED
        self._result_type = result_type
        self._is_applicable = result_type != ResultType.NOT_APPLICABLE \
                              and result_type != ResultType.NOT_APPLICABLE_MISSING_STRUCTURE \
                              and result_type != ResultType.NOT_APPLICABLE_MISSING_DOSE
        self._message = message
        self._value = value

    @property
    def constraint(self):
        return self._constraint

    @constraint.setter
    def constraint(self, value):
        self._constraint = value

    @property
    def is_success(self):
        """
             Signifies if constraint passed.
        :return: bool
        """
        return self._is_success

    @is_success.setter
    def is_success(self, value):
        self._is_success = value

    @property
    def is_applicable(self):
        """
            Signifies if constraint was applicable to current plan.
        :return: bool
        """
        return self._is_applicable

    @is_applicable.setter
    def is_applicable(self, value):
        self.is_applicable = value

    @property
    def result_type(self):
        """
             The result value including action level for the constraint
        :return:
        """
        return self._result_type

    @result_type.setter
    def result_type(self, value):
        self._result_type = value

    @property
    def message(self):
        """
            The message indicating why a test failed
        :return: string
        """
        return self._message

    @message.setter
    def message(self, value):
        self._message = value

    @property
    def value(self):
        """
            The message indicating why a test failed
        :return: string
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class DoseAtVolumeConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()
        self._volume = None
        self._volume_type = None

    @property
    def volume(self):
        """
        :return: volume
        """
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def volume_type(self):
        """
        :return: VolumePresentation
        """
        return self._volume_type

    @volume_type.setter
    def volume_type(self, value):
        self._volume_type = value

    def passing_func(self, dose_at_vol):
        return NotImplementedError

    def get_dose_at_volume(self, pi):
        """
            Gets the dose at a volume for a structure
        :param pi: PlaningItem class
        :return: DoseValue
        """
        d_pres = self.constraint_dose.get_presentation()
        v_pres = self.volume_type
        dose_at_vol = pi.get_dose_at_volume(self.structure_name,
                                            self.volume,
                                            v_pres,
                                            d_pres)
        return dose_at_vol

    def constrain(self, pi):
        dose_at_vol = self.get_dose_at_volume(pi)
        passed = self.passing_func(dose_at_vol)
        string_unit = self.volume_type.symbol
        dose_value = dose_at_vol.get_dose(self.constraint_dose.unit)
        msg = 'Dose to %1.3f %s of %s is %s' % (self.volume, string_unit, self.structure_name, str(dose_value))
        return ConstraintResult(self, passed, msg, str(dose_value))


class MaxDoseAtVolConstraint(DoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, dose_at_vol):
        return ResultType.PASSED if dose_at_vol <= self.constraint_dose else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'D%s%s[%s] <= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MinDoseAtVolConstraint(DoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, dose_at_vol):
        return ResultType.PASSED if dose_at_vol >= self.constraint_dose else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%1.3f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'D%s%s[%s] >= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class ComplimentDoseAtVolumeConstraint(DoseStructureConstraint):
    def __init__(self):
        """
            Encapsulates the dose compliment (cold spot) of a structure. Dose compliment at 2% will give the maximum dose
                 in the coldest 2 % of a structure.
        """
        super().__init__()
        self._volume = None
        self._volume_type = None

    @property
    def volume(self):
        """
        :return: volume
        """
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def volume_type(self):
        return self._volume_type

    @volume_type.setter
    def volume_type(self, value):
        self._volume_type = value

    def passing_func(self, dc_at_vol):
        return NotImplementedError

    def get_dose_compliment_at_volume(self, pi):
        """
            Gets the dose compliment at a volume for a structure
        :param pi: PlaningItem class
        :return: DoseValue
        """
        d_pres = self.constraint_dose.get_presentation()
        v_pres = self.volume_type
        dc_at_vol = pi.get_dose_compliment_at_volume(self.structure_name,
                                                     self.volume,
                                                     v_pres,
                                                     d_pres)
        return dc_at_vol

    def constrain(self, pi):
        dc_at_vol = self.get_dose_compliment_at_volume(pi)
        passed = self.passing_func(dc_at_vol)
        string_unit = self.volume_type.symbol
        dose_value = dc_at_vol.get_dose(self.constraint_dose.unit)
        msg = 'Dose compliment to %1.3f %s of %s is %s' % (self.volume,
                                                           string_unit,
                                                           self.structure_name,
                                                           str(dose_value))
        return ConstraintResult(self, passed, msg, str(dose_value))


class MaxComplimentDoseAtVolumeConstraint(ComplimentDoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, dc_at_vol):
        return ResultType.PASSED if dc_at_vol <= self.constraint_dose else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%1.3f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'DC%s%s[%s] <= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MinComplimentDoseAtVolumeConstraint(ComplimentDoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, dc_at_vol):
        return ResultType.PASSED if dc_at_vol >= self.constraint_dose else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'DC%s%s[%s] >= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class VolumeAtDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()
        self._volume = None
        self._volume_type = None

    @property
    def volume(self):
        """
        :return: volume
        """
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def volume_type(self):
        return self._volume_type

    @volume_type.setter
    def volume_type(self, value):
        self._volume_type = value

    def passing_func(self, vol):
        return NotImplementedError

    def get_volume_at_dose(self, pi):
        """
           Get volume at dose constrain
        :param pi: PlaningItem class
        :return: Volume
        """
        v_pres = self.volume_type
        volume_at_dose = pi.get_volume_at_dose(self.structure_name,
                                               self.constraint_dose,
                                               v_pres)

        return volume_at_dose

    def constrain(self, pi):
        volume_at_dose = self.get_volume_at_dose(pi)
        passed = self.passing_func(volume_at_dose)
        # string_unit = self.volume_type.symbol
        msg = 'Volume of %s at %s was %s.' % (self.structure_name,
                                              str(self.constraint_dose),
                                              str(volume_at_dose))
        return ConstraintResult(self, passed, msg, str(volume_at_dose))


class MinVolAtDoseConstraint(VolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, vol):
        return ResultType.PASSED if vol >= self.volume else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%1.3f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'V%s%s[%s] >= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MaxVolAtDoseConstraint(VolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, vol):
        return ResultType.PASSED if vol <= self.volume else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%1.3f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'V%s%s[%s] <= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class ComplimentVolumeAtDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()
        self._volume = None
        self._volume_type = None

    @property
    def volume(self):
        """
        :return: volume
        """
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def volume_type(self):
        return self._volume_type

    @volume_type.setter
    def volume_type(self, value):
        self._volume_type = value

    def passing_func(self, cv):
        return NotImplementedError

    def get_compliment_volume_at_dose(self, pi):
        """
            Gets the dose at a volume for all structures in this constraint by merging their dvhs
            # TODO the planning item containing the dose to be queried
        :return:
        """
        v_pres = self.volume_type
        cv_at_dose = pi.get_compliment_volume_at_dose(self.structure_name,
                                                      self.constraint_dose,
                                                      v_pres)

        return cv_at_dose

    def constrain(self, pi):
        cv_at_dose = self.get_compliment_volume_at_dose(pi)
        passed = self.passing_func(cv_at_dose)
        # string_unit = self.volume_type.symbol
        msg = 'Compliment volume of %s at %s was %s.' % (self.structure_name,
                                                         str(self.constraint_dose),
                                                         str(cv_at_dose))
        return ConstraintResult(self, passed, msg, str(cv_at_dose))


class MinComplimentVolumeAtDose(ComplimentVolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, volume):
        return ResultType.PASSED if volume >= self.volume else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%1.3f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'CV%s%s[%s] >= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MaxComplimentVolumeAtDose(ComplimentVolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()

    def passing_func(self, vol):
        return ResultType.PASSED if vol <= self.volume else self.get_failed_result_type()

    def __str__(self):
        # Mayo format
        vol = ('%f' % self.volume).rstrip('0').rstrip('.')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'CV%s%s[%s] <= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MaxDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self, pi):
        d_pres = self.constraint_dose.get_presentation()
        v_pres = self.volume_type
        dvh = pi.get_dvh_cumulative_data(self.structure_name, d_pres, v_pres)
        value = str(dvh.max_dose)
        passed = ResultType.PASSED if dvh.max_dose <= self.constraint_dose else self.get_failed_result_type()
        msg = 'Maximum dose to %s is %s.' % (self.structure_name, value)

        return ConstraintResult(self, passed, msg, value)

    def __str__(self):
        # Mayo format
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose_str = ('%f' % dose).rstrip('0').rstrip('.')
        return 'Max[%s] <= %s' % (dose_unit, dose_str)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MinDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self, pi):
        d_pres = self.constraint_dose.get_presentation()
        v_pres = self.volume_type
        dvh = pi.get_dvh_cumulative_data(self.structure_name, d_pres, v_pres)
        value = str(dvh.min_dose)
        passed = ResultType.PASSED if dvh.min_dose >= self.constraint_dose else self.get_failed_result_type()
        msg = 'Minimum dose to %s is %s.' % (self.structure_name, value)

        return ConstraintResult(self, passed, msg, value)

    def __str__(self):
        # Mayo format
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose_str = ('%f' % dose).rstrip('0').rstrip('.')
        return 'Min[%s] >= %s' % (dose_unit, dose_str)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MinMeanDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self, pi):
        d_pres = self.constraint_dose.get_presentation()
        v_pres = self.volume_type
        dvh = pi.get_dvh_cumulative_data(self.structure_name, d_pres, v_pres)
        value = str(dvh.mean_dose)
        passed = ResultType.PASSED if dvh.mean_dose >= self.constraint_dose else self.get_failed_result_type()
        msg = 'Mean dose to %s is %s.' % (self.structure_name, value)

        return ConstraintResult(self, passed, msg, value)

    def __str__(self):
        # Mayo format
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'Mean[%s] >= %s' % (dose_unit, dose)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MaxMeanDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self, pi):
        d_pres = self.constraint_dose.get_presentation()
        v_pres = self.volume_type
        dvh = pi.get_dvh_cumulative_data(self.structure_name, d_pres, v_pres)
        value = str(dvh.mean_dose)
        passed = ResultType.PASSED if dvh.mean_dose <= self.constraint_dose else self.get_failed_result_type()
        msg = 'Mean dose to %s is %s.' % (self.structure_name, value)

        return ConstraintResult(self, passed, msg, value)

    def __str__(self):
        # Mayo format
        dose_unit = self.constraint_dose.unit.symbol
        dose = self.constraint_dose.value
        dose = ('%f' % dose).rstrip('0').rstrip('.')
        return 'Mean[%s] <= %s' % (dose_unit, dose)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


# TODO implement Conformation and HI constraints

class ConformationIndexConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()
        self._mc = None
        self._constraint_value = None

    @property
    def constraint_value(self):
        return self._constraint_value

    @constraint_value.setter
    def constraint_value(self, value):
        self._constraint_value = value

    @property
    def mc(self):
        return self._mc

    @mc.setter
    def mc(self, value):
        self._mc = value

    def constrain(self, pi):
        dm = pi.execute_query(str(self.mc.query), self.structure_name)
        value = str(dm)
        passed = ResultType.PASSED if dm >= self.constraint_value else self.get_failed_result_type()
        msg = '%s to %s is %s.' % (str(self.mc.query), self.structure_name, value)

        return ConstraintResult(self, passed, msg, value)

    def __str__(self):
        txt = '%s <= %s' % (str(self.mc.query), str(self.mc.constraint_value))
        return txt

    def __repr__(self):
        return self.__str__()


class HomogeneityIndexConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()
        self._mc = None
        self._constraint_value = None

    @property
    def constraint_value(self):
        return self._constraint_value

    @constraint_value.setter
    def constraint_value(self, value):
        self._constraint_value = value

    @property
    def mc(self):
        return self._mc

    @mc.setter
    def mc(self, value):
        self._mc = value

    def constrain(self, pi):
        dm = pi.execute_query(str(self.mc.query), self.structure_name)
        value = str(dm)
        passed = ResultType.PASSED if dm <= self.constraint_value else self.get_failed_result_type()
        msg = '%s to %s is %s.' % (str(self.mc.query), self.structure_name, value)

        return ConstraintResult(self, passed, msg, value)

    def __str__(self):
        txt = '%s <= %s' % (str(self.mc.query), str(self.mc.constraint_value))
        return txt

    def __repr__(self):
        return self.__str__()


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

        self._query = QueryExtensions().read(split[0])
        self.discriminator = DiscriminatorConverter.read_discriminator(split[1])
        self._constraintValue = float(split[2])

        return self

    def write(self):
        query = MayoQueryWriter().write(self.query)
        discriminator = DiscriminatorConverter.write_discriminator(self.discriminator)
        constraint_value = ('%f' % self.constraint_value).rstrip('0').rstrip('.')
        return query + ' ' + discriminator + ' ' + constraint_value


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


class MayoConstraintConverter:
    def convert_to_dvh_constraint(self, structure_name, priority, mc):
        """
            Converts a Mayo constraint type to a DVH Constraint class
        :param structure_name: string structure_name
        :param priority:  PriorityType
        :param mc: class MayoConstraint
        :return: IConstraint
        """
        # if constraint query is string, so read it int MayoConstraint Object
        if isinstance(mc, str):
            mc = MayoConstraint().read(mc)

        switch = {QueryType.MAX_DOSE: self.build_max_dose_constraint,
                  QueryType.MIN_DOSE: self.build_min_dose_constraint,
                  QueryType.MEAN_DOSE: self.build_mean_dose_constraint,
                  QueryType.DOSE_AT_VOLUME: self.build_dose_at_volume_constraint,
                  QueryType.VOLUME_AT_DOSE: self.build_volume_at_dose_constraint,
                  QueryType.DOSE_COMPLIMENT: self.build_dose_compliment_constraint,
                  QueryType.COMPLIMENT_VOLUME: self.build_compliment_volume_constraint,
                  QueryType.CI: self.build_ci_constraint,
                  QueryType.HI: self.build_hi_constraint}

        build_function = switch.get(mc.query.query_type)

        return build_function(mc, structure_name, priority)

    @staticmethod
    def get_volume_units(mayo_unit):
        """

        :param mayo_unit: Units mayo_unit
        :return: VolumePresentation atribute
        """
        switcher = {Units.CC: VolumePresentation.absolute_cm3,
                    Units.PERC: VolumePresentation.relative}

        return switcher.get(mayo_unit, VolumePresentation.Unknown)

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
                  Discriminator.LESS_THAN_OR_EQUAL: max_mean}

        return switch.get(mc.discriminator)

    def build_dose_at_volume_constraint(self, mc, structure_name, priority):
        """

        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: dose_at_volume_constraint
        """
        volume = mc.query.query_value
        volume_unit = self.get_volume_units(mc.query.query_units)
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_dv = MinDoseAtVolConstraint()
        min_dv.constraint_dose = dv
        min_dv.structure_name = structure_name
        min_dv.priority = priority
        min_dv.volume = volume * volume_unit
        min_dv.volume_type = volume_unit

        # constraint classes
        max_dv = MaxDoseAtVolConstraint()
        max_dv.constraint_dose = dv
        max_dv.structure_name = structure_name
        max_dv.priority = priority
        max_dv.volume = volume * volume_unit
        max_dv.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_dv,
                  Discriminator.GREATER_THAN: min_dv,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_dv,
                  Discriminator.LESS_THAN: max_dv,
                  Discriminator.LESS_THAN_OR_EQUAL: max_dv}

        return switch.get(mc.discriminator)

    def build_volume_at_dose_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: volume_at_dose_constraint
        """
        volume = mc.constraint_value
        volume_unit = self.get_volume_units(mc.query.units_desired)
        dose_unit = self.get_dose_units(mc.query.query_units)
        dose = mc.query.query_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_vd = MinVolAtDoseConstraint()
        min_vd.constraint_dose = dv
        min_vd.structure_name = structure_name
        min_vd.priority = priority
        min_vd.volume = volume * volume_unit
        min_vd.volume_type = volume_unit

        # constraint classes
        max_vd = MaxVolAtDoseConstraint()
        max_vd.constraint_dose = dv
        max_vd.structure_name = structure_name
        max_vd.priority = priority
        max_vd.volume = volume * volume_unit
        max_vd.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_vd,
                  Discriminator.GREATER_THAN: min_vd,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_vd,
                  Discriminator.LESS_THAN: max_vd,
                  Discriminator.LESS_THAN_OR_EQUAL: max_vd}

        return switch.get(mc.discriminator)

    def build_compliment_volume_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: compliment_volume_constraint
        """
        volume = mc.constraint_value
        volume_unit = self.get_volume_units(mc.query.units_desired)
        dose_unit = self.get_dose_units(mc.query.query_units)
        dose = mc.query.query_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_cv = MinComplimentVolumeAtDose()
        min_cv.constraint_dose = dv
        min_cv.structure_name = structure_name
        min_cv.priority = priority
        min_cv.volume = volume * volume_unit
        min_cv.volume_type = volume_unit

        # constraint classes
        max_cv = MaxComplimentVolumeAtDose()
        max_cv.constraint_dose = dv
        max_cv.structure_name = structure_name
        max_cv.priority = priority
        max_cv.volume = volume * volume_unit
        max_cv.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_cv,
                  Discriminator.GREATER_THAN: min_cv,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_cv,
                  Discriminator.LESS_THAN: max_cv,
                  Discriminator.LESS_THAN_OR_EQUAL: max_cv}

        return switch.get(mc.discriminator)

    def build_dose_compliment_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: compliment_volume_constraint
        """
        volume = mc.query.query_value
        volume_unit = self.get_volume_units(mc.query.query_units)
        dose_unit = self.get_dose_units(mc.query.units_desired)
        dose = mc.constraint_value
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        min_dc = MinComplimentDoseAtVolumeConstraint()
        min_dc.constraint_dose = dv
        min_dc.structure_name = structure_name
        min_dc.priority = priority
        min_dc.volume = volume * volume_unit
        min_dc.volume_type = volume_unit

        # constraint classes
        max_dc = MaxComplimentDoseAtVolumeConstraint()
        max_dc.constraint_dose = dv
        max_dc.structure_name = structure_name
        max_dc.priority = priority
        max_dc.volume = volume * volume_unit
        max_dc.volume_type = volume_unit

        switch = {Discriminator.EQUAL: min_dc,
                  Discriminator.GREATER_THAN: min_dc,
                  Discriminator.GREATHER_THAN_OR_EQUAL: min_dc,
                  Discriminator.LESS_THAN: max_dc,
                  Discriminator.LESS_THAN_OR_EQUAL: max_dc}

        return switch.get(mc.discriminator)

    def build_hi_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: Homogeneity index constraint
        """
        dose = mc.query.query_value
        dose_unit = self.get_dose_units(mc.query.query_units)
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        hi = HomogeneityIndexConstraint()
        hi.constraint_dose = dv
        hi.constraint_value = mc.constraint_value
        hi.structure_name = structure_name
        hi.priority = priority
        hi.mc = mc

        return hi

    def build_ci_constraint(self, mc, structure_name, priority):
        """
        :param mc: MayoConstraint
        :param structure_name: string
        :param priority: PriorityType
        :return: Paddick conformality index constraint
        """
        dose = mc.query.query_value
        dose_unit = self.get_dose_units(mc.query.query_units)
        dv = DoseValue(dose, dose_unit)

        # constraint classes
        ci = ConformationIndexConstraint()
        ci.constraint_dose = dv
        ci.constraint_value = mc.constraint_value
        ci.structure_name = structure_name
        ci.priority = priority
        ci.mc = mc

        return ci

# class StructureNameConstraint:
#     def __init__(self):
#         super().__init__()
#         self._regex = ''
#         self._structure_name = ''
#
#     @property
#     def regex(self):
#         return self._regex
#
#     @regex.setter
#     def regex(self, value):
#         self._regex = value
#
#     @property
#     def structure_name(self):
#         return self._structure_name
#
#     @structure_name.setter
#     def structure_name(self, value):
#         self._structure_name = value
#
#     @property
#     def name(self):
#         return self.structure_name + ' required'
#
#     @property
#     def full_name(self):
#         return self.name
#
#     def can_constrain(self, pi):
#         """
#
#         :param pi: class PlanItem
#         :return: ConstraintResult
#         """
#         message = ''
#         valid = True
#         if not pi.plan:
#             message = 'Plan is None'
#             valid = False
#
#         # Check structure exists
#         valid = valid and pi.get_structures() != {}
#         if not valid:
#             message = "No structure set in {pi.Id}"
#
#         return ConstraintResult(self, ResultType.NOT_APPLICABLE, message)
#
#     def constrain(self, pi):
#
#         msg = ''
#         structure = pi.get_structure(self.structure_name, self.regex)
#         passed = ResultType.ACTION_LEVEL_1
#
#         if structure is not None:
#             passed = ResultType.PASSED
#             msg = 'contains structure %s' % self.structure_name
#
#             if structure.volume_original < 0.0001:
#                 passed = ResultType.ACTION_LEVEL_1
#                 msg = "%s is empty" % self.structure_name
#
#         return ConstraintResult(self, passed, msg)
#
#     def __str__(self):
#         return "Required Structure %s" % self.structure_name
#
#     def __repr__(self):  # pragma: no cover
#         return self.__str__()
