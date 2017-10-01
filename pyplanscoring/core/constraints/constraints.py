"""
Classes to implement Mayo constraints

based on:

https://rexcardan.github.io/ESAPIX/api/ESAPIX.Constraints

Extended to use pydicom packages

"""
from pyplanscoring.core.constraints.query import MayoQuery, MayoQueryWriter
from pyplanscoring.core.constraints.types import ResultType, QueryType, Units, VolumePresentation, DoseUnit, DoseValue, \
    Discriminator


class IConstraint:
    def __init__(self, name='', full_name=''):
        self._name = name
        self._full_name = full_name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def constrain(self, pi):
        pass

    def can_constrain(self, pi):
        pass


class IPriorityConstraint(IConstraint):
    def __init__(self, name='', full_name=''):
        super().__init__(name, full_name)
        self._priority = None

    @property
    def priority(self):
        return self._priority

    @priority.setter
    def priority(self, value):
        self._priority = value

    @staticmethod
    def get_failed_result_type():
        pass


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
        return self._is_success

    @is_success.setter
    def is_success(self, value):
        self._is_success = value

    @property
    def result_type(self):
        return self._result_type

    @result_type.setter
    def result_type(self, value):
        self._result_type = value

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value):
        self._message = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class StructureNameConstraint(IConstraint):
    def __init__(self, name='', full_name=''):
        super().__init__(name, full_name)
        self._regex = ''
        self._structure_name = ''

    @property
    def regex(self):
        return self._regex

    @regex.setter
    def regex(self, value):
        self._regex = value

    @property
    def structure_name(self):
        return self._structure_name

    @structure_name.setter
    def structure_name(self, value):
        self._structure_name = value

    @property
    def name(self):
        return self.structure_name + ' required'

    @property
    def full_name(self):
        return self.name

    def can_constrain(self, pi):
        """

        :param pi: class PlanItem
        :return: ConstraintResult
        """
        message = ''
        # Check for null plan
        valid = pi is not {}
        if not valid:
            message = 'Plan is None'

        # Check structure exists
        valid = valid and pi.get_structures() != {}
        if not valid:
            message = "No structure set in {pi.Id}"

        return ConstraintResult(self, ResultType.NOT_APPLICABLE, message)

    def constrain(self, pi):

        msg = ''
        structure = pi.get_structure(self.structure_name, self.regex)
        passed = ResultType.ACTION_LEVEL_1

        if structure is not None:
            passed = ResultType.PASSED
            msg = '%s contains structure %s' % (pi.id, self.structure_name)

            if structure.volume_original < 0.0001:
                passed = ResultType.ACTION_LEVEL_1
                msg = "%s is empty" % self.structure_name

        return ConstraintResult(self, passed, msg)

    def __str__(self):
        return "Required Structure %s" % self.structure_name

    def __repr__(self):
        return self.__str__()


class TargetStats:
    def get_ci_rtog(self, structure, pi, reference_dose):
        """
            Calculates the RTOG conformity index as isodose volume irradiated at reference dose (Body contour volume irradiated)
        /// divided by the target volume. Does not necessarily mean the volumes are coincident!
        :param structure:
        :param pi:
        :param reference_dose:
        :return:
        """
        return NotImplementedError

    def get_ci_paddick(self, structure, pi, reference_dose):
        """
            Calculates the Paddick conformity index (PMID 11143252) as Paddick CI = (TVPIV)2 / (TV x PIV).
            TVPIV = Target volume covered by Prescription Isodose volume
            TV = Target volume

        :param structure:
        :param pi:
        :param reference_dose:
        :return:
        """
        return NotImplementedError

    def get_homogeneity_index(self):
        return NotImplementedError

    def get_voxel_homogeneity_index(self):
        return NotImplementedError


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
        constraint_value = ('%f' % self.constraint_value).rstrip('0').rstrip('.')
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

    @property
    def structure_name(self):
        return self._structure_name

    @structure_name.setter
    def structure_name(self, value):
        self._structure_name = value

    @property
    def constraint_dose(self):
        return DoseValue(self.dose, self.unit)

    @constraint_dose.setter
    def constraint_dose(self, dose_value):
        self.dose = dose_value.dose
        self.unit = dose_value.unit

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

    @property
    def constrain(self):
        """
        :return: ConstraintResult
        """
        return NotImplementedError

    @staticmethod
    def can_constrain():
        """
        :return: ConstraintResult
        """
        return NotImplementedError

    @staticmethod
    def get_failed_result_type():
        return NotImplementedError

    @staticmethod
    def get_merged_dvh():
        return NotImplementedError

    @staticmethod
    def get_structures():
        """
            Implement it from DICOM RS files
        :return:
        """
        return NotImplementedError


class MaxDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self):
        msg = ''
        passed = self.get_failed_result_type()
        # TODO implement it from DVH data
        return NotImplementedError

    def __str__(self):
        # Mayo format
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'Max[%s] <= %s' % (dose_unit, dose)

    def __repr__(self):
        return self.__str__()


class MinDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self):
        msg = ''
        passed = self.get_failed_result_type()
        # TODO implement it from DVH data
        return NotImplementedError

    def __str__(self):
        # Mayo format
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'Min[%s] <= %s' % (dose_unit, dose)

    def __repr__(self):
        return self.__str__()


class MinMeanDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self):
        msg = ''
        passed = self.get_failed_result_type()
        # TODO implement it from DVH data
        return NotImplementedError

    def __str__(self):
        # Mayo format
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'Mean[%s] >= %s' % (dose_unit, dose)

    def __repr__(self):
        return self.__str__()


class MaxMeanDoseConstraint(DoseStructureConstraint):
    def __init__(self):
        super().__init__()

    def constrain(self):
        msg = ''
        passed = self.get_failed_result_type()
        # TODO implement it from DVH data
        return NotImplementedError

    def __str__(self):
        # Mayo format
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'Mean[%s] <= %s' % (dose_unit, dose)

    def __repr__(self):
        return self.__str__()


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
        return self._volume_type

    @volume_type.setter
    def volume_type(self, value):
        self._volume_type = value

    @property
    def passing_func(self):
        return NotImplementedError

    @passing_func.setter
    def passing_func(self, value):
        pass

    def get_dose_at_volume(self):
        """
            Gets the dose at a volume for all structures in this constraint by merging their dvhs
            # TODO the planning item containing the dose to be queried
        :return:
        """
        return 'DoseAtVol'

    def constrain(self):
        # TODO constrain result
        return 'constrain result'


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

    @property
    def passing_func(self):
        return NotImplementedError

    @passing_func.setter
    def passing_func(self, value):
        pass

    def get_volume_at_dose(self):
        """
            Gets the dose at a volume for all structures in this constraint by merging their dvhs
            # TODO the planning item containing the dose to be queried
        :return:
        """
        return 'volume_at_dose'

    def constrain(self):
        # TODO constrain result
        return 'constrain result'


class MinDoseAtVolConstraint(DoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()

    @staticmethod
    def min_dose_at_vol_constraint():
        # TODO implement it using interpolation
        return NotImplementedError

    def constrain(self):
        msg = ''
        passed = self.get_failed_result_type()
        # TODO implement it from DVH data
        return NotImplementedError

    def __str__(self):
        # Mayo format
        vol = str(self.volume).replace(',', '')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'D%s%s[%s] >= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):
        return self.__str__()


class MaxDoseAtVolConstraint(DoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()

    @staticmethod
    def max_dose_at_vol_constraint():
        # TODO implement it using interpolation
        return NotImplementedError

    def constrain(self):
        msg = ''
        passed = self.get_failed_result_type()
        # TODO implement it from DVH data
        return NotImplementedError

    def __str__(self):
        # Mayo format
        vol = str(self.volume).replace(',', '')
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'D%s%s[%s] <= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):
        return self.__str__()


class MinVolAtDoseConstraint(VolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()
        # TODO implement lambda passing function

    def __str__(self):
        # Mayo format
        vol = str(self.volume)
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'V%s%s[%s] >= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):
        return self.__str__()


class MaxVolAtDoseConstraint(VolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()
        # TODO implement lambda passing function

    def __str__(self):
        # Mayo format
        vol = str(self.volume)
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'V%s%s[%s] <= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):
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

    @property
    def passing_func(self):
        return NotImplementedError

    @passing_func.setter
    def passing_func(self, value):
        pass

    def get_compliment_volume_at_dose(self):
        """
            Gets the dose at a volume for all structures in this constraint by merging their dvhs
            # TODO the planning item containing the dose to be queried
        :return:
        """
        return 'compliment_volume_at_dose'

    def constrain(self):
        # TODO constrain result
        return 'constrain result'


class MinComplimentVolumeAtDose(ComplimentVolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()
        # TODO implement lambda passing function

    def __str__(self):
        # Mayo format
        vol = str(self.volume).replace(",", "")
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'CV%s%s[%s] >= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):
        return self.__str__()


class MaxComplimentVolumeAtDose(ComplimentVolumeAtDoseConstraint):
    def __init__(self):
        super().__init__()
        # TODO implement lambda passing function

    def __str__(self):
        # Mayo format
        vol = str(self.volume).replace(",", "")
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'CV%s%s[%s] <= %s' % (dose, dose_unit, vol_unit, vol)

    def __repr__(self):
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

    @property
    def passing_func(self):
        return NotImplementedError

    @passing_func.setter
    def passing_func(self, value):
        pass

    def get_dose_compliment_at_volume(self):
        """
            Gets the dose at a volume for all structures in this constraint by merging their dvhs
            # TODO the planning item containing the dose to be queried
        :return:
        """
        return 'compliment_volume_at_dose'

    def constrain(self):
        # TODO constrain result
        return 'constrain result'


class MinComplimentDoseAtVolumeConstraint(ComplimentDoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()
        # TODO implement lambda passing function

    def __str__(self):
        # Mayo format
        vol = str(self.volume).replace(",", "")
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'DC%s%s[%s] >= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):
        return self.__str__()


class MaxComplimentDoseAtVolumeConstraint(ComplimentDoseAtVolumeConstraint):
    def __init__(self):
        super().__init__()
        # TODO implement lambda passing function

    def __str__(self):
        # Mayo format
        vol = str(self.volume).replace(",", "")
        vol_unit = 'cc' if self.volume_type == VolumePresentation.absolute_cm3 else '%'
        dose_unit = str(self.constraint_dose.unit)
        dose = str(self.constraint_dose.value)
        return 'DC%s%s[%s] <= %s' % (vol, vol_unit, dose_unit, dose)

    def __repr__(self):
        return self.__str__()


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
