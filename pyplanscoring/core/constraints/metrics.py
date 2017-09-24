"""
Classes to DVH metrics

Author: Victor Alves

based on:
https://rexcardan.github.io/ESAPIX/

"""
from pyplanscoring.core.constraints.types import DoseValue, VolumePresentation


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
        :return: Volume
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
        :return: Volume
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
        :return: Volume
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
        :return: Volume
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
