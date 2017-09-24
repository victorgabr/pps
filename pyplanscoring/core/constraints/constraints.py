"""
Classes to implement Mayo constraints

based on:

https://rexcardan.github.io/ESAPIX/api/ESAPIX.Constraints

"""
from pyplanscoring.core.constraints.types import ResultType


class PlanningItem:
    def __init__(self, id=''):
        self.id = id

    def get_structure(self, structure_name, regex):
        pass

    def get_structures(self):
        pass


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
    """
        /// <summary>
        /// Calculates the RTOG conformity index as isodose volume irradiated at reference dose (Body contour volume irradiated)
        /// divided by the target volume. Does not necessarily mean the volumes are coincident!
        /// </summary>
        /// <param name="s">the target structure</param>
        /// <param name="pi">the planning item containing the dose</param>
        /// <param name="referenceDose">the reference isodose (eg. prescription dose)</param>
        /// <returns>RTOG conformity index</returns>
    """

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
            TVPIV = Target Volume covered by Prescription Isodose Volume
            TV = Target Volume

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
