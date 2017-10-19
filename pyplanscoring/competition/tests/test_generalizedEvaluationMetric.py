from unittest import TestCase

import pandas as pd
from competition.tests import data_path, database_file, sheet
from pyplanscoring.competition.statistical_dvh import StatisticalDVH, GeneralizedEvaluationMetric, PlanningItemDVH

# set stats dvh

str_names = ['LENS LT',
             'PAROTID LT',
             'BRACHIAL PLEXUS',
             'OPTIC N. RT PRV',
             'OPTIC CHIASM PRV',
             'OPTIC N. RT',
             'ORAL CAVITY',
             'BRAINSTEM',
             'SPINAL CORD',
             'OPTIC CHIASM',
             'LENS RT',
             'LARYNX',
             'SPINAL CORD PRV',
             'EYE LT',
             'PTV56',
             'BRAINSTEM PRV',
             'PTV70',
             'OPTIC N. LT PRV',
             'EYE RT',
             'PTV63',
             'OPTIC N. LT',
             'LIPS',
             'ESOPHAGUS']

# global constraints data
df = pd.read_excel(data_path, sheetname=sheet)
stats_dvh = StatisticalDVH()
stats_dvh.load_data_from_hdf(database_file)


class TestGeneralizedEvaluationMetric(TestCase):
    def test_discrete_constraints(self):
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        ctr = gem.discrete_constraints
        pr = gem.priority_constraints
        pass

    def test_eval_constraints(self):
        # init PlanningItemDVH
        plan_dvh = stats_dvh.get_plan_dvh(0)
        pi = PlanningItemDVH(plan_dvh=plan_dvh)

        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.discrete_constraints = df
        res = gem.eval_constraints(pi)
        assert 'Result' in res

    def test_stats_constraints(self):
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.discrete_constraints = df
        constraints_stats = gem.calc_constraints_stats()
        # test save

    def test_save_constraints_stats(self):
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.discrete_constraints = df
        gem.save_constraints_stats(database_file, sheet)
        # how assert saving files?

    def test_load_constraints_stats(self):
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.discrete_constraints = df
        gem.load_constraints_stats(database_file, sheet)
        cst = gem.constraints_stats
        assert cst.any().any()

    def test_calc_gem(self):
        # select a plan to estimate GEM
        plan_dvh = stats_dvh.get_plan_dvh(0)
        pi = PlanningItemDVH(plan_dvh=plan_dvh)

        # instantiate GEM class
        gem = GeneralizedEvaluationMetric(stats_dvh, df)

        # try getting without load pre-calculated stats
        gem_0 = gem.calc_gem(pi)

        # First load constraint stats from HDF
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.load_constraints_stats(database_file, sheet)
        gem_1 = gem.calc_gem(pi)

        self.assertAlmostEqual(gem_1, gem_0)

        # todo DEBUG confidence interval
        # plan_values = gem.constraints_stats
        # import matplotlib.pyplot as plt
        # for i in range(len(plan_values)):
        #     plt.figure()
        #     plan_values.loc[i].hist()
        #     plt.title(gem.discrete_constraints['Structure Name'].loc[i])
        #

    def test_calculated_priority(self):
        import numpy as np
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        assert not gem.calculated_priority

        gem.load_constraints_stats(database_file, sheet)
        plan_values = gem.constraints_stats

        constraints_values = gem.constraints_values
        counts = plan_values <= constraints_values
        counts.sum(axis=1)
        ratio = counts.sum(axis=1) / counts.shape[1]
        calc_priority = np.round(1 - np.log2(ratio))
        np.testing.assert_array_almost_equal(calc_priority, gem.calculated_priority)

    def test_normalized_incomplete_gamma(self):
        val = GeneralizedEvaluationMetric.normalized_incomplete_gamma(1, 1)
        self.assertAlmostEqual(val, 0.63212055882855767)
        # TODO implement solve gamma equations
        self.fail()
        # test nonlinear solver
        # import scipy.special as sp
        #
        # from scipy.optimize import fsolve
        # import math
        #
        # def equations(p):
        #     x, y = p
        #     return x + y ** 2 - 4, math.exp(x) + x * y - 3
        #
        # x, y = fsolve(equations, (1, 1))
        #
        # g1 = lambda k, t, c: sp.gammainc(k, c / t) - 0.5
        # g2 = lambda k, t, c: sp.gammainc(k, c / t) - 0.95
        #
        # def gamma_equations(guess, constraint_value, upper_90_ci):
        #     k, t = guess
        #     return g1(k, t, constraint_value), g2(k, t, upper_90_ci)
        #
        # fsolve(gamma_equations, (7.5, 1), args=(100, 10))
