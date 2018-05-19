import re
from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd

from competition.tests import data_path, database_file, sheet, low_score, high_score, root_folder
from pyplanscoring.core.dvhcalculation import load
from pyplanscoring.competition.statistical_dvh import StatisticalDVH, GeneralizedEvaluationMetric, PlanningItemDVH, \
    HistoricPlanDVH, PopulationBasedGEM

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
             'ESOPHAGUS',
             'PTV70']

# global constraints data
df = pd.read_excel(data_path, sheetname=sheet)
stats_dvh = StatisticalDVH()
stats_dvh.load_data_from_hdf(database_file)
hist_data = HistoricPlanDVH(root_folder)
hist_data.set_participant_folder()


# calculate STATS for all plans

class TestGeneralizedEvaluationMetric(TestCase):
    def test_discrete_constraints(self):
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        ctr = gem.discrete_constraints
        pr = gem.priority_constraints
        pass

    def test_eval_constraints(self):
        # init PlanningItemDVH
        plan_dvh = stats_dvh.get_plan_dvh(16)
        pi = PlanningItemDVH(plan_dvh=plan_dvh)

        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.discrete_constraints = df
        res = gem.eval_constraints(pi)

        # init PlanningItemDVH
        plan_dvh = stats_dvh.get_plan_dvh(0)
        pi = PlanningItemDVH(plan_dvh=plan_dvh)

        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.discrete_constraints = df
        res = gem.eval_constraints(pi)

        # debug error in plan 16

        assert 'Result' in res

    def test_stats_constraints(self):
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.discrete_constraints = df
        constraints_stats = gem.calc_constraints_stats()
        # test save
        pass

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

        # test method overloading
        gem_2 = gem.calc_gem(0)
        self.assertAlmostEqual(gem_0, gem_2)

        # test low score plan
        pi = PlanningItemDVH(plan_dvh=low_score)
        low_gem = gem.calc_gem(pi)

        # test high score plan
        pi = PlanningItemDVH(plan_dvh=high_score)
        hig_gem = gem.calc_gem(pi)

        assert low_gem > hig_gem

    def calc_stats_gem(self):
        # First load constraint stats from HDF
        df = pd.read_excel(data_path, sheetname=sheet)
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.load_constraints_stats(database_file, sheet)

        # TODO evaluate new ranking against classic scores
        gem_plans = []
        for part in hist_data.map_part:
            dvh = load(part[1][0])['DVH']
            pi_t = PlanningItemDVH(plan_dvh=dvh)
            gem_t = gem.calc_gem(pi_t)
            if gem_t:
                gem_plans.append([part[0], gem_t])
        df = pd.DataFrame(gem_plans)
        df['sc'] = df[0].apply(lambda row: re.findall("\d+\.\d+", row)[0] if re.findall("\d+\.\d+", row) else None)

        df = df.dropna()
        plt.plot(df['sc'], df[1], '.')

        # load winner DVH
        winner_df = df.sort_values(1).iloc[0]

        from sklearn.neighbors import KDTree

        cstats = gem.constraints_stats.T.values
        tree = KDTree(cstats)

        dist, ind = tree.query([cstats[20]], k=5)

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

    def test_stats_paper(self):
        import matplotlib.pyplot as plt

        # First load constraint stats from HDF
        df = pd.read_excel(data_path, sheetname=sheet)
        gem = GeneralizedEvaluationMetric(stats_dvh, df)
        gem.load_constraints_stats(database_file, sheet)

        gem_pop = PopulationBasedGEM(stats_dvh, df)
        gem_pop.load_constraints_stats(database_file, sheet)

        # get wes
        # gem.weighted_cumulative_probability(0, structure_name='LIPS')
        # matching db and dvhs
        db_df = stats_dvh.db_df
        dvh_df = stats_dvh.dvh_data
        gem_plans = []
        gemp = []
        for row in range(len(db_df)):
            dvh_i = dvh_df.iloc[row]
            pi_t = PlanningItemDVH(plan_dvh=dvh_i)
            gem_t = gem.calc_gem(pi_t)
            gem_p = gem_pop.calc_gem(pi_t)
            gem_plans.append(gem_t)
            gemp.append(gem_p)

        db_df['GEM'] = gem_plans
        db_df['GEM_pop'] = gemp

        plt.plot(db_df['score'], db_df['GEM'], '.')

        ranking = db_df.sort_values('GEM').drop("Path", axis=1)
        ranking_pop = db_df.sort_values('GEM_pop').drop("Path", axis=1)
        # removing outliers of fake planes

        ranking = ranking.drop([197, 198])
        ranking_pop = ranking_pop.drop([197, 198])
        pass
