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


class TestPopulationBasedGEM(TestCase):
    def test_get_empirical_median(self):
        # test popupation based GE score
        df = pd.read_excel(data_path, sheetname=sheet)

        # select a plan to estimate GEM
        plan_dvh = stats_dvh.get_plan_dvh(0)
        pi = PlanningItemDVH(plan_dvh=plan_dvh)

        # First load constraint stats from HDF
        gem1 = GeneralizedEvaluationMetric(stats_dvh, df)
        gem1.load_constraints_stats(database_file, sheet)
        gem_1 = gem1.calc_gem(pi)

        # First load constraint stats from HDF
        gem_pop = PopulationBasedGEM(stats_dvh, df)
        gem_pop.load_constraints_stats(database_file, sheet)
        gem_2 = gem_pop.calc_gem(pi)
        self.assertNotAlmostEqual(gem_1, gem_2)

        # test low score plan
        pi = PlanningItemDVH(plan_dvh=low_score)
        low_gem_pop = gem_pop.calc_gem(pi)
        low_gem = gem1.calc_gem(pi)

        # test high score plan
        pi = PlanningItemDVH(plan_dvh=high_score)
        high_gem_pop = gem_pop.calc_gem(pi)
        high_gem = gem1.calc_gem(pi)
        assert low_gem_pop > high_gem_pop

    def calc_stats_gem_pop(self):
        # First load constraint stats from HDF
        df = pd.read_excel(data_path, sheetname=sheet)
        gem = PopulationBasedGEM(stats_dvh, df)
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
        df['sc'] = df[0].apply(lambda row: re.findall("\d+\.\d+", row)[0])

        plt.plot(df['sc'], df[1], '.')

        print(df.sort_values(1))
        # load winner DVH
        winner_df = df.sort_values(1).iloc[0]

        winner = hist_data.map_part[61]
        dvh = load(winner[1][0])['DVH']
        pi_t = PlanningItemDVH(plan_dvh=dvh)
        constraints_winner = gem.eval_constraints(pi_t)
        gem_t = gem.calc_gem(pi_t)
        plan_constraints_results = gem.plan_constraints_results

        # load score 100
        winner_100 = hist_data.map_part[34]
        dvh = load(winner_100[1][0])['DVH']
        pi_t = PlanningItemDVH(plan_dvh=dvh)
        constraints_winner_100 = gem.eval_constraints(pi_t)
        gem_t_100 = gem.calc_gem(pi_t)
        plan_constraints_results_100 = gem.plan_constraints_results

        # inspect constraints stats
        # upper_90 = self.constraints_stats.quantile(0.9, axis=1)
        lower_90, upper_90 = gem.empirical_ci(gem.constraints_stats, 0.9)
