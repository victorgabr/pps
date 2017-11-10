import re
from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd

from competition.tests import data_path, database_file, sheet, root_folder
from core.dvhcalculation import load
from pyplanscoring.competition.statistical_dvh import StatisticalDVH, PlanningItemDVH, \
    HistoricPlanDVH, GeneralizedEvaluationMetricWES

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


class TestGeneralizedEvaluationMetricWES(TestCase):
    def test_init(self):
        gem_wes = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes.load_constraints_stats(database_file, sheet)

    def test_constraints_q_parameter(self):
        gem_wes = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes.load_constraints_stats(database_file, sheet)
        q_param = gem_wes.constraints_q_parameter
        assert len(q_param) > 0

    def test_constraints_gem(self):
        gem_wes = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes.load_constraints_stats(database_file, sheet)
        gem_stats = gem_wes.constraints_gem

        assert not gem_stats.empty

    def test_get_kendall_weights(self):
        gem_wes = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes.load_constraints_stats(database_file, sheet)
        kt = gem_wes.get_kendall_weights('SPINAL CORD', )
        assert len(kt) > 0

    def test_get_gem_wes(self):
        structure_name = 'PAROTID LT'
        plan_id = 1
        gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes_obj.load_constraints_stats(database_file, sheet)
        gem_wes = gem_wes_obj.get_gem_wes(plan_id, structure_name, structure_name)
        wes = gem_wes_obj.weighted_cumulative_probability(plan_id, structure_name)
        self.assertNotAlmostEqual(wes, gem_wes)

    def test_difficulty_ranking_score(self):
        gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes_obj.load_constraints_stats(database_file, sheet)
        drs0 = gem_wes_obj.difficulty_ranking_score()
        assert not drs0.empty

    def test_plot_scores(self):
        structure_name = 'SPINAL CORD'

        plan_id = 100
        constraint = structure_name
        gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes_obj.load_constraints_stats(database_file, sheet)
        gem_wes_obj.plot_scores(plan_id, structure_name, constraint)
        import matplotlib.pyplot as plt
        plt.show()

    def test_stats_paper(self):
        # todo rank WES
        # First load constraint stats from HDF
        df = pd.read_excel(data_path, sheetname=sheet)
        gem = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem.load_constraints_stats(database_file, sheet)

        # get wes

        gem.weighted_cumulative_probability(0, structure_name='LIPS')

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

        # load winner DVH
        winner_df = df.sort_values(1).iloc[0]
