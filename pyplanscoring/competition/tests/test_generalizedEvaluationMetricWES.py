from unittest import TestCase

import pandas as pd
# TODO STORE PRECOMPUTED STATISTICS
from competition.tests import data_path, sheet, database_file
from pyplanscoring.competition.statistical_dvh import StatisticalDVH, GeneralizedEvaluationMetricWES

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
        kt = gem_wes.get_kendall_weights('SPINAL CORD')
        assert len(kt) > 0

    def test_get_gem_wes(self):
        structure_name = 'PAROTID LT'
        gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes_obj.load_constraints_stats(database_file, sheet)
        gem_wes = gem_wes_obj.get_gem_wes(0, structure_name)
        wes = gem_wes_obj.weighted_cumulative_probability(0, structure_name)
        self.assertNotAlmostEqual(wes, gem_wes)
