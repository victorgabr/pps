from unittest import TestCase

import pandas as pd
# TODO STORE PRECOMPUTED STATISTICS
from competition.tests import data_path, sheet, database_file
from pyplanscoring.competition.statistical_dvh import StatisticalDVH, GeneralizedEvaluationMetric, PlanningItemDVH, \
    PopulationBasedGEM

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


class TestPopulationBasedGEM(TestCase):
    def test_get_empirical_median(self):
        # test popupation based GE score

        # select a plan to estimate GEM
        plan_dvh = stats_dvh.get_plan_dvh(0)
        pi = PlanningItemDVH(plan_dvh=plan_dvh)

        # First load constraint stats from HDF
        gem1 = GeneralizedEvaluationMetric(stats_dvh, df)
        gem1.load_constraints_stats(database_file, sheet)
        gem_1 = gem1.calc_gem(pi)

        # First load constraint stats from HDF
        gem2 = PopulationBasedGEM(stats_dvh, df)
        gem2.load_constraints_stats(database_file, sheet)
        gem_2 = gem2.calc_gem(pi)
        self.assertNotAlmostEqual(gem_1, gem_2)
