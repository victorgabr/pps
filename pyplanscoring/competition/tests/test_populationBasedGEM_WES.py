from unittest import TestCase

import pandas as pd
# TODO STORE PRECOMPUTED STATISTICS
from competition.tests import data_path, sheet, database_file
from pyplanscoring.competition.statistical_dvh import StatisticalDVH, GeneralizedEvaluationMetricWES, \
    PopulationBasedGEM_WES

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


class TestPopulationBasedGEM_WES(TestCase):
    def test_load_constraints_stats(self):
        structure_name = 'LIPS'
        gem_wes_obj = GeneralizedEvaluationMetricWES(stats_dvh, df)
        gem_wes_obj.load_constraints_stats(database_file, sheet)
        gem_wes = gem_wes_obj.get_gem_wes(0, structure_name)

        gem_wes_obj1 = PopulationBasedGEM_WES(stats_dvh, df)
        gem_wes_obj1.load_constraints_stats(database_file, sheet)
        gem_wes1 = gem_wes_obj1.get_gem_wes(0, structure_name)
        self.assertNotAlmostEqual(gem_wes, gem_wes1)
