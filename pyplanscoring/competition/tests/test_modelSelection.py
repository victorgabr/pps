from unittest import TestCase

import pandas as pd

from competition.statistical_dvh import StatisticalDVH, GeneralizedEvaluationMetric, ModelSelection
from competition.tests import data_path, database_file, sheet

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

gem = GeneralizedEvaluationMetric(stats_dvh, df)
gem.load_constraints_stats(database_file, sheet)
data = gem.constraints_stats.dropna(axis=1).T.values

sname = 'PAROTID LT'
dvh = stats_dvh.vf_data[sname]


class TestModelSelection(TestCase):
    def test_fit(self):
        ms = ModelSelection(data)
        # ms.fit()
        ms.fit(plot=True)
        cp = ms.get_fa_weights(fit=True)
        pcaw = ms.get_pca_weights(fit=True)
        df['fa_weigts'] = cp
        df['pca_weights'] = pcaw
        print(df.sort_values('fa_weigts'))
