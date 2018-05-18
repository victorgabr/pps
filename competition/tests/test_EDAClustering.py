from unittest import TestCase

import pandas as pd

from competition.statistical_dvh import StatisticalDVH, GeneralizedEvaluationMetric, EDAClustering
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


class TestEDAClustering(TestCase):
    def test_get_projected_clusters(self):
        # ms = ModelSelection(data)
        # ms.fit()
        # ms.fit(plot=True)

        # plot constraint data
        # ec = EDAClustering(data_reduced)
        # ec.get_projected_clusters('Constraints Data')
        # # plot DVH data

        for sname in str_names:
            pass
            dvh = stats_dvh.vf_data[sname]
            proj = stats_dvh.get_t_sne(sname)
            ec = EDAClustering(dvh, proj)
            ec.get_projected_clusters(sname)

            # projected = TSNE().fit_transform(dvh)

    def test_plot_clusters(self):
        self.fail()
