from unittest import TestCase

import matplotlib.pyplot as plt

from pyplanscoring.competition.statistical_dvh import HistoricPlanData, StatisticalDVH

root_folder = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/ECLIPSE/ECPLIPSE_VMAT'
hist_data = HistoricPlanData(root_folder)
hist_data.set_participant_folder()
hist_data.load_dvh()

# set stats dvh

stats_dvh = StatisticalDVH()
stats_dvh.set_data(hist_data.dvh_data)

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


class TestStatisticalDVH(TestCase):
    def test_set_data(self):
        assert len(stats_dvh.df_data) == len(stats_dvh.structure_names)

    def test_plot_historical_dvh(self):
        for str_name in str_names:
            stats_dvh.plot_historical_dvh(str_name)

        plt.close('all')
