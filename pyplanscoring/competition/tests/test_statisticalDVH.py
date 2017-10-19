from unittest import TestCase

import matplotlib.pyplot as plt
from competition.tests import root_folder, database_file
from pyplanscoring.competition.statistical_dvh import HistoricPlanDVH, StatisticalDVH

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
        hist_data = HistoricPlanDVH(root_folder)
        hist_data.set_participant_folder()
        hist_data.load_dvh()
        stats_dvh = StatisticalDVH()
        stats_dvh.set_data(hist_data.dvh_data)

    def test_calc_quantiles(self):

        stat_dvh = StatisticalDVH()
        stat_dvh.load_data_from_hdf(database_file)

        # calculate quantiles using volume focused format
        volume_focused_data = stat_dvh.vf_data
        for k, dvh in volume_focused_data.items():
            quantiles = stat_dvh.calc_quantiles(structure_dvhs=dvh)
            quantiles.to_hdf(database_file, key=k)

    def test_get_quantiles(self):
        stat_dvh = StatisticalDVH()
        stat_dvh.load_data_from_hdf(database_file)
        qtl = stat_dvh.get_quantiles('LIPS')
        pass

    def test_plot_historical_dvh(self):
        stats_dvh = StatisticalDVH()
        stats_dvh.load_data_from_hdf(database_file)
        for str_name in str_names:
            stats_dvh.plot_historical_dvh(str_name)

        plt.show()

        plt.close('all')
