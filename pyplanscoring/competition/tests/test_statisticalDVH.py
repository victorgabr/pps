from unittest import TestCase

import matplotlib.pyplot as plt

from competition.tests import root_folder, database_file
from core.dvhcalculation import load
from pyplanscoring.competition.statistical_dvh import HistoricPlanDVH, StatisticalDVH, PlanningItemDVH

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

hist_data = HistoricPlanDVH(root_folder)
hist_data.set_participant_folder()


class TestStatisticalDVH(TestCase):
    def test_set_data(self):
        hist_data = HistoricPlanDVH(root_folder)
        hist_data.set_participant_folder()
        hist_data.load_dvh()
        stats_dvh = StatisticalDVH()
        stats_dvh.set_data(hist_data.dvh_data)

    def test_get_vf_dvh(self):
        stats_dvh = StatisticalDVH()
        stats_dvh.load_data_from_hdf(database_file)

        dvh = load(hist_data.map_part[0][1][0])['DVH']
        pi_t = PlanningItemDVH(plan_dvh=dvh)
        dvhi = stats_dvh.get_vf_dvh(dvh)

        assert dvhi

    def test_save_quantiles_data(self):
        stats_dvh = StatisticalDVH()
        stats_dvh.load_data_from_hdf(database_file)
        stats_dvh.save_quantiles_data(database_file)

    def test_save_t_sne_data(self):
        stats_dvh = StatisticalDVH()
        stats_dvh.load_data_from_hdf(database_file)
        stats_dvh.save_t_sne_data(database_file)

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

    def test_get_t_sne(self):
        stat_dvh = StatisticalDVH()
        stat_dvh.load_data_from_hdf(database_file)
        t_sne = stat_dvh.get_t_sne('SPINAL CORD')
        plt.scatter(*t_sne)
        plt.show()
        pass

    def test_plot_historical_dvh(self):
        stats_dvh = StatisticalDVH()
        stats_dvh.load_data_from_hdf(database_file)
        tech_df = stats_dvh.db_df['Technique']
        txt = tech_df.value_counts().to_string()
        for str_name in str_names:
            plot_data = stats_dvh.vf_data[str_name]
            nplans = len(plot_data)
            xlabel, ylabel = 'Dose [cGy]', 'Volume [%]'
            title = 'Nasopharynx plans - %s - N=%i' % (str_name, nplans)
            stats_dvh.plot_historical_dvh(str_name, xlabel, ylabel, title)
            plt.figtext(0.2, 0.2, txt)
        plt.show()

        plt.close('all')

    def test_get_db(self):

        stats_dvh = StatisticalDVH()
        stats_dvh.load_data_from_hdf(database_file)
