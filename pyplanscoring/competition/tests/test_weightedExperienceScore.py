from unittest import TestCase

from pyplanscoring.competition.statistical_dvh import HistoricPlanData, StatisticalDVH, WeightedExperienceScore

root_folder = r'E:\COMPETITION 2017\final_plans\ECLIPSE\ECPLIPSE_VMAT'
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

# stats_dvh
sn = 'PAROTID LT'
sc = stats_dvh.df_data[sn]

stats_dvh.plot_historical_dvh(sn)


class TestWeightedExperienceScore(TestCase):
    def test_calc_quantiles(self):
        wes = WeightedExperienceScore(sc)
        quantiles_data = wes.calc_quantiles(sc)
        assert len(quantiles_data.columns) == 101
        assert len(quantiles_data.index) == 31

    def test_get_probability_interpolator(self):
        wes = WeightedExperienceScore(sc)
        quantiles_data = wes.calc_quantiles(sc)
        prob_interp = wes.get_probability_interpolator(quantiles_data)
        assert len(prob_interp) == 31

    def test_calc_probabilities(self):
        wes = WeightedExperienceScore(sc)
        quantiles_data = wes.calc_quantiles(sc)
        prob_interp = wes.get_probability_interpolator(quantiles_data)

        # query dvh
        dvh = sc.loc[101]
        probs = wes.calc_probabilities(dvh, prob_interp)
        assert len(probs) == 31

        # query dvh
        dvh1 = sc.loc[1]
        probs1 = wes.calc_probabilities(dvh1, prob_interp)
        assert probs.sum() > probs1.sum()

    def test_get_pca_eingenvector(self):
        wes = WeightedExperienceScore(sc)
        wpca = wes.get_pca_eingenvector()
        assert len(wpca) == 31

    def test_weighted_cumulative_probability(self):
        wes = WeightedExperienceScore(sc)
        # query dvh
        dvh = sc.loc[101]
        probs = wes.weighted_cumulative_probability(dvh)

        # query dvh
        dvh1 = sc.loc[1]
        probs1 = wes.weighted_cumulative_probability(dvh1)
        # assert probs > probs1
        self.fail()
        #
        # import matplotlib.pyplot as plt
        #
        # plt.plot(dvh.values, dvh.index)
        # plt.plot(dvh1.values, dvh1.index)
