"""
Module based on paper by Mayo et al

http://www.sciencedirect.com/science/article/pii/S2452109417300611

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as itp
# Todo historic plan data from each folder
from sklearn.decomposition import PCA

from pyplanscoring.core.constraints.types import DVHData
from pyplanscoring.core.dvhcalculation import load


class HistoricPlanData:
    def __init__(self, root_folder):
        self._root_folder = root_folder
        self.folder_data = []
        self.dvh_files = []
        self.dvh_data = []

    @property
    def root_folder(self):
        return self._root_folder

    def set_participant_folder(self):
        for folder in os.listdir(self.root_folder):
            participant_folder = os.path.join(self.root_folder, folder)
            self.folder_data.append(participant_folder)
            dvh = self.get_calculated_dvh(participant_folder)
            if dvh:
                self.dvh_files.append(dvh)

    @staticmethod
    def get_calculated_dvh(participant_folder):
        return [os.path.join(root, name) for root, dirs, files in os.walk(participant_folder) for name in files if
                name.strip().endswith('.dvh')]

    def load_dvh(self):
        for p in self.dvh_files:
            for f in p:
                pyplan_dvh = load(f)
                dvh = pyplan_dvh['DVH']
                self.dvh_data.append(dvh)


class StatisticalDVH:
    def __init__(self):
        self._dvh_data = None
        self._structure_names = []
        self._df_data = {}

    @property
    def structure_names(self):
        return self._structure_names

    @property
    def df_data(self):
        return self._df_data

    @property
    def dvh_data(self):
        return self._dvh_data

    def set_data(self, dvh_data):
        self._dvh_data = dvh_data
        self._structure_names = list(dvh_data[0].keys())
        for s in self._structure_names:
            doses = []
            volume = []
            for data in dvh_data:
                dvh = DVHData(data[s])
                doses.append(dvh.dose_focused_format)
                volume = dvh.volume_focused_format
            self._df_data[s] = pd.DataFrame(doses, columns=volume)

    def plot_historical_dvh(self, structure_name):
        # Todo implement structure name matching
        plot_data = self.df_data[structure_name]
        fig, ax = self.statistical_dvh_plot(plot_data, structure_name, 'Dose [cGy]', 'Volume [%]')

    @staticmethod
    def statistical_dvh_plot(sc_data, title='', x_label='', y_label=''):
        """
            Plots statistical DVH with confidence intervals

        :param sc_data: Structure historical DVH data - DataFrame
        :param ax: Matplotlib axis
        :param title: Figure title - string
        :param x_label: xlabel - string. e.g. Dose [cGy]
        :param y_label: ylabel - string. e.g. Volume [%]
        :return:
        """
        median_data = sc_data.median()
        x_data = median_data.values
        y_data = median_data.index
        low_50 = sc_data.quantile(0.30).values
        high_50 = sc_data.quantile(0.70).values
        low_95 = sc_data.quantile(0.10).values
        high_95 = sc_data.quantile(0.90).values

        # plot data
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, '--', lw=1, color='black', alpha=1, label='Median')
        ax.fill_between(high_50, y_data, color='#539caf', alpha=0.8, label='40% CI')
        ax.fill_between(low_50, y_data, color='white', alpha=1)
        ax.fill_between(high_95, y_data, color='#539caf', alpha=0.4, label='90% CI')
        ax.fill_between(low_95, y_data, color='white', alpha=1)
        ax.set_ylim([0, 110])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(loc='best')

        return fig, ax


class WeightedExperienceScore:
    def __init__(self, df_data):
        """
            WeightedExperienceScore constructor
        :param df_data: StatisticalDVH data from a structure
        """

        self._df_data = df_data
        self._quantiles = self.calc_quantiles(df_data)
        self._prob_interp = self.get_probability_interpolator(self._quantiles)

    @property
    def quantiles(self):
        return self._quantiles

    @property
    def probability_interpolator(self):
        return self._prob_interp

    @property
    def df_data(self):
        return self._df_data

    @df_data.setter
    def df_data(self, value):
        self._df_data = value
        self._quantiles = self.calc_quantiles(value)

    @property
    def weight_bin_width(self):
        """
        The volume intervals spacing the Dx%[Gy] points
        defined the weighting values for bin width (wbi).
        :return:  Weighting values for bin width (wbi).
        """
        return np.asarray(self._df_data.columns, dtype=float)

    @property
    def weight_pca(self):
        """
        :return: weighting factor coefficients (wpcai) from PCA
        """
        return self.get_pca_eingenvector()

    @staticmethod
    def calc_probabilities(dvh, prob_interp):
        """
            probability of historical Dx%[Gy] values
            being less than or equal the queried dvh
        :param dvh: Pandas Series (index=Volume[%] 100 - 0, values= Dose[cGy])
        :param prob_interp: Dictionary {Vol[%], Probability Interpolator}
        :return: Numpy array of cumulative probabilities
        """
        probs = np.zeros(len(dvh))
        i = 0
        for row in dvh.iteritems():
            func = prob_interp[row[0]]
            probs[i] = func(row[1])
            i += 1

        return probs

    @staticmethod
    def calc_quantiles(structure_dvhs):
        """
            Quantile is the number k that P(x<k)= probability

        :param structure_dvhs: DataFrame index = doses, columns=volumes[%]
        :return: Quantiles - DataFrame(index=volumes,columns=probabilities)
        """
        percentil_range = np.arange(0, 101, 1)
        n_lines = len(structure_dvhs.columns)
        n_cols = len(percentil_range)
        # Pre allocate memory to improve performance
        qtd_volumes = np.zeros((n_lines, n_cols))
        values = structure_dvhs.values
        # each line refers to DVH observation
        for i in range(n_lines):
            # Pre allocate memory to improve performance
            percentile_per_volume = np.zeros(n_cols)
            # calculate 0-100 percentil per volume
            for j in range(n_cols):
                percentile_per_volume[j] = np.percentile(values[:, i], percentil_range[j])
            qtd_volumes[i, :] = percentile_per_volume

        result = pd.DataFrame(qtd_volumes, index=structure_dvhs.columns, columns=percentil_range / 100.0)

        return result

    @staticmethod
    def get_probability_interpolator(quantiles_data):
        """
            Map volumes (%) to probability interpolation from quantiles
        :param quantiles_data:
        :return: Dictionary {Vol[%], Probability Interpolator}
        """
        prob_interp = {}
        for vol in quantiles_data.index:
            data = quantiles_data.loc[vol]
            prob_interp[vol] = itp.interp1d(data.values, data.index, bounds_error=False, fill_value=(0, 1))

        return prob_interp

    def get_pca_eingenvector(self):
        """
            calc the magnitude of the components of the first eigenvector
            from principal component analysis of the Dx%[Gy]
        :return: weighting factor coefficients (wpcai)
        """

        pca = PCA(n_components=1)
        pca.fit(self.df_data)
        eigenvector = np.asarray(pca.components_).flatten()

        return np.abs(eigenvector)

    def weighted_cumulative_probability(self, dvh):
        """
            return the cumulative
            to that of the present treatment plan. dvh probability of
            historical Dx%[Gy] values being less than or equal the queried dvh
        :param dvh: Actual DVH
        :return: Array of cumulative probabilities
        """

        pi = self.calc_probabilities(dvh, self.probability_interpolator)
        num = np.sum(self.weight_bin_width * self.weight_pca * pi)
        den = np.sum(self.weight_bin_width * self.weight_pca)

        wes = num / den if den != 0 else None

        return wes


class GeneralizedEvaluationMetric:
    pass
