"""
Module based on paper by Mayo et al

http://www.sciencedirect.com/science/article/pii/S2452109417300611

"""
import os

import matplotlib.pyplot as plt
import pandas as pd

# Todo historic plan data from each folder
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
