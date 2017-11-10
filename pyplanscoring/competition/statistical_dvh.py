"""
Module based on paper by Mayo et al

http://www.sciencedirect.com/science/article/pii/S2452109417300611

"""
import difflib
import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as itp
import scipy.special as sp
import seaborn as sns
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, GridSearchCV

from core.constraints.query import QueryExtensions
from pyplanscoring.core.constraints.constraints import MayoConstraintConverter, ConstraintType
from pyplanscoring.core.constraints.metrics import PlanningItem
from pyplanscoring.core.constraints.types import DVHData, QuantityRegex, DoseUnit, DoseValuePresentation, DoseValue, \
    VolumePresentation, QueryType
from pyplanscoring.core.dvhcalculation import load


class PlanningItemDVH(PlanningItem):
    """Sub class to encapsulate pyplanscoring DVH data
    """

    def __init__(self, rp_dcm=None, rs_dcm=None, rd_dcm=None, plan_dvh=None):
        super().__init__(rp_dcm, rs_dcm, rd_dcm)
        # TODO OVERLOAD DICOM-DVH DATA ?
        self.dvh_data = plan_dvh

    @property
    def dvh_data(self):
        return self._dvhs

    @dvh_data.setter
    def dvh_data(self, value):
        """

            set calculated dvh data by pyplanscoring
        :param value: dvh dictionary
        """
        self._dvhs = value

    @property
    def dose_value_presentation(self):
        du = ''
        for k, v in self.dvh_data.items():
            du = v['doseunits']
            break
        unit = QuantityRegex.string_to_quantity(du)
        if unit == DoseUnit.Gy or unit == DoseUnit.cGy:
            return DoseValuePresentation.Absolute
        else:
            return DoseValuePresentation.Relative

    @property
    def structures(self):
        """ Return list of structure names in DVH"""

        return [s for s in self.dvh_data.keys()]

    def contains_structure(self, struct_id):
        # normalize structure names
        struct_id = self.normalize_string(struct_id)
        norm_struc_names = [self.normalize_string(s) for s in self.structures]

        # map normalized and original strings
        structure_names_map = dict(zip(norm_struc_names, self.structures))

        matches = difflib.get_close_matches(struct_id, norm_struc_names, n=1)

        return matches, structure_names_map

    def get_structure(self, struct_id):
        """
             Gets a structure (if it exists from the structure set references by the planning item
        :param struct_id:
        :return: Structure
        """
        match, names_map = self.contains_structure(struct_id)
        if match:
            original_name = names_map[match[0]]
            return self.dvh_data.get(original_name)
        else:
            return "Structure %s not found" % struct_id

    def get_dvh_cumulative_data(self, structure, dose_presentation=None, volume_presentation=None):
        """
            Get CDVH data DVH dictionary - pyplanscoring
        :param structure: Structure
        :param dose_presentation: DoseValuePresentation
        :param volume_presentation: VolumePresentation
        :return: DVHData
        """
        dvh = self.get_structure(structure)
        return DVHData(dvh)

    def execute_query(self, mayo_format_query, ss):
        """
        :param pi: PlanningItem
        :param mayo_format_query: String Mayo query
        :param ss: Structure string
        :return: Query result
        """
        # TODO Refactor CI calculation
        query = QueryExtensions()
        query.read(mayo_format_query)
        if query.query_type == QueryType.CI:
            return self.query_ci_stats(query, ss)
        else:
            return query.run_query(query, self, ss)

    def query_ci_stats(self, query, target_name):
        """
            Calculates the Paddick conformity index (PMID 11143252) as Paddick CI = (TVPIV)2 / (TV x PIV).
            TVPIV = Target volume covered by Prescription Isodose volume
            TV = Target volume
            using plan DVH data of BODY
        :param target_name: string structure name
        :param query: Query extensions
        :return: CI
        """
        # getting external
        max_volume_key = max(self.dvh_data, key=lambda i: self.dvh_data[i]['data'][0])

        target_structure = self.get_structure(target_name)
        dose_unit = query.get_dose_unit(query)
        reference_dose = DoseValue(query.query_value, dose_unit)
        prescription_vol_isodose = self.get_volume_at_dose(max_volume_key, reference_dose,
                                                           VolumePresentation.absolute_cm3)

        target_vol_isodose = self.get_volume_at_dose(target_name, reference_dose,
                                                     VolumePresentation.absolute_cm3)
        target_vol = target_structure['data'][0] * VolumePresentation.absolute_cm3

        ci = (target_vol_isodose * target_vol_isodose) / (target_vol * prescription_vol_isodose)

        # if not np.isfinite(ci):
        #     raise Exception('Error')

        d_pres = reference_dose.get_presentation()
        dvh = self.get_dvh_cumulative_data(max_volume_key, d_pres, VolumePresentation.absolute_cm3)
        tmp = dvh.get_volume_at_dose(reference_dose, VolumePresentation.absolute_cm3)

        fci = float(ci)

        return fci


class HistoricPlanDVH:
    def __init__(self, root_folder=''):
        self._root_folder = root_folder
        self.folder_data = []
        self.dvh_files = []
        self.dvh_data = []
        self.map_part = []

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
                _, part = os.path.split(participant_folder)
                self.map_part.append([part, dvh])

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

    def to_hdf(self, path_to_filename):
        """
            Save historic DVH data into HDF5 file
        :param path_to_filename: path to hdf5 filename
        """
        data_df = pd.DataFrame(self.dvh_data)
        key = self.__class__.__name__
        data_df.to_hdf(path_to_filename, key)

    def load_hdf(self, path_to_filename):
        """
            Load HDF DVH file. Key data - HistoricPlanDVH
        :param path_to_filename: *.hdf5
        :return: Pandas Dataframe - Historical DVH data
        """
        key = self.__class__.__name__
        return pd.read_hdf(path_to_filename, key)


class StatisticalDVH:
    def __init__(self):
        self._dvh_data = None
        self._structure_names = []
        self._vf_data = {}
        self._path_to_hdf_file = ''

    @property
    def structure_names(self):
        return self._structure_names

    @property
    def vf_data(self):
        return self._vf_data

    @property
    def dvh_data(self):
        return self._dvh_data

    def load_data_from_hdf(self, path_to_hdf_file):
        """
            Load original DVH data from HDF file - HistoricPlanDVH
            and set volume focused format data
        :param path_to_hdf_file: path to hdf5 file
        """
        df = HistoricPlanDVH().load_hdf(path_to_hdf_file)
        # load original data
        self._dvh_data = df
        # set file path
        self._path_to_hdf_file = path_to_hdf_file

        # Generate volume focused format
        vf_data = {}
        for s in self.dvh_data.columns:
            doses = []
            volume = []
            for row in self.dvh_data.iterrows():
                try:
                    dvh = DVHData(row[1][s])
                    doses.append(dvh.dose_focused_format)
                    volume = dvh.volume_focused_format
                except:
                    # todo debug this
                    pass
            vf_data[s] = pd.DataFrame(doses, columns=volume)

        self._vf_data = vf_data

    @staticmethod
    def get_vf_dvh(dvh_data):
        """
            Convert plan DVH data to VF format
        :param dvh_data: Plan DVH data
        :return: Plan DVH in VF format
        """
        vf_data = {}
        for s, val in dvh_data.items():
            dvh = DVHData(val)
            doses = dvh.dose_focused_format
            volume = dvh.volume_focused_format
            vf_data[s] = pd.Series(doses, index=volume)

        return vf_data

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
            self._vf_data[s] = pd.DataFrame(doses, columns=volume)

    def get_structure_stats(self, structure_name):
        return self.vf_data[structure_name]

    def save_quantiles_data(self, path_to_hdf_file):
        """
            Saves pre calculated quantiles at HDF database file.
            key: structure name
        :param path_to_hdf_file:
        """
        for k, dvh in self.vf_data.items():
            quantiles = self.calc_quantiles(structure_dvhs=dvh)
            quantiles.to_hdf(path_to_hdf_file, key=k)

    def save_t_sne_data(self, path_to_hdf_file):
        """
            Saves pre calculated quantiles at HDF database file.
            key: structure name
        :param path_to_hdf_file:
        """
        for k, dvh in self.vf_data.items():
            projected = TSNE().fit_transform(dvh)
            key = 'T_SNE_' + k
            projected = pd.DataFrame(projected)
            projected.to_hdf(path_to_hdf_file, key=key)

    def get_t_sne(self, structure_name):
        structure_name = 'T_SNE_' + structure_name

        t_sne = pd.read_hdf(self._path_to_hdf_file, structure_name)
        return t_sne.values.T

    def get_quantiles(self, structure_name):
        qtl = pd.read_hdf(self._path_to_hdf_file, structure_name)
        return qtl

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

    def plot_historical_dvh(self, structure_name, xlabel, ylabel, title):
        # Todo implement structure name matching
        plot_data = self.vf_data[structure_name]
        fig, ax = self.statistical_dvh_plot(plot_data, xlabel, ylabel, title)
        return fig, ax

    def get_plan_dvh(self, plan_id):
        """
            Gets plan DVH - full dictionary

        :param plan_id: plan id - from database 0-indexed
        :return: Plan DVH
        :rtype: dict
        """
        return self.dvh_data.loc[plan_id].to_dict()

    # confidence intervals
    @staticmethod
    def empirical_ci(data_samples, alpha):
        pl = ((1.0 - alpha) / 2.0)
        lower = data_samples.quantile(pl)
        pu = (alpha + ((1.0 - alpha) / 2.0))
        upper = data_samples.quantile(pu)
        return lower.values, upper.values

    def statistical_dvh_plot(self, structure_vf_data, x_label='', y_label='', title=''):
        """
            Plots statistical DVH with confidence intervals

        :param structure_vf_data: Structure historical DVH data - DataFrame
        :param ax: Matplotlib axis
        :param title: Figure title - string
        :param x_label: xlabel - string. e.g. Dose [cGy]
        :param y_label: ylabel - string. e.g. Volume [%]
        :return:
        """
        median_data = structure_vf_data.median()
        x_data = median_data.values
        y_data = median_data.index

        low_50, high_50 = self.empirical_ci(structure_vf_data, 0.5)
        low_95, high_95 = self.empirical_ci(structure_vf_data, 0.75)

        # plot data
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, '--', lw=1, color='black', alpha=1, label='Median')
        ax.fill_between(high_50, y_data, color='#539caf', alpha=0.8, label='50% CI')
        ax.fill_between(low_50, y_data, color='white', alpha=1)
        ax.fill_between(high_95, y_data, color='#539caf', alpha=0.4, label='75% CI')
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

        self._vf_data = df_data
        self._quantiles = self.calc_quantiles(df_data)
        self._prob_interp = self.get_probability_interpolator(self._quantiles)
        self._constraints_stats = []

    @property
    def quantiles(self):
        return self._quantiles

    @property
    def probability_interpolator(self):
        return self._prob_interp

    @property
    def vf_data(self):
        return self._vf_data

    @vf_data.setter
    def vf_data(self, value):
        self._vf_data = value
        self._quantiles = self.calc_quantiles(value)
        self._prob_interp = self.get_probability_interpolator(self._quantiles)

    @property
    def weight_bin_width(self):
        """
        The volume intervals spacing the Dx%[Gy] points
        defined the weighting values for bin width (wbi).
        :return:  Weighting values for bin width (wbi).
        """
        return np.asarray(self._vf_data.columns, dtype=float)

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
        pca.fit(self.vf_data)
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


class WeightedExperienceScoreBase:
    def __init__(self, statistical_dvh_data):
        self._stats_dvh_data = statistical_dvh_data

    @property
    def stats_dvh(self):
        return self._stats_dvh_data

    @stats_dvh.setter
    def stats_dvh(self, value):
        """
            Set dvh database object StatisticalDVH
        :param value: String - Path to *.hdf6 file or StatisticalDVH with load hdf5 file
        """
        if isinstance(value, str):
            self.stats_dvh.load_data_from_hdf(value)
        else:
            self._stats_dvh_data = value

    def weighted_cumulative_probability(self, plan_id, structure_name):

        struc_vf_data = self.stats_dvh.vf_data[structure_name]
        dvh = struc_vf_data.loc[plan_id]
        quantiles = self.stats_dvh.get_quantiles(structure_name)
        prob_interp = self.get_probability_interpolator(quantiles)
        weight_bin_width = np.asarray(dvh.index, dtype=float)
        pi = self.calc_probabilities(dvh, prob_interp)
        weight_pca = self.get_pca_eingenvector(struc_vf_data)

        num = np.sum(weight_bin_width * weight_pca * pi)
        den = np.sum(weight_bin_width * weight_pca)
        wes = num / den if den != 0 else None
        return wes

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
    def get_pca_eingenvector(dvh):
        """
            calc the magnitude of the components of the first eigenvector
            from principal component analysis of the Dx%[Gy]
        :return: weighting factor coefficients (wpcai)
        """
        pca = PCA(n_components=1)
        pca.fit(dvh)
        eigenvector = np.asarray(pca.components_).flatten()

        return np.abs(eigenvector)


class GeneralizedEvaluationMetric:
    """
    A generalized evaluation metric (GEM) provides a continuous scoring value
    for a set of discrete threshold-priority constraints
    """

    def __init__(self, statistical_dvh_data, discrete_constraints):
        """
            A generalized evaluation metric (GEM) provides a continuous scoring value
            for a set of discrete threshold-priority constraints
        :param statistical_dvh_data: StatisticalDVH object
        """

        self._stats_dvh_data = statistical_dvh_data
        self._discrete_constraints = pd.DataFrame()
        self._constraints_stats = pd.DataFrame()
        self._priority = []
        self._constraints_values = pd.DataFrame()
        self.plan_constraints_results = pd.DataFrame()

        self.discrete_constraints = discrete_constraints

    @property
    def constraints_values(self):
        return self._constraints_values

    @constraints_values.setter
    def constraints_values(self, value):
        self._constraints_values = value

    @property
    def priority_constraints(self):
        return self._priority

    @property
    def discrete_constraints(self):
        return self._discrete_constraints

    @discrete_constraints.setter
    def discrete_constraints(self, value):
        mcs = []
        for row in value.iterrows():
            structure_name = row[1]['Structure Name']
            priority = row[1]['Priority']
            constraint_string = row[1]['Constraint']
            converter = MayoConstraintConverter()
            structure_constraint = converter.convert_to_dvh_constraint(structure_name, priority, constraint_string)
            mcs.append(structure_constraint)

        value['Mayo Constraints'] = mcs
        value['constraint_value'] = value['Mayo Constraints'].apply(lambda x: float(x.mayo_constraint.constraint_value))

        self._priority = value['Priority'].astype(float)
        self._constraints_values = value['constraint_value']
        value['Metric Type'] = value['Mayo Constraints'].apply(lambda x: x.constraint_type)
        self._discrete_constraints = value

    @property
    def confidence_intervals(self):
        # Calc quantiles 90% quantile
        # upper_90 = self.constraints_stats.quantile(0.9, axis=1)
        lower_90, upper_90 = self.empirical_ci(self.constraints_stats, 0.9)
        quantile_90 = np.zeros(len(lower_90))
        mask_min = self.discrete_constraints['Metric Type'] == ConstraintType.MINIMUM
        mask_max = self.discrete_constraints['Metric Type'] == ConstraintType.MAXIMUM
        quantile_90[mask_min] = lower_90[mask_min]
        quantile_90[mask_max] = upper_90[mask_max]
        return quantile_90

    @property
    def constraints_stats(self):
        return self._constraints_stats

    @constraints_stats.setter
    def constraints_stats(self, value):
        self._constraints_stats = value

    def eval_constraints(self, pi_dvh):
        """
        :rtype: pd.DataFrame
        :param pi_dvh: PlanningItemDVH object
        :return: DataFrame - Constraint Results
        """

        constraint_results = []
        ctr = self.discrete_constraints.copy()
        for row in ctr.iterrows():
            mayo_constraint = row[1]['Mayo Constraints']
            cr = mayo_constraint.constrain(pi_dvh)
            constraint_results.append(float(cr.constraint_result))

        ctr['Result'] = constraint_results

        return ctr

    def calc_constraints_stats(self):
        n = len(self._stats_dvh_data.dvh_data)
        # pre alocating
        constraints_stats = []
        for i in range(n):
            try:
                plan_dvh = self._stats_dvh_data.get_plan_dvh(i)
                pi = PlanningItemDVH(plan_dvh=plan_dvh)
                constraints_result_df = self.eval_constraints(pi)
                # getting values to later save it using hdf5 store
                constraints_stats.append(constraints_result_df['Result'])
            except:
                # TODO debug it
                pass

        self._constraints_stats = pd.concat(constraints_stats, axis=1)
        self._constraints_stats.columns = range(len(self._constraints_stats.columns))
        return self._constraints_stats

    def save_constraints_stats(self, path_to_hdf, key):
        """
            Save constrains stats to a hdf5 file
        :param path_to_hdf: Path to hdf5 file.
        :param key: string. hdf5 data key
        """
        constraints_data_stats = self.calc_constraints_stats()
        constraints_data_stats.to_hdf(path_to_hdf, key)

    def load_constraints_stats(self, path_to_hdf, key):
        """
            load constrains stats from a hdf5 file
        :param path_to_hdf: Path to hdf5 file.
        :param key: string. hdf5 data key
        """
        self.constraints_stats = pd.read_hdf(path_to_hdf, key)

    @staticmethod
    def sigmoidal_curve_using_normal_cdf(plan_value, constraint_value, q, mc_type=0):
        """Appendix B: Sigmoidal curve using Normal C.D.F.
        The normal p.d.f. is frequently used for values that can range over positive and negative values.
        In that case the sigmoidal function used in the GEM calculation is the normal c.d.f.."""
        # TODO VECTORIZE IT
        if mc_type == 0:
            delta = plan_value - constraint_value
            return 1 / 2 * (1 + sp.erf((delta) / (q * constraint_value)))
        if mc_type == 1:
            delta = constraint_value - plan_value
            return 1 / 2 * (1 + sp.erf((delta) / (q * constraint_value)))

    @staticmethod
    def get_q_parameter(constraint_ci, constraint_value, target_prob=0.95, mc_type=0):
        """
            If Upper 90% CI ≥ Constraint Value, q is selected for this equation B.2
            Appendix B.

            Ref. http://www.sciencedirect.com/science/article/pii/S2452109417300611

        :param constraint_ci: {max: upper_90_ci, low: lower_ci}
        :param constraint_value: Protocol's Constraint Value
        :param target_prob: desired ALARA target probability
        :return: scale parameter of normal CDF
        """
        # TODO REVIEW IT STEP BY STEP
        u = constraint_ci
        c = constraint_value
        g = target_prob
        if mc_type == ConstraintType.MAXIMUM:  # maximum constraint
            if constraint_ci >= constraint_value:
                q = (c - u) / (c * sp.erfinv(1 - 2 * g))
                return abs(q)

            else:
                return 1 - target_prob
        if mc_type == ConstraintType.MINIMUM:  # minimum  constraint
            if constraint_ci <= constraint_value:
                q = (c - u) / (c * sp.erfinv(1 - 2 * g))
                return abs(q)

            else:
                return 1 - target_prob

    @staticmethod
    def empirical_ci(data_samples, alpha):
        pl = ((1.0 - alpha) / 2.0)
        lower = data_samples.quantile(pl, axis=1)
        pu = (alpha + ((1.0 - alpha) / 2.0))
        upper = data_samples.quantile(pu, axis=1)
        return lower.values, upper.values

    def calc_gem(self, pi):
        """
            The GEM is calculated as a normalized weighed sum of deviation scores.
            In keeping with clinical practice, low numerical values for prioritization (eg, 1)
            conveyed greater weight than higher values (eg, 3)

        :param pi:  PlanningItemDVH or DataFrame plan index [0-indexed]
        :return: Plan GEM
        """
        # method overloading
        if not isinstance(pi, PlanningItemDVH) and isinstance(pi, int):
            plan_dvh = self._stats_dvh_data.get_plan_dvh(pi)
            pi = PlanningItemDVH(plan_dvh=plan_dvh)

        if self.constraints_stats.empty:
            # calculate stats if not loaded
            self.calc_constraints_stats()

        # get constraints results
        plan_constraints_results = self.eval_constraints(pi)

        # pre allocating memory
        cdf = [0] * len(plan_constraints_results)
        b2_exp = 2 ** -(self.priority_constraints - 1)
        ci_90 = self.confidence_intervals

        for row in plan_constraints_results.iterrows():
            plan_value = row[1]['Result']
            constraint_value = self.constraints_values[row[0]]

            # check plan value
            if not np.isfinite(plan_value):
                return None

            mc = row[1]['Mayo Constraints']
            qi = self.get_q_parameter(ci_90[row[0]], constraint_value, mc_type=mc.constraint_type)
            cdfi = self.sigmoidal_curve_using_normal_cdf(plan_value, constraint_value, qi, mc_type=mc.constraint_type)
            cdf[row[0]] = cdfi

        plan_constraints_results['GEM'] = cdf
        self.plan_constraints_results = plan_constraints_results

        gem_score = np.sum(b2_exp * cdf) / np.sum(b2_exp)

        return gem_score

    @staticmethod
    def normalized_incomplete_gamma(k, c_theta):
        """
            P is the cumulative distribution function for the gamma probability distribution function (p.d.f.),
            operating over the same range of input values as DVH metrics (≥0).
            ref. https://www.johndcook.com/blog/gamma_python/
        :return:
        """

        return sp.gammainc(k, c_theta)

    @property
    def calculated_priority(self):
        """
            calculated_priority = round(1-log2(count(plan_values <= constraint values)/(count(plan_values))
        :return:
        """
        if not self.constraints_stats.empty:
            plan_values = self.constraints_stats.values
            constraints_values = self.constraints_values.values[:, np.newaxis]
            # Using numpy broadcasting
            counts = plan_values <= constraints_values
            ratio = counts.sum(axis=1) / counts.shape[1]
            calc_priority = np.round(1 - np.log2(ratio))

            return calc_priority

    def normalized_weighed_sum(self):
        """
            The GEM is calculated as a normalized weighed sum of deviation scores.
            In keeping with clinical practice, low numerical values for prioritization (eg, 1)
            conveyed greater weight than higher values (eg, 3)
        :return:
        """
        return NotImplementedError


class PopulationBasedGEM(GeneralizedEvaluationMetric):
    """
    In practice, individual treatment plans may rarely exceed the constraint values
    defined by literature-derived risk factors. In those cases, GEM scores such as NTCP tend to be near 0.
    An alternative is to use the empirical median of the historical population as the constraint value.
    We define this as the population-based GEM, or GEMpop.
    Using GEMpop, historical distributions determine the steepness of the penalty for exceeding constraint values
    and allow measured distributions to quantify as low as reasonably achievable (ALARA) dose limits
    with respect to historical experience.
    """

    def __init__(self, statistical_dvh_data, discrete_constraints):
        super().__init__(statistical_dvh_data, discrete_constraints)

    def load_constraints_stats(self, path_to_hdf, key):
        super().load_constraints_stats(path_to_hdf, key)
        self.constraints_values = self.get_empirical_median()

    def get_empirical_median(self):
        """
            Get the empirical median of the historical population as the constraint value
        :return:
        """
        return self.constraints_stats.quantile(0.5, axis=1)


class GeneralizedEvaluationMetricWES(GeneralizedEvaluationMetric, WeightedExperienceScoreBase):
    def __init__(self, statistical_dvh_data, discrete_constraints):
        super().__init__(statistical_dvh_data, discrete_constraints)

    @property
    def constraints_q_parameter(self):
        """
            If Upper 90% CI ≥ Constraint Value, q is selected for this equation B.2
            Appendix B.
            calculated for all selected constraints
        :return: array of q parameters
        """
        ctr_type = self.discrete_constraints['Metric Type'].values
        calc_q_parameter = np.vectorize(self.get_q_parameter)
        q = calc_q_parameter(self.confidence_intervals, self.constraints_values.values, mc_type=ctr_type)
        return q

    @staticmethod
    def sigmoidal_curve_using_normal_cdf(plan_value, constraint_value, q, mc_type=0):
        """Appendix B: Sigmoidal curve using Normal C.D.F.
        The normal p.d.f. is frequently used for values that can range over positive and negative values.
        In that case the sigmoidal function used in the GEM calculation is the normal c.d.f.."""
        # TODO VECTORIZE IT
        delta0 = plan_value - constraint_value
        delta1 = constraint_value - plan_value
        delta = np.zeros(plan_value.shape)
        delta[mc_type == 0, :] = delta0[mc_type == 0, :]
        delta[mc_type == 1, :] = delta1[mc_type == 1, :]
        return 1 / 2 * (1 + sp.erf((delta) / (q * constraint_value)))

    @property
    def constraints_gem(self):
        ctr_type = self.discrete_constraints['Metric Type'].values
        plan_values = self.constraints_stats.values
        constraint_values = self.constraints_values.values[:, np.newaxis]
        qi = self.constraints_q_parameter[:, np.newaxis]

        cdfi = self.sigmoidal_curve_using_normal_cdf(plan_values, constraint_values, qi, mc_type=ctr_type)

        sn = self.discrete_constraints['Structure Name']
        cm = self.discrete_constraints['Mayo Constraints']
        c_index = sn.astype(str) + ' - ' + cm.astype(str)
        gem_df = pd.DataFrame(cdfi, index=c_index)

        return gem_df

    def get_structure_gem(self, plan_id, constraint_string):
        """
            Get structure Generalized evaluation metric
        :param plan_id: DataFrame column plan id
        :param constraint_string:constraint_string
        :return: Structure GEM
        """
        return self.constraints_gem[plan_id].loc[constraint_string]

    def get_kendall_weights(self, structure_name, constraint=''):
        """
            Not all points along the DVH curve are equally relevant.
            Toxicities may be more strongly driven by Max[Gy], Mean[Gy] or Dx%[Gy] values,
            dependent on the organ at risk structure.
            To reflect this, an additional weighting factor(wkti) was calculated using the Kendall's tau
            (kti) correlation of Dx%[Gy] values with structure GEM scores.
        :param constraint: Constraint string
        :param structure_name: string
        """
        bp_data = self.stats_dvh.vf_data[structure_name].copy()

        constraint = structure_name + ' - ' + constraint
        col_gem = self.constraints_gem.loc[constraint]
        bp_data['gem'] = col_gem
        res_corr = bp_data.corr(method='kendall')

        # Weighting factors (wkt) were set equal to 0 for kt <0
        # so that they only penalize those DVH points that are associated with undesirable outcomes.

        kt = res_corr.iloc[:-1]['gem'].values
        kt[kt < 0] = 0
        return kt

    def get_gem_wes(self, plan_id, structure_name, structure_constraint=''):
        """
            Generalized evaluation metric–correlated weighted experience score
        :param structure_constraint:
        :param plan_id: int - plan id
        :param structure_name: string
        :return: gem_wes
        """
        struc_vf_data = self.stats_dvh.vf_data[structure_name]
        dvh = struc_vf_data.loc[plan_id]
        quantiles = self.stats_dvh.get_quantiles(structure_name)
        prob_interp = self.get_probability_interpolator(quantiles)
        weight_bin_width = np.asarray(dvh.index, dtype=float)
        pi = self.calc_probabilities(dvh, prob_interp)
        weight_pca = self.get_pca_eingenvector(struc_vf_data)
        # adding (kti) correlation
        kti = self.get_kendall_weights(structure_name, structure_constraint)

        num = np.sum(weight_bin_width * weight_pca * kti * pi)
        den = np.sum(weight_bin_width * weight_pca * kti)

        gem_wes = num / den if den != 0 else None

        return gem_wes

    def calc_plan_gem_wes(self):
        pass

    def difficulty_ranking_score(self):
        """
            The difficulty in meeting each threshold-priority constraint value on the basis of historical experience
            was quantified with a difficulty ranking score (DRS)

            DRS = (2 ** -(Priority - 1)) * GEM upper 50% CI

        :return: difficulty_ranking_score
        """

        _, upper = self.empirical_ci(self.constraints_gem, 0.5)
        b2_exp = 2 ** -(self.priority_constraints - 1.0)
        values = b2_exp.values * upper
        drs = pd.DataFrame(values, index=self.constraints_gem.index, columns=['DRS'])

        return drs

    def plot_scores(self, plan_id, structure_name, constraint='', x_label='Dose [cGy]', y_label='Volume [%]', title=''):
        """
                Plot statistical scores
                ref: http://www.sciencedirect.com/science/article/pii/S2452109417300611
        :param plan_id: Dataframe plan index [0-indexed]
        :param structure_name: string: Structure name
        :param constraint: string: Structure constraint
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param title: Plot title
        :return: Figure and axis object
        """
        plan_dvh = self.stats_dvh.get_plan_dvh(plan_id)
        dvh = plan_dvh[structure_name]
        fig, ax = self.stats_dvh.plot_historical_dvh(structure_name, x_label, y_label, title)

        ax.plot(dvh['dose_axis'], dvh['data'] / dvh['data'][0] * 100, linewidth=2, color='r', label='Plan DVH')
        props = dict(boxstyle='round', facecolor='DarkGreen', alpha=0.3)

        # getting scores
        wes = self.weighted_cumulative_probability(plan_id, structure_name)
        plan_gem = self.calc_gem(plan_id)
        gem_wes = self.get_gem_wes(plan_id, structure_name, constraint)
        textstr = 'Structure WES: %1.3f\n' \
                  'Constraint GEM_WES: %1.3f\n' \
                  'Plan GEM: %1.3f' % (wes, gem_wes, plan_gem)
        ax.text(0.7, 0.90, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', bbox=props)
        ax.legend()

        return fig, ax


class PopulationBasedGEM_WES(GeneralizedEvaluationMetricWES):
    def __init__(self, statistical_dvh_data, discrete_constraints):
        super().__init__(statistical_dvh_data, discrete_constraints)

    def load_constraints_stats(self, path_to_hdf, key):
        super().load_constraints_stats(path_to_hdf, key)
        self.constraints_values = self.get_empirical_median()

    def get_empirical_median(self):
        """
            Get the empirical median of the historical population as the constraint value
        :return:
        """
        return self.constraints_stats.quantile(0.5, axis=1)

    def plot_scores(self, plan_id, structure_name, constraint='', x_label='Dose [cGy]', y_label='Volume [%]', title=''):
        plan_dvh = self.stats_dvh.get_plan_dvh(plan_id)
        dvh = plan_dvh[structure_name]
        fig, ax = self.stats_dvh.plot_historical_dvh(structure_name, x_label, y_label, title)

        ax.plot(dvh['dose_axis'], dvh['data'] / dvh['data'][0] * 100, linewidth=2, color='r', label='Plan DVH')
        props = dict(boxstyle='round', facecolor='DarkGreen', alpha=0.3)

        # getting scores
        wes = self.weighted_cumulative_probability(plan_id, structure_name)
        plan_gem = self.calc_gem(plan_id)
        gem_wes = self.get_gem_wes(plan_id, structure_name, constraint)
        textstr = 'Structure WES: %1.3f\n' \
                  'Constraint GEM_WESpop: %1.3f\n' \
                  'Plan GEMpop: %1.3f' % (wes, gem_wes, plan_gem)
        ax.text(0.7, 0.90, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', bbox=props)
        ax.legend()


class ModelSelection:
    """
        with Probabilistic PCA and Factor Analysis (FA)
        ref.  http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
    """

    def __init__(self, data):
        """
        :param data: [n obs, n features] matrix
        """
        self.data = data
        self.n_components = np.arange(0, data.shape[1])

    def get_fa_weights(self, fit=False):
        if fit:
            n_components_pca, n_components_fa, n_components_pca_mle = self.fit()
        else:
            n_components_fa = 1

        fa = FactorAnalysis(n_components=n_components_fa)
        fa.fit(self.data)

        return abs(fa.components_[0])

    def get_pca_weights(self, fit=False):
        if fit:
            n_components_pca, n_components_fa, n_components_pca_mle = self.fit()
        else:
            n_components_fa = 1

        pca = PCA(svd_solver='full', n_components=n_components_fa)
        pca.fit(self.data)

        return abs(pca.components_[0])

    def fit(self, plot=False):
        pca_scores, fa_scores = self.compute_scores(self.data)
        n_components_pca = self.n_components[np.argmax(pca_scores)]
        n_components_fa = self.n_components[np.argmax(fa_scores)]

        pca = PCA(svd_solver='full', n_components='mle')
        pca.fit(self.data)
        n_components_pca_mle = pca.n_components_

        print("best n_components by PCA CV = %d" % n_components_pca)
        print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
        print("best n_components by PCA MLE = %d" % n_components_pca_mle)

        if plot:
            self.plot_fit(pca_scores, fa_scores, n_components_pca, n_components_fa, n_components_pca_mle)

        return n_components_pca, n_components_fa, n_components_pca_mle

    def compute_scores(self, X):
        pca = PCA(svd_solver='full')
        fa = FactorAnalysis()

        pca_scores, fa_scores = [], []
        for n in self.n_components:
            pca.n_components = n
            fa.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca, X)))
            fa_scores.append(np.mean(cross_val_score(fa, X)))

        return pca_scores, fa_scores

    @staticmethod
    def shrunk_cov_score(X):
        shrinkages = np.logspace(-2, 0, 30)
        cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
        return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))

    @staticmethod
    def lw_score(X):
        return np.mean(cross_val_score(LedoitWolf(), X))

    def plot_fit(self, pca_scores, fa_scores, n_components_pca, n_components_fa, n_components_pca_mle):
        fig, ax = plt.subplots()
        ax.plot(self.n_components, pca_scores, 'b', label='PCA scores')
        ax.plot(self.n_components, fa_scores, 'r', label='FA scores')
        # plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
        ax.axvline(n_components_pca, color='b',
                   label='PCA CV: %d' % n_components_pca, linestyle='--')
        ax.axvline(n_components_fa, color='r',
                   label='FactorAnalysis CV: %d' % n_components_fa,
                   linestyle='--')
        ax.axvline(n_components_pca_mle, color='k',
                   label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

        # compare with other covariance estimators
        ax.axhline(self.shrunk_cov_score(self.data), color='violet',
                   label='Shrunk Covariance MLE', linestyle='-.')
        ax.axhline(self.lw_score(self.data), color='orange',
                   label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

        ax.set_xlabel('nb of components')
        ax.set_xlabel('CV scores')
        ax.legend(loc='lower right')
        plt.show()


class EDAClustering:
    def __init__(self, data, projected):
        self.data = data
        self.projection = projected

    def get_projected_clusters(self, title):
        self.plot_clusters(self.data,
                           self.projection,
                           hdbscan.HDBSCAN,
                           (), {},
                           {'alpha': 1, 's': 80, 'linewidths': 0})

        txt = '%s - Clusters found by %s' % (title, hdbscan.HDBSCAN.__name__)
        plt.title(txt, fontsize=8)
        plt.show()

    @staticmethod
    def plot_clusters(data, projection, algorithm, args, kwds, plot_kwds):
        clusterer = algorithm(*args, **kwds).fit(data)
        labels = algorithm(*args, **kwds).fit_predict(data)
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(*projection, c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

        plt.figure()
        clusterer.condensed_tree_.plot(select_clusters=True,
                                       selection_palette=sns.color_palette('deep', 8))

        plt.figure()
        clusterer.single_linkage_tree_.plot()
