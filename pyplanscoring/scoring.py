from __future__ import division

import logging
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as itp
from xlsxwriter.utility import xl_rowcol_to_cell

from pyplanscoring.dev.dvhcalculation import calc_dvhs_upsampled, Structure
from pyplanscoring.dicomparser import ScoringDicomParser
from pyplanscoring.dosimetric import read_scoring_criteria
from pyplanscoring.dvhcalc import load

logger = logging.getLogger('scoring')


def get_dvh_files(root_path):
    """
        List all *.dvh files inside a root path and its subfolders
    :param root_path:
    :return: List of *.dvh files
    """
    dvh_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(root_path)
                 for name in files
                 if name.endswith('.dvh')]
    return dvh_files


def get_participant_folder_data(participant_name, root_path):
    """
        Provide all participant required files (RP,RS an RD DICOM FILES)
    :param participant_name: Participant string name
    :param root_path: participant folder
    :return: Pandas DataFrame containing path to files
    """
    files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    filtered_files = OrderedDict()
    for f in files:
        try:
            obj = ScoringDicomParser(filename=f)
            rt_type = obj.GetSOPClassUID()
            if rt_type == 'rtplan':
                participant_data = [participant_name, rt_type]
                filtered_files[f] = participant_data
            if rt_type == 'rtss':
                participant_data = [participant_name, rt_type]
                filtered_files[f] = participant_data
            if rt_type == 'rtdose':
                participant_data = [participant_name, rt_type]
                filtered_files[f] = participant_data
        except:
            logger.exception('Error in file %s' % f)

    data = pd.DataFrame(filtered_files).T

    # Check data consistency
    data_truth = []
    dcm_types = pd.DataFrame(['rtdose', 'rtplan', 'rtss'])
    dcm_files = pd.DataFrame(['RT-DOSE', 'RT-PLAN', 'RT-STRUCTURE'])
    for t in dcm_types.values:
        data_truth.append(t in data[1].values)
    data_truth = np.array(data_truth)

    if len(data) == 3 and np.all(data_truth):
        return True, data
    else:
        return False, dcm_files.loc[~data_truth]


# def get_participant_folder_data(participant_name, root_path):
#     """
#         Provide all participant required files (RP,RS an RD DICOM FILES)
#     :param participant_name: Participant string name
#     :param root_path: participant folder
#     :return: Pandas DataFrame containing path to files
#     """
#     files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
#              name.endswith(('.dcm', '.DCM'))]
#
#     filtered_files = OrderedDict()
#     for f in files:
#         try:
#             obj = ScoringDicomParser(filename=f)
#             rt_type = obj.GetSOPClassUID()
#             if rt_type == 'rtplan':
#                 participant_data = [participant_name, rt_type]
#                 filtered_files[f] = participant_data
#             # if rt_type == 'rtss':
#             #     participant_data = [participant_name, rt_type]
#             #     filtered_files[f] = participant_data
#             if rt_type == 'rtdose':
#                 participant_data = [participant_name, rt_type]
#                 filtered_files[f] = participant_data
#         except:
#             logger.exception('Error in file %s' % f)
#
#     data = pd.DataFrame(filtered_files).T
#
#     # Check data consistency
#     data_truth = []
#     dcm_types = pd.DataFrame(['rtdose', 'rtplan'])
#     dcm_files = pd.DataFrame(['RT-DOSE', 'RT-PLAN'])
#     for t in dcm_types.values:
#         data_truth.append(t in data[1].values)
#     data_truth = np.array(data_truth)
#
#     if len(data) == 2 and np.all(data_truth):
#         return True, data
#     else:
#         return False, dcm_files.loc[~data_truth]


class DVHMetrics(object):
    def __init__(self, dvh):
        """
            Class to encapsulate DVH constrains metrics

        :param dvh: DVH dictionary
        """
        vpp = dvh['data'] * 100 / dvh['data'][0]
        self.volume_pp = np.append(vpp, 0.0)  # add 0 volume to interpolate
        # self.volume_pp = vpp
        self.scaling = dvh['scaling']
        # self.dose_axis = np.arange(len(dvh['data'])) * self.scaling
        self.dose_axis = np.arange(len(dvh['data']) + 1) * self.scaling
        # self.volume_cc = dvh['data']
        self.volume_cc = np.append(dvh['data'], 0.0)
        self.stats = (dvh['max'], dvh['mean'], dvh['min'])
        self.data = dvh

        # setting constrain interpolation functions
        self.fv = itp.interp1d(self.dose_axis, self.volume_pp, fill_value='extrapolate')  # pp
        self.fv_cc = itp.interp1d(self.dose_axis, self.volume_cc, fill_value='extrapolate')  # pp
        self.fd = itp.interp1d(self.volume_pp, self.dose_axis, fill_value='extrapolate')  # pp
        self.fd_cc = itp.interp1d(self.volume_cc, self.dose_axis, fill_value='extrapolate')  # cc

    def eval_constrain(self, key, value):
        """
            Eval constrain helper function
        :param key: Structure name key.
        :param value: Constrain metric
        :return: Constrain value
        """
        if key == 'Global Max':
            return self.data['max']

        if key == 'Dmax_position' and value == 'max':
            return self.stats[0]
        elif key[0] == 'Dcc':
            value = np.asarray([value])
            return self.get_dose_constrain_cc(value)
        elif key[0] == 'D':
            value = np.asarray([value])
            return self.get_dose_constraint(value)
        elif key[0] == 'V':
            value = np.asarray([value])
            return self.get_volume_constrain(value)
        elif value == 'min':
            return self.stats[2]
        elif value == 'mean':
            return self.stats[1]
        elif value == 'max':
            return self.stats[0]
        elif key == 'HI':
            # calculating homogeneity index
            D1 = self.get_dose_constraint(1)
            D99 = self.get_dose_constraint(99)
            HI = (D1 - D99) / value
            return HI
        elif key[0] == 'T':
            # total volume
            return self.get_volume()

    def get_dose_constraint(self, volume):
        """ Return the maximum dose (in cGy) that a specific volume (in percent)
            receives. i.e. D90, D20.

            :param volume: Volume constrain in %
            :return: Dose in cGy
            """
        return float(self.fd(volume))

    def get_dose_constrain_cc(self, volumecc):
        """ Return the maximum dose (in cGy) that a specific volume in cc.
        :param volumecc: Volume in cc
        :return: Dose in cGy
        """

        return float(self.fd_cc(volumecc))

    def get_volume_constrain(self, dose):
        """ Return the volume (in percent) of the structure that receives at
            least a specific dose in cGy.
            i.e. V100, V150.
            :param dose:  Dose value in cGy
            :return: Percentage volume constrain (%)
            """

        return float(self.fv(dose))

    def get_volume_constrain_cc(self, dose):
        """ Return the volume (in cc) of the structure that receives at
            least a specific dose in cGy. i.e. Vcc, Vcc.
            :param dose:  Dose value in cGy
            :return: Volume in cc
            """
        return float(self.fv_cc(dose))

    def get_volume(self):
        return self.data['data'][0]


class Scoring(object):
    def __init__(self, rd_file, rs_file, rp_file, constrain, score, criteria=None, calculation_options=None):
        """
            Scoring class to encapsulate methods to extract constrains from DICOM-RP/RS/RD
        :param rd_file: DICOM-RT Dose file path
        :param rs_file: DICOM-RT Structures file path
        :param rp_file: DICOM-RT Plan file path
        :param constrain: Constrain dict {"Structure Name": {Metric:Value}}
        :param score: Scores dict {"Structure Name": {Metric: [type, (constrain_min, constrain_max), (point_min, point_max)}
        :param criteria: Scores DataFrame
        """
        # TODO debug importing RP files
        # self.rt_plan = ScoringDicomParser(filename=rp_file).GetPlan()
        self.structures = ScoringDicomParser(filename=rs_file).GetStructures()
        self.rtdose = ScoringDicomParser(filename=rd_file)
        self.constrains = dict((k.upper(), v) for k, v in constrain.items())
        self.scores = dict((k.upper(), v) for k, v in score.items())
        self.dvhs = {}
        self.constrains_values = {}
        self.score_result = {}
        self.score = 0
        self.criteria = criteria
        self.is_dicom_dvh = False
        self.calculation_options = calculation_options

    # @lazyproperty
    @property
    def scoring_result(self):
        return self.calc_score()

    def set_constrains_values(self, values_dict):
        self.constrains_values = values_dict

    def get_total_score(self):
        if self.dvhs:
            self.score = pd.DataFrame(self.scoring_result).sum().sum()
            return self.score

    def get_report_df(self):
        """
            Save detailed scoring results on spreadsheet
        :param out_file: File path to excel report (*.xls file)
        :param banner_path: Path the competition banner *.png
        """
        if self.dvhs:
            score_results = pd.DataFrame(self.scoring_result).T
            constrains_results = pd.DataFrame(self.constrains_values).T
            self.criteria.index = [name.upper() for name in self.criteria.index]

            # saving results report on spreadsheet
            s_names = self.criteria.index.unique()
            report = []
            for name in s_names:
                sc_tmp = self.criteria.loc[[name]]
                ctr_tmp = constrains_results.loc[[name]].dropna(axis=1)
                score_tmp = score_results.loc[[name]].dropna(axis=1)
                for res in ctr_tmp.columns:
                    mask = sc_tmp['constrain'] == res
                    tmp = sc_tmp.loc[mask].copy()
                    tmp['Result'] = ctr_tmp[res].values[0]
                    tmp['Raw Score'] = score_tmp[res].values[0]
                    report.append(tmp)

            df_report = pd.concat(report)
            df_report['Performance'] = df_report['Raw Score'] / df_report['Max Score']
            return df_report
        else:
            print('You need to set DVH data first!')

    def save_score_results(self, out_file, banner_path=None, report_header=''):
        """
            Save detailed scoring results on spreadsheet
        :param out_file: File path to excel report (*.xls file)
        :param banner_path: Path the competition banner *.png
        """
        df_report = self.get_report_df()
        self.save_formatted_report(df_report, out_file, banner_path, report_header)

    @staticmethod
    def save_formatted_report(df, out_file, banner_path=None, report_header='', io=None):

        """
            Save an formated report using pandas and xlsxwriter
        :param df: Results dataframe
        :param out_file: filename path
        :param banner_path: banner path
        """
        start_row = 31
        number_rows = len(df.index)
        writer = pd.ExcelWriter(out_file, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='report', startrow=start_row)

        # Get access to the workbook and sheet
        workbook = writer.book
        worksheet = writer.sheets['report']

        # Reduce the zoom a little
        worksheet.set_zoom(65)
        # constrain_fmt = workbook.add_format({'align': 'center'})
        constrain_fmt = workbook.add_format({'align': 'center'})

        # # Total formatting
        number_format = workbook.add_format({'align': 'right', 'num_format': '0.00'})
        # # Total percent format
        total_percent_fmt = workbook.add_format({'align': 'right', 'num_format': '0.0%', 'bold': True})

        # Add a format. Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                                       'font_color': '#9C0006'})

        # Add a format. Green fill with dark green text.
        format2 = workbook.add_format({'bg_color': '#C6EFCE',
                                       'font_color': '#006100'})

        # Format the columns by width and include number formats

        # Structure name
        nr = number_rows + start_row
        sname = "A2:A{}".format(nr + 1)
        worksheet.set_column(sname, 24)
        # constrain
        constrain = "B2:B{}".format(nr + 1)
        worksheet.set_column(constrain, 20, constrain_fmt)

        # constrain value
        constrain_value = "C2:C{}".format(nr + 1)
        worksheet.set_column(constrain_value, 20, constrain_fmt)

        # constrain type
        constrain_type = "D2:D{}".format(nr + 1)
        worksheet.set_column(constrain_type, 20, constrain_fmt)

        worksheet.conditional_format(constrain_type, {'type': 'text',
                                                      'criteria': 'containing',
                                                      'value': 'upper',
                                                      'format': format1})

        # Highlight the bottom 5 values in Red
        worksheet.conditional_format(constrain_type, {'type': 'text',
                                                      'criteria': 'containing',
                                                      'value': 'lower',
                                                      'format': format2})

        # value low and high
        worksheet.set_column('E:I', 20, number_format)

        # Define our range for the color formatting
        color_range = "J2:J{}".format(nr + 1)
        worksheet.set_column(color_range, 20, total_percent_fmt)

        # Highlight the top 5 values in Green
        worksheet.conditional_format(color_range, {'type': 'data_bar'})

        # write total score rows
        total_fmt = workbook.add_format({'align': 'right', 'num_format': '0.00',
                                         'bold': True, 'bottom': 6})

        total_fmt_header = workbook.add_format({'align': 'right', 'num_format': '0.00',
                                                'bold': True, 'bottom': 6, 'bg_color': '#C6EFCE'})

        total_score = df['Raw Score'].sum()
        max_score = df['Max Score'].sum()
        performance = total_score / max_score

        worksheet.write_string(nr + 1, 5, "Max Score:", total_fmt)
        worksheet.write_string(nr + 1, 7, "Total Score:", total_fmt_header)

        # performance format
        performance_format = workbook.add_format(
            {'align': 'right', 'num_format': '0.0%', 'bold': True, 'bottom': 6, 'bg_color': '#C6EFCE'})

        cell_location = xl_rowcol_to_cell(nr + 1, 9)
        worksheet.write_number(cell_location, performance, performance_format)

        cell_location = xl_rowcol_to_cell(nr + 1, 6)
        # Get the range to use for the sum formula
        worksheet.write_number(cell_location, max_score, total_fmt)
        cell_location = xl_rowcol_to_cell(nr + 1, 8)
        worksheet.write_number(cell_location, total_score, total_fmt_header)

        # SAVE BANNER
        if banner_path is not None:
            options = {'x_scale': 0.87}
            worksheet.insert_image('A1', banner_path, options=options)

        # adding participant header
        # Create a format to use in the merged range.
        merge_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'font_size': 15,
        })

        # Merge 3 cells.
        worksheet.merge_range('A31:J31', report_header, merge_format)

        writer.save()

    def set_dvh_data(self, dvh_filepath):
        """
            Set DVH data from *.dvh files
        :param dvh_filepath:
        """
        dvh_obj = load(dvh_filepath)
        self.dvhs = dvh_obj['DVH']
        # All keys Upper
        self.dvhs = dict((k.upper(), v) for k, v in self.dvhs.items())
        self._set_constrains_values()

    def set_dicom_dvh_data(self):
        """
            Getting DHV data from RT-DOSE file or calculate it.
        """
        self.is_dicom_dvh = True

        dvhs = self.rtdose.GetDVHs()
        temp_dvh = {}
        for key, structure in self.structures.items():
            tmp = dvhs[key]
            tmp['key'] = key
            tmp['name'] = structure['name']
            temp_dvh[structure['name']] = dvhs[key]

        self.dvhs = dict((k.upper(), v) for k, v in temp_dvh.items())
        self._set_constrains_values()

    def _set_constrains_values(self):
        """
            Set constrains data using constrains-score protocol
        """

        for key, values in self.constrains.items():
            if key == 'GLOBAL MAX':
                max_dose = self.rtdose.global_max
                self.dvhs['GLOBAL MAX'] = {'data': np.asarray([1, 100]),
                                           'max': max_dose,
                                           'mean': 0,
                                           'min': 0,
                                           'scaling': 1.0}
                dvh_values = self.dvhs[key]

            else:
                dvh_values = self.dvhs[key]

            # dvh_values = self.dvhs[key]
            # print(dvh_values)

            dvh_metrics = DVHMetrics(dvh_values)
            values_constrains = {}
            for k in values.keys():
                try:
                    ct = dvh_metrics.eval_constrain(k, values[k])
                    values_constrains[k] = ct
                    if k == 'CI':
                        if self.is_dicom_dvh:
                            ct = self.calc_conformity(key, values[k])
                        else:
                            nk = self.dvhs[key]['key']
                            ct = self.get_conformity(nk, values[k])
                        values_constrains[k] = ct
                    if k == 'Dmax_position':
                        nk = self.dvhs[key]['key']
                        values_constrains[k] = self.check_dmax_inside(self.structures[nk]['name'], self.dvhs)
                    if key == 'GLOBAL MAX':
                        values_constrains[k] = self.rtdose.global_max
                except:
                    nk = self.dvhs[key]['key']
                    sname = self.structures[nk]['name']
                    txt = 'error in constrain: %s value %1.3f on structure %s' % (k, values[k], sname)
                    logger.exception(txt)

            self.constrains_values[key] = values_constrains

    def calc_score(self):
        for key, values in self.scores.items():
            struc = self.constrains_values[key]
            struc_score = {}
            for k, val in struc.items():
                score_values = values[k]
                sc = self.score_function(val, score_values[1], score_values[2], score_values[0])
                struc_score[k] = sc
            self.score_result[key] = struc_score
        return self.score_result

    def get_conformity(self, nk, value):

        stmp = Structure(self.structures[nk], self.calculation_options)
        cindex = stmp.calc_conformation_index(self.rtdose, value)
        return cindex

    def _get_conformity(self, k, values):

        """
            Calculates CI using only DVH curves from TPS.
        :param k: Structure name
        :param values: Value in cGy to calculate CV

        :return:
        """
        max_volume_key = max(self.dvhs, key=lambda i: self.dvhs[i]['data'][0])

        metrics = DVHMetrics(self.dvhs[max_volume_key])

        PITV = metrics.get_volume_constrain_cc(values)
        target_metrics = DVHMetrics(self.dvhs[k])
        CV = target_metrics.get_volume_constrain_cc(values)
        TV = target_metrics.get_volume()
        CI = CV ** 2 / (TV * PITV)

        return CI

    def calc_conformity(self, k, values):

        """
            Calculates CI using only DVH curves from TPS.
        :param k: Structure name
        :param values: Value in cGy to calculate CV

        :return:
        """
        max_volume_key = max(self.dvhs, key=lambda i: self.dvhs[i]['data'][0])

        metrics = DVHMetrics(self.dvhs[max_volume_key])

        PITV = metrics.get_volume_constrain_cc(values)
        target_metrics = DVHMetrics(self.dvhs[k])
        CV = target_metrics.get_volume_constrain_cc(values)
        TV = target_metrics.get_volume()
        CI = CV ** 2 / (TV * PITV)

        return CI

    @staticmethod
    def check_dmax_inside(structure_name, dvhs):
        max_doses = pd.DataFrame([(k, v['max']) for k, v in dvhs.items()])
        mask = max_doses[1] == max_doses[1].max()
        volumes_dmax = max_doses.loc[mask][0].values
        return structure_name in volumes_dmax

    @staticmethod
    def score_function(val, values, points_score, constrain_type):
        """
            function to score each constrain from DVH
        :param val: Value to be scored
        :param values: min and max bounds
        :param points_score: min and max score points
        :param constrain_type: 'lower_constrain','upper constrain'
        :return:
        """
        if constrain_type == 'upper':
            points_score = points_score[::-1]
            return np.interp(val, values, points_score)
        if constrain_type == 'lower':
            return np.interp(val, values, points_score)
        if constrain_type == 'Dmax_position':
            return val * max(points_score)


class Participant(object):
    # TODO save important data a
    def __init__(self, rp_file, rs_file, rd_file, dvh_file='', calculation_options=None):
        """
            Class to encapsulate all plan participant planning data to eval using pyplanscoring app
        :param rp_file: path to DICOM-RTPLAN file
        :param rs_file: path to DICOM-STRUCTURE file
        :param rd_file: path to DICOM-DOSE file
        """
        self.rp_file = rp_file
        self.rs_file = rs_file
        self.rd_file = rd_file
        self.rd_dcm = ScoringDicomParser(filename=rd_file)
        self.rs_dcm = ScoringDicomParser(filename=rs_file)
        self.rp_dcm = ScoringDicomParser(filename=rp_file)
        self.dvh_file = dvh_file
        self.tps_info = ''
        self.plan_data = {}
        self.score_obj = None
        self.participant_name = ''
        self.structure_names = []
        self.calculation_options = calculation_options

    def set_participant_data(self, participant_name):

        self.participant_name = participant_name
        # TODO debug import RP files.
        # self.plan_data = self.rp_dcm.GetPlan()
        self.tps_info = self.rd_dcm.get_tps_data()

    def _save_dvh_fig(self, calc_dvhs, dest):
        p = os.path.splitext(dest)
        _, filename = os.path.split(dest)

        fig_name = p[0] + '_RD_calc_' + 'DVH.png'

        fig, ax = plt.subplots()
        fig.set_figheight(12)
        fig.set_figwidth(20)
        structures = self.rs_dcm.GetStructures()
        for key, structure in structures.items():
            sname = structure['name']
            if sname in self.structure_names:
                ax.plot(calc_dvhs[sname]['data'] / calc_dvhs[sname]['data'][0] * 100,
                        label=sname, linewidth=2.0, color=np.array(structure['color'], dtype=float) / 255)
                ax.legend(loc=7, borderaxespad=-5)

        ax.set_ylabel('Vol (%)')
        ax.set_xlabel('Dose (cGy)')
        ax.set_title(filename)
        fig.savefig(fig_name, format='png', dpi=100)

    def _save_dvh(self, structure_names):
        self.structure_names = structure_names
        if not self.dvh_file:
            p = os.path.splitext(self.rd_file)
            self.dvh_file = p[0] + '.dvh'
            # if not os.path.exists(self.dvh_file):
            cdvh = calc_dvhs_upsampled(self.participant_name, self.rs_file, self.rd_file,
                                       structure_names,
                                       out_file=self.dvh_file,
                                       calculation_options=self.calculation_options)
            self._save_dvh_fig(cdvh, self.rd_file)

    def eval_score(self, constrains_dict, scores_dict, criteria_df, calculation_options):

        self.score_obj = Scoring(self.rd_file,
                                 self.rs_file,
                                 self.rp_file,
                                 constrains_dict,
                                 scores_dict,
                                 criteria_df,
                                 calculation_options=calculation_options)

        if calculation_options['use_tps_dvh']:
            self.score_obj.set_dicom_dvh_data()
        else:
            self._save_dvh(criteria_df.index.unique())
            self.score_obj.set_dvh_data(self.dvh_file)

        return self.score_obj.get_total_score()

    def get_score_report(self, banner_path, report_header, io):
        rep = self.score_obj.get_report_df()
        self.score_obj.save_formatted_report(rep,
                                             out_file='',
                                             banner_path=banner_path,
                                             report_header=report_header,
                                             io=io)

    def save_score(self, out_file, banner_path=None, report_header=''):
        self.score_obj.save_score_results(out_file, banner_path, report_header)


if __name__ == '__main__':
    participant_name = 'Rense Lamsma'
    dicom_dir = r'/media/victor/TOURO Mobile/COMPETITION 2017/plans/Rense Lamsma - IMPT'
    rp = r'/media/victor/TOURO Mobile/COMPETITION 2017/plans/Rense Lamsma - IMPT/RP1.2.752.243.1.1.20170314143351969.2000.27546.dcm'
    rs = r'/media/victor/TOURO Mobile/COMPETITION 2017/plans/Rense Lamsma - IMPT/RS1.2.752.243.1.1.20170228161420610.1600.70016.dcm'
    rd = r'/media/victor/TOURO Mobile/COMPETITION 2017/plans/Rense Lamsma - IMPT/RD1.2.752.243.1.1.20170314143351971.9000.67615.dcm'
    f = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/Scoring Criteria.txt'

    truth, files_data = get_participant_folder_data(participant_name, dicom_dir)

    constrains, scores, criteria = read_scoring_criteria(f)

    calculation_options = dict()
    calculation_options['end_cap'] = 0.5
    calculation_options['use_tps_dvh'] = False
    calculation_options['up_sampling'] = True
    calculation_options['maximum_upsampled_volume_cc'] = 100.0
    calculation_options['voxel_size'] = 0.5

    print('------------- Calculating DVH and score --------------')

    participant = Participant(rp, rs, rd, calculation_options=calculation_options)
    participant.set_participant_data(participant_name)
    val = participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria,
                                 calculation_options=calculation_options)
    print(val)

    print('Plan Score: %1.3f' % val)
    out_file = os.path.join(dicom_dir, participant_name + '_plan_scoring_report.xls')
    banner_path = '/home/victor/Dropbox/Plan_Competition_Project/scoring_report/2017 Plan Comp Banner.jpg'
    participant.save_score(out_file, banner_path=banner_path)
    print('Report saved: %s' % out_file)
    input("Press enter to exit.")
