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
from pyplanscoring.dicomparser import ScoringDicomParser, lazyproperty
from pyplanscoring.dvhcalc import load

logger = logging.getLogger('scoring')


# from interpolation.splines import CubicSpline, LinearSpline
#
#
# def get_cubic_interp(dose_range, cdvh):
#     a = np.asarray([dose_range[0]])
#     b = np.asarray([dose_range[-1]])
#     orders = np.asarray([len(cdvh)])
#     interp = CubicSpline(a, b, orders, cdvh)
#
#     return interp


def get_dvh_files(root_path):
    dvh_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(root_path)
                 for name in files
                 if name.endswith('.dvh')]
    return dvh_files


def get_participant_folder_data(participant_name, root_path):
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


class DVHMetrics(object):
    def __init__(self, dvh):
        # Todo - interpolate DVH using 10 cGy bins
        vpp = dvh['data'] * 100 / dvh['data'][0]
        self.volume_pp = np.append(vpp, 0)  # add 0 volume to interpolate
        # self.volume_pp = vpp
        self.scaling = dvh['scaling']
        # self.dose_axis = dvh['dose_axis'] * self.scaling
        self.dose_axis = np.arange(len(dvh['data']) + 1) * self.scaling
        self.volume_cc = np.append(dvh['data'], 0)
        # self.volume_cc = dvh['data']
        self.stats = (dvh['max'], dvh['mean'], dvh['min'])
        self.data = dvh

        self.fv = itp.interp1d(self.dose_axis, self.volume_pp, fill_value='extrapolate')  # pp
        self.fv_cc = itp.interp1d(self.dose_axis, self.volume_cc, fill_value='extrapolate')  # pp
        self.fd = itp.interp1d(self.volume_pp, self.dose_axis, fill_value='extrapolate')  # pp
        self.fd_cc = itp.interp1d(self.volume_cc, self.dose_axis, fill_value='extrapolate')  # cc
        #
        # self.fv = get_cubic_interp(self.dose_axis, self.volume_pp)
        # self.fv_cc = get_cubic_interp(self.dose_axis, self.volume_cc)
        # self.fd = get_cubic_interp(self.volume_pp, self.dose_axis)
        # self.fd_cc = get_cubic_interp(self.volume_cc, self.dose_axis)


        # self.fv = itp.interp1d(self.dose_axis, self.volume_pp, kind='cubic')  # , fill_value='extrapolate')  # pp
        # self.fv_cc = itp.interp1d(self.dose_axis, self.volume_pp, kind='cubic')  # ,, fill_value='extrapolate')  # pp
        # self.fd = itp.interp1d(self.volume_pp, self.dose_axis, kind='cubic')  # ,, fill_value='extrapolate')  # pp
        # self.fd_cc = itp.interp1d(self.volume_cc, self.dose_axis, kind='cubic')  # ,, fill_value='extrapolate')  # cc

    def eval_constrain(self, key, value):

        # TODO refactor using dictionary
        if key == 'Global Max':
            return self.data['max']

        if key == 'Dmax_position' and value == 'max':
            return self.stats[0]
        elif key[0] == 'Dcc':
            value = np.asarray([value])
            return self.GetDoseConstraintCC(value)
        elif key[0] == 'D':
            value = np.asarray([value])
            return self.GetDoseConstraint(value)
        elif key[0] == 'V':
            value = np.asarray([value])
            return self.GetVolumeConstraint(value)
        elif value == 'min':
            return self.stats[2]
        elif value == 'mean':
            return self.stats[1]
        elif value == 'max':
            return self.stats[0]
        elif key == 'HI':
            D1 = self.GetDoseConstraint(1)
            D99 = self.GetDoseConstraint(99)
            HI = (D1 - D99) / value
            return HI
        elif key[0] == 'T':
            return self.get_volume()

    def GetDoseConstraint(self, volume):
        """ Return the maximum dose (in cGy) that a specific volume (in percent)
            receives. i.e. D90, D20."""

        return float(self.fd(volume))

    def GetDoseConstraintCC(self, volumecc):
        """ Return the maximum dose (in cGy) that a specific volume in cc."""

        return float(self.fd_cc(volumecc))

    def GetVolumeConstraint(self, dose):

        """ Return the volume (in percent) of the structure that receives at
            least a specific dose in cGy. i.e. V100, V150. fix by Victor Gabriel"""

        return float(self.fv(dose))

    def GetVolumeConstraintCC(self, dose):

        """ Return the volume (in cc) of the structure that receives at
            least a specific dose in cGy. i.e. V100, V150. fix by Victor Gabriel"""

        return float(self.fv_cc(dose))

    def get_volume(self):
        return self.data['data'][0]


class Scoring(object):
    def __init__(self, rd_file, rs_file, rp_file, constrain, score, criteria=None):
        self.rt_plan = ScoringDicomParser(filename=rp_file).GetPlan()
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

    @lazyproperty
    def scoring_result(self):
        return self.calc_score()

    def get_total_score(self):
        if self.dvhs:
            self.score = pd.DataFrame(self.scoring_result).sum().sum()
            return self.score

    def save_score_results(self, out_file, banner_path=None):
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
            #
            save_formatted_report(df_report, out_file, banner_path)
        else:
            print('You need to set DVH data first!')

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

        stmp = Structure(self.structures[nk])
        cindex = stmp.calc_conformation_index(self.rtdose, value)
        return cindex

    def calc_conformity(self, k, values):

        """
            Calculates CI using only DVH curves from TPS.
        :param k: Structure name
        :param values: Value in cGy to calculate CV

        :return:
        """
        max_volume_key = max(self.dvhs, key=lambda i: self.dvhs[i]['data'][0])

        metrics = DVHMetrics(self.dvhs[max_volume_key])

        PITV = metrics.GetVolumeConstraintCC(values)
        target_metrics = DVHMetrics(self.dvhs[k])
        CV = target_metrics.GetVolumeConstraintCC(values)
        TV = target_metrics.get_volume()
        CI = CV ** 2 / (TV * PITV)

        return CI



        # CI = CV * CV / (TV * PITV)

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
    def __init__(self, rp_file, rs_file, rd_file, dvh_file='', upsample='', end_cap=False):
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
        self.up_sample = upsample
        self.end_cap = end_cap

    def set_participant_data(self, participant_name):

        self.participant_name = participant_name
        self.plan_data = self.rp_dcm.GetPlan()
        self.tps_info = self.rd_dcm.get_tps_data()

    def _save_dvh_fig(self, calc_dvhs, dest):
        p = os.path.splitext(dest)
        _, filename = os.path.split(dest)

        fig_name = p[0] + '_RD_calc_' + self.up_sample + 'DVH.png'

        if self.end_cap:
            fig_name = p[0] + '_RD_calc_' + self.up_sample + '_END_CAPPED_DVH.png'

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
            self.dvh_file = p[0] + self.up_sample + '.dvh'
            if self.end_cap:
                self.dvh_file = p[0] + self.up_sample + 'end_cap.dvh'
            if not os.path.exists(self.dvh_file):
                cdvh = calc_dvhs_upsampled(self.participant_name, self.rs_file, self.rd_file,
                                           structure_names,
                                           out_file=self.dvh_file,
                                           end_cap=self.end_cap, upsample=self.up_sample)
                self._save_dvh_fig(cdvh, self.rd_file)

    def eval_score(self, constrains_dict, scores_dict, criteria_df, dicom_dvh=False):
        self.score_obj = Scoring(self.rd_file, self.rs_file, self.rp_file, constrains_dict, scores_dict, criteria_df)
        if dicom_dvh:
            self.score_obj.set_dicom_dvh_data()
        else:
            self._save_dvh(criteria_df.index.unique())
            self.score_obj.set_dvh_data(self.dvh_file)

        return self.score_obj.get_total_score()

    def save_score(self, out_file, banner_path=None):
        self.score_obj.save_score_results(out_file, banner_path)


def save_formatted_report(df, out_file, banner_path=None):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    number_rows = len(df.index)
    writer = pd.ExcelWriter(out_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='report')

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
    sname = "A2:A{}".format(number_rows + 1)
    worksheet.set_column(sname, 24)
    # constrain
    constrain = "B2:B{}".format(number_rows + 1)
    worksheet.set_column(constrain, 20, constrain_fmt)

    # constrain value
    constrain_value = "C2:C{}".format(number_rows + 1)
    worksheet.set_column(constrain_value, 20, constrain_fmt)

    # constrain type
    constrain_type = "D2:D{}".format(number_rows + 1)
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
    color_range = "J2:J{}".format(number_rows + 1)
    worksheet.set_column(color_range, 20, total_percent_fmt)

    # Highlight the top 5 values in Green

    worksheet.conditional_format(color_range, {'type': 'data_bar'})

    # write total score rows
    total_fmt = workbook.add_format({'align': 'right', 'num_format': '0.00',
                                     'bold': True, 'bottom': 6})
    # Determine where we will place the formula
    for i in [6, 8]:
        cell_location = xl_rowcol_to_cell(number_rows + 1, i)
        # Get the range to use for the sum formula
        start_range = xl_rowcol_to_cell(1, i)
        end_range = xl_rowcol_to_cell(number_rows, i)
        # Construct and write the formula
        formula = "=SUM({:s}:{:s})".format(start_range, end_range)
        worksheet.write_formula(cell_location, formula, total_fmt)

    worksheet.write_string(number_rows + 1, 5, "Max Score:", total_fmt)
    worksheet.write_string(number_rows + 1, 7, "Total Score:", total_fmt)

    # performance format
    performance_format = workbook.add_format({'align': 'right', 'num_format': '0.0%', 'bold': True, 'bottom': 6})
    percent_formula = "=I{0}/G{0}".format(number_rows + 2)
    worksheet.write_formula(number_rows + 1, 9, percent_formula, performance_format)

    # SAVE BANNER
    if banner_path is not None:
        worksheet.insert_image('K1', banner_path)

    writer.save()


if __name__ == '__main__':
    from pyplanscoring.dosimetric import read_scoring_criteria

    rd = r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/dicom_files/RD.1.2.246.352.71.7.584747638204.1758320.20170210154830.dcm'
    rs = r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/dicom_files/RS.1.2.246.352.71.4.584747638204.248648.20170209152429.dcm'
    rp = r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/dicom_files/RP.1.2.246.352.71.5.584747638204.955801.20170210152428.dcm'
    participant_name = 'test_CI'

    f_2017 = r'/home/victor/Dropbox/Plan_Competition_Project/scoring_report/Scoring Criteria.txt'
    constrains, scores, criteria = read_scoring_criteria(f_2017)

    print('------------- Calculating DVH and score --------------')
    participant = Participant(rp, rs, rd, upsample='_up_sampled', end_cap=True)
    participant.set_participant_data(participant_name)
    val = participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria, dicom_dvh=True)

    # # DVH DATA
    # dvhs = deepcopy(participant.score_obj.dvhs)
    #
    # max_volume_key = max(dvhs, key=lambda i: dvhs[i]['data'][0])
    #
    # metrics = DVHMetrics(dvhs[max_volume_key])
    #
    # val_test = 5320  # cGy
    #
    # PITV = metrics.GetVolumeConstraintCC(val_test)
    #
    # # calculating coverage volume
    # target_metrics = DVHMetrics(dvhs['PTV56'])
    # CV = target_metrics.GetVolumeConstraintCC(val_test)
    # TV = target_metrics.get_volume()
    # CI = CV ** 2 / (TV * PITV)
    #
    # # print('Plan Score: %1.3f' % val)
    # # out_file = os.path.join(dicom_dir, 'plan_scoring_report.xls')
    # # banner_path = os.path.join(wd, '2017 Plan Comp Banner.jpg')
    # # participant.save_score(out_file, banner_path=banner_path)
    # # print('Report saved: %s' % out_file)
    # app = 0.8460
    # PLANIQ = 0.8360
    # CI_from_eclipse_DVH = 0.844
