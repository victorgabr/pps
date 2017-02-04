import logging
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as itp
from joblib import Parallel
from joblib import delayed

from dev.dvhcalculation import calc_dvhs_upsampled, Structure
from dicomparser import ScoringDicomParser, lazyproperty
from dosimetric import read_scoring_criteria
from dvhcalc import load, calc_dvhs

logger = logging.getLogger('scoring')


# pd.options.display.float_format = '{:.2f}'.format


def get_dvh_files(root_path):
    dvh_files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(root_path)
                 for name in files
                 if name.endswith('.dvh')]
    return dvh_files


def get_competition_data(root_path):
    files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
             name.endswith(('.dcm', '.DCM'))]

    report_files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
                    name.endswith(('.pdf', '.PDF'))]

    filtered_files = OrderedDict()
    for f in files:
        try:
            obj = ScoringDicomParser(filename=f)
            rt_type = obj.GetSOPClassUID()
            if rt_type == 'rtdose':
                tmp = f.split(os.path.sep)[-2].split()
                name = tmp[0].split('-')[0]
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
            if rt_type == 'rtplan':
                tmp = f.split(os.path.sep)[-2].split()
                name = tmp[0].split('-')[0]
                participant_data = [name, rt_type]
                filtered_files[f] = participant_data
        except:
            logger.exception('Error in file %s' % f)

    data = pd.DataFrame(filtered_files).T

    plan_iq_scores = []
    for f in report_files:
        p, r = os.path.split(f)
        s = re.findall('\d+\.\d+', r)
        plan_iq_scores.append(s * 2)

    plan_iq_scores = np.ravel(plan_iq_scores).astype(float)
    data['plan_iq_scores'] = plan_iq_scores

    return data.reset_index()


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
        self.fv_cc = itp.interp1d(self.dose_axis, self.volume_pp, fill_value='extrapolate')  # pp
        self.fd = itp.interp1d(self.volume_pp, self.dose_axis, fill_value='extrapolate')  # pp
        self.fd_cc = itp.interp1d(self.volume_cc, self.dose_axis, fill_value='extrapolate')  # cc

    def eval_constrain(self, key, value):
        # TODO refactor using dictionary
        if key == 'Global Max':
            return self.data['max']

        if key == 'Dmax_position' and value == 'max':
            return self.stats[0]
        elif key[0] == 'Dcc':
            return self.GetDoseConstraintCC(value)
        elif key[0] == 'D':
            return self.GetDoseConstraint(value)
        elif key[0] == 'V':
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

        """ Return the volume (in percent) of the structure that receives at
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

    @lazyproperty
    def scoring_result(self):
        return self.calc_score()

    def total_score(self):
        if self.dvhs:
            self.score = pd.DataFrame(self.scoring_result).sum().sum()
            return self.score

    def save_score_results(self, out_file):
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
            df_report.to_excel(out_file)

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
        dvhs = self.rtdose.GetDVHs()
        temp_dvh = {}
        for key, structure in self.structures.items():
            tmp = dvhs[key]
            tmp['key'] = key
            tmp['name'] = structure['name']
            temp_dvh[structure['name']] = dvhs[key]
        self.dvhs = temp_dvh

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


class EvalCompetition(object):
    def __init__(self, root_path, rs_file, constrains, scores):
        self.root_path = root_path
        self.rs_file = rs_file
        self.constrains = constrains
        self.scores = scores
        self.comp_data = None
        self.dvh_files = []
        self.results = []

    def set_data(self):
        self.comp_data = get_competition_data(self.root_path)
        self.dvh_files = [os.path.join(root, name) for root, dirs, files in os.walk(self.root_path) for name in files if
                          name.endswith('.dvh')]

    def calc_scores(self):
        res = Parallel(n_jobs=-1, verbose=11)(
            delayed(self.get_score)(dvh_file) for dvh_file in self.dvh_files)
        self.results = res
        return res

    def get_score(self, dvh_file):
        rd_file, rp_file, name = self.get_dicom_data(self.comp_data, dvh_file)
        try:
            obj = Scoring(rd_file, self.rs_file, rp_file, self.constrains, self.scores)
            obj.set_dvh_data(dvh_file)
            print('Score:', name, obj.total_score)
            return name, obj.total_score
        except:
            logger.exception('Error in file: %s' % dvh_file)
            try:
                obj = Scoring(rd_file, self.rs_file, rp_file, self.constrains, self.scores)
                obj.set_dicom_dvh_data()
                print('Score:', name, obj.total_score)
                return name, obj.total_score
            except:
                logger.exception('No DVH data in file  %s' % rd_file)
                return rd_file

    @staticmethod
    def get_dicom_data(data, dvh_file):
        try:
            dvh = load(dvh_file)
            name = dvh['participant']
            p_files = data[data[0] == name].set_index(1)
            rd_file = p_files.ix['rtdose']['index']
            rp_file = p_files.ix['rtplan']['index']
            return rd_file, rp_file, name
        except:
            logger.exception('error on file %s' % dvh_file)


class Participant(object):
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
        # TODO IMPLEMENT CALCULATING ONLY SCORED STRUCTURES DVH ?
        self.participant_name = participant_name
        self.plan_data = self.rp_dcm.GetPlan()
        self.tps_info = self.rd_dcm.get_tps_data()
        structures = self.rs_dcm.GetStructures()
        for key, structure in structures.items():
            self.structure_names.append(structure['name'])

        if not self.dvh_file:
            p = os.path.splitext(self.rd_file)
            self.dvh_file = p[0] + self.up_sample + '.dvh'
            if self.end_cap:
                self.dvh_file = p[0] + self.up_sample + 'end_cap.dvh'
            if not os.path.exists(self.dvh_file):
                if self.up_sample:
                    cdvh = calc_dvhs_upsampled(participant_name, self.rs_file, self.rd_file, out_file=self.dvh_file,
                                               end_cap=self.end_cap)
                    self._save_dvh_fig(cdvh, self.rd_file)
                else:
                    cdvh = calc_dvhs(participant_name, self.rs_file, self.rd_file, out_file=self.dvh_file)
                    self._save_dvh_fig(cdvh, self.rd_file)

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
            ax.plot(calc_dvhs[sname]['data'] / calc_dvhs[sname]['data'][0] * 100,
                    label=sname, linewidth=2.0, color=np.array(structure['color'], dtype=float) / 255)
            ax.legend(loc=7, borderaxespad=-5)
            ax.set_ylabel('Vol (%)')
            ax.set_xlabel('Dose (cGy)')
            ax.set_title(filename)
            fig.savefig(fig_name, format='png', dpi=100)

    def eval_score(self, constrains_dict, scores_dict, criteria_df):
        self.score_obj = Scoring(self.rd_file, self.rs_file, self.rp_file, constrains_dict, scores_dict, criteria_df)
        self.score_obj.set_dvh_data(self.dvh_file)
        return self.score_obj.total_score()

    def save_score(self, out_file):
        self.score_obj.save_score_results(out_file)


if __name__ == '__main__':
    rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/RS.1.2.246.352.71.4.584747638204.248648.20170123083029.dcm'
    rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/RD.1.2.246.352.71.7.584747638204.1750110.20170123082607.dcm'
    rp = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/RP.1.2.246.352.71.5.584747638204.952069.20170122155706.dcm'

    f_2017 = r'/home/victor/Dropbox/Plan_Competition_Project/competition_2017/All Required Files - 23 Jan2017/PlanIQ Criteria TPS PlanIQ matched str names - TXT Fromat - Last mod Jan23.txt'

    constrains, scores, criteria = read_scoring_criteria(f_2017)

    obj = Participant(rp, rs_file, rd_file, upsample='_up_sampled_', end_cap=True)
    obj.set_participant_data('Ahmad')
    val1 = obj.eval_score(constrains_dict=constrains, scores_dict=scores)
    #
    # print('Radiation Knowledge Plan Competition - 2017 ')
    # print('Ahmad Score (no up-sampling):', val)
    print('Ahmad Score end capped:', val1)
